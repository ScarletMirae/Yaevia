"""
tests/experiment_d_position_audit.py — Experiment D: Page-Position / Sample-Position Bias Audit
================================================================================================
Diagnostic audit investigating systematic accuracy bias across sample positions #1 through #20
on both 128x128 baseline and 256x256 experimental representations.

Dataset: 18 students x 20 samples = 360 images.
Pipeline: Orientation norm -> Grayscale -> Gaussian blur -> Otsu -> Noise removal -> Baseline extract_roi()
          -> Shared ROI resized to 128x128 and 256x256 -> HOG (8100 vs 34596) -> KNN (K=5, Euclidean, distance-weighted).

Outputs: backend/tests/evaluation_results/page_position_audit/
  - position_accuracy_128_256.csv
  - correctness_matrix_128.csv
  - correctness_matrix_256.csv
  - prediction_matrix_128.csv
  - prediction_matrix_256.csv
  - position_visual_metrics.csv
  - position_confusion_analysis.csv
  - sample1_deep_audit.csv
  - student_position_effect.csv
  - resolution_position_transitions.csv
  - page_position_audit.json
  - contact_sheets/ (contact_sheet_position_01.png, contact_sheet_highest.png, contact_sheet_median.png)
"""

import os
import sys
import json
import csv
import sqlite3
import math
import numpy as np
import cv2
from collections import defaultdict
from skimage.feature import hog

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import (
    DB_PATH, DATASET_RAW_DIR,
    HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK, HOG_BLOCK_NORM,
    KNN_N_NEIGHBORS,
)
from preprocessing.image_processor import (
    normalize_orientation, convert_to_grayscale,
    apply_gaussian_blur, apply_otsu_threshold, remove_noise,
    extract_roi, resize_with_aspect_ratio,
)

SIZE_128 = (128, 128)
SIZE_256 = (256, 256)
K = KNN_N_NEIGHBORS  # 5
EPS = 1e-7

OUT_DIR = os.path.join(BASE_DIR, "tests", "evaluation_results", "page_position_audit")
CONTACT_DIR = os.path.join(OUT_DIR, "contact_sheets")
os.makedirs(CONTACT_DIR, exist_ok=True)


def extract_hog(image: np.ndarray):
    img_f = image.astype(np.float64) / 255.0 if image.dtype != np.float64 else image
    feat, vis = hog(
        img_f,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        visualize=True,
        feature_vector=True,
    )
    return feat, vis


def pairwise_dist(X):
    dot = np.dot(X, X.T)
    sq = np.diag(dot)
    dsq = np.maximum(0.0, sq[:, None] + sq[None, :] - 2 * dot)
    d = np.sqrt(dsq)
    np.fill_diagonal(d, 0.0)
    return d


def run_loocv(X, y, dist_matrix=None):
    n = len(X)
    y_a = np.array(y)
    if dist_matrix is None:
        dist_matrix = pairwise_dist(X)

    unique_classes = sorted(set(y))
    results = []

    for i in range(n):
        true_label = y_a[i]
        row = np.delete(dist_matrix[i], i)
        lbls = np.delete(y_a, i)

        sorted_idx = np.argsort(row)
        topk_idx = sorted_idx[:K]
        topk_dists = row[topk_idx]
        topk_labels = lbls[topk_idx]

        wk = 1.0 / np.maximum(topk_dists, EPS)
        total_w = wk.sum()

        vote_w = defaultdict(float)
        vote_c = defaultdict(int)
        for d_val, lbl, wv in zip(topk_dists, topk_labels, wk):
            vote_w[lbl] += wv
            vote_c[lbl] += 1

        sorted_votes = sorted(vote_w.items(), key=lambda x: -x[1])
        pred_label = sorted_votes[0][0]
        pred_share = sorted_votes[0][1] / total_w * 100.0
        second_share = sorted_votes[1][1] / total_w * 100.0 if len(sorted_votes) > 1 else 0.0

        true_neighbors = vote_c.get(true_label, 0)
        true_vote_share = vote_w.get(true_label, 0.0) / total_w * 100.0

        class_min = {}
        for cls in unique_classes:
            mask = (lbls == cls)
            class_min[cls] = float(row[mask].min()) if mask.any() else float("inf")

        results.append({
            "sample_idx": i,
            "true_label": true_label,
            "pred_label": pred_label,
            "is_correct": bool(pred_label == true_label),
            "pred_vote_share": round(float(pred_share), 2),
            "second_vote_share": round(float(second_share), 2),
            "margin": round(float(pred_share - second_share), 2),
            "true_neighbors": true_neighbors,
            "true_vote_share": round(float(true_vote_share), 2),
            "min_dist_pred": round(class_min.get(pred_label, 999.0), 4),
            "min_dist_true": round(class_min.get(true_label, 999.0), 4),
            "nearest_dist": round(float(topk_dists[0]), 4),
            "topk_labels": list(topk_labels),
            "topk_dists": [round(float(d), 4) for d in topk_dists],
        })

    return results, dist_matrix


def calc_shannon_entropy(labels_list):
    if not labels_list:
        return 0.0
    n = len(labels_list)
    counts = defaultdict(int)
    for l in labels_list:
        counts[l] += 1
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / n
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def create_contact_sheet(samples, title, save_path, cols=6, thumb_w=200, thumb_h=280):
    """
    Creates a contact sheet montage of samples (original images with labels).
    """
    n_samples = len(samples)
    rows = math.ceil(n_samples / cols)
    card_w = thumb_w
    card_h = thumb_h + 60
    sheet = np.full((rows * card_h + 50, cols * card_w, 3), 25, dtype=np.uint8)

    # Title header
    cv2.putText(sheet, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)

    for i, s in enumerate(samples):
        r_idx = i // cols
        c_idx = i % cols
        x_offset = c_idx * card_w
        y_offset = 50 + r_idx * card_h

        raw_path = s["raw_path"]
        img = cv2.imread(raw_path)
        if img is None:
            continue
        img = normalize_orientation(img)
        ih, iw = img.shape[:2]
        scale = min(thumb_w / iw, thumb_h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        # Place image centered in card image area
        dx = (card_w - nw) // 2
        dy = (thumb_h - nh) // 2
        sheet[y_offset + dy:y_offset + dy + nh, x_offset + dx:x_offset + dx + nw] = resized

        # Border around image
        c128_ok = s["loocv_128"]["is_correct"]
        c256_ok = s["loocv_256"]["is_correct"]
        border_color = (0, 255, 0) if (c128_ok and c256_ok) else ((0, 200, 255) if c256_ok else (0, 0, 255))
        cv2.rectangle(sheet, (x_offset + 2, y_offset + 2), (x_offset + card_w - 2, y_offset + card_h - 2), border_color, 1)

        # Text labels below thumbnail
        label_y = y_offset + thumb_h + 15
        student_short = s["student_name"][:18]
        cv2.putText(sheet, f"#{s['sample_num']:02d} {student_short}", (x_offset + 5, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        
        status_128 = "OK" if c128_ok else f"ERR->{s['loocv_128']['pred_label'][:8]}"
        color_128 = (0, 255, 0) if c128_ok else (100, 100, 255)
        cv2.putText(sheet, f"128: {status_128}", (x_offset + 5, label_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, color_128, 1, cv2.LINE_AA)

        status_256 = "OK" if c256_ok else f"ERR->{s['loocv_256']['pred_label'][:8]}"
        color_256 = (0, 255, 0) if c256_ok else (100, 100, 255)
        cv2.putText(sheet, f"256: {status_256}", (x_offset + 5, label_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, color_256, 1, cv2.LINE_AA)

    cv2.imwrite(save_path, sheet)
    print(f"  Contact sheet saved: {save_path}")


def run_experiment_d():
    print("=" * 80)
    print("EXPERIMENT D: PAGE-POSITION / SAMPLE-POSITION BIAS AUDIT")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Inspect DB Records & Ordering Verification
    # ------------------------------------------------------------------
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, student_name, student_id, mata_kuliah, original_filename, saved_filename, file_path, upload_timestamp "
        "FROM dataset ORDER BY student_name ASC, original_filename ASC, saved_filename ASC"
    ).fetchall()
    conn.close()

    n_records = len(rows)
    students = sorted(list(set(r["student_name"] for r in rows)))
    n_students = len(students)
    print(f"Loaded {n_records} records across {n_students} students.")

    # Ordering audit:
    print("\n[ORDERING AUDIT] Checking sample filename structures and ordering...")
    filename_patterns = defaultdict(int)
    for r in rows:
        fn = r["original_filename"] or ""
        if "MVIMG" in fn:
            filename_patterns["MVIMG_timestamp"] += 1
        elif "IMG_" in fn:
            filename_patterns["IMG_prefix"] += 1
        elif "WhatsApp" in fn:
            filename_patterns["WhatsApp_image"] += 1
        else:
            filename_patterns["Other_custom"] += 1

    for pat, count in filename_patterns.items():
        print(f"  Pattern: {pat:<20} Count: {count}/{n_records} ({count/n_records*100:.1f}%)")

    # ------------------------------------------------------------------
    # 2. Extract Features (128 & 256) & Measure Visual Characteristics
    # ------------------------------------------------------------------
    feat128_list = []
    feat256_list = []
    labels = []
    samples_data = []

    print("\nProcessing all 360 images through shared preprocessing...")
    for idx, r in enumerate(rows):
        db_id = r["id"]
        student = r["student_name"]
        filename = r["original_filename"] or ""
        raw_name = r["saved_filename"] or os.path.basename(r["file_path"])
        raw_path = os.path.join(DATASET_RAW_DIR, raw_name)

        if not os.path.exists(raw_path):
            print(f"  [ERROR] File not found: {raw_path}")
            continue

        img_bgr = cv2.imread(raw_path)
        if img_bgr is None:
            print(f"  [ERROR] Cannot read: {raw_path}")
            continue

        # Pipeline
        oriented = normalize_orientation(img_bgr)
        orig_h, orig_w = oriented.shape[:2]
        gray = convert_to_grayscale(oriented)
        blurred = apply_gaussian_blur(gray)
        binary = apply_otsu_threshold(blurred)
        denoised = remove_noise(binary)
        roi = extract_roi(denoised)
        roi_h, roi_w = roi.shape[:2]

        img128 = resize_with_aspect_ratio(roi, SIZE_128)
        img256 = resize_with_aspect_ratio(roi, SIZE_256)

        feat128, _ = extract_hog(img128)
        feat256, _ = extract_hog(img256)

        orig_area = orig_w * orig_h
        roi_area = roi_w * roi_h
        roi_cov = roi_area / max(1, orig_area)
        fg_pix_roi = int(np.count_nonzero(roi))
        fg_pix_denoised = int(np.count_nonzero(denoised))
        fg_ratio_orig = fg_pix_denoised / max(1, orig_area)
        fg_ratio_roi = fg_pix_roi / max(1, roi_area)
        fg_ratio_128 = float(np.count_nonzero(img128)) / (128 * 128)
        fg_ratio_256 = float(np.count_nonzero(img256)) / (256 * 256)
        orig_aspect = orig_w / max(1, orig_h)
        roi_aspect = roi_w / max(1, roi_h)
        hog_norm128 = float(np.linalg.norm(feat128))
        hog_norm256 = float(np.linalg.norm(feat256))

        sample_num = (len([s for s in samples_data if s["student_name"] == student]) + 1)

        sample_meta = {
            "sample_idx": len(samples_data),
            "db_id": db_id,
            "student_name": student,
            "filename": filename,
            "saved_filename": raw_name,
            "raw_path": raw_path,
            "upload_timestamp": r["upload_timestamp"],
            "sample_num": sample_num,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "orig_aspect": round(orig_aspect, 4),
            "roi_w": roi_w,
            "roi_h": roi_h,
            "roi_aspect": round(roi_aspect, 4),
            "roi_coverage": round(roi_cov, 4),
            "fg_pix_count": fg_pix_roi,
            "fg_ratio_orig": round(fg_ratio_orig, 4),
            "fg_ratio_roi": round(fg_ratio_roi, 4),
            "fg_ratio_128": round(fg_ratio_128, 4),
            "fg_ratio_256": round(fg_ratio_256, 4),
            "hog_norm_128": round(hog_norm128, 4),
            "hog_norm_256": round(hog_norm256, 4),
        }

        feat128_list.append(feat128)
        feat256_list.append(feat256)
        labels.append(student)
        samples_data.append(sample_meta)

    n = len(labels)
    X128 = np.array(feat128_list)
    X256 = np.array(feat256_list)

    # ------------------------------------------------------------------
    # 3. LOOCV KNN (128 and 256)
    # ------------------------------------------------------------------
    print("\nRunning LOOCV KNN (128x128)...")
    loocv128, dist128 = run_loocv(X128, labels)
    print("Running LOOCV KNN (256x256)...")
    loocv256, dist256 = run_loocv(X256, labels)

    for i in range(n):
        samples_data[i]["loocv_128"] = loocv128[i]
        samples_data[i]["loocv_256"] = loocv256[i]

    # ------------------------------------------------------------------
    # 4. Position-wise Accuracy & Metrics (Positions #1 .. #20)
    # ------------------------------------------------------------------
    print("\nComputing position-wise metrics for positions #1 .. #20...")
    position_data = []
    position_visual = []

    pos_samples = defaultdict(list)
    for s in samples_data:
        pos_samples[s["sample_num"]].append(s)

    for p in range(1, 21):
        s_list = pos_samples[p]
        cnt = len(s_list)
        corr_128 = sum(1 for s in s_list if s["loocv_128"]["is_correct"])
        corr_256 = sum(1 for s in s_list if s["loocv_256"]["is_correct"])
        acc_128 = round(corr_128 / cnt * 100, 2)
        acc_256 = round(corr_256 / cnt * 100, 2)
        delta_acc = round(acc_256 - acc_128, 2)

        votes_128 = [s["loocv_128"]["pred_vote_share"] for s in s_list]
        votes_256 = [s["loocv_256"]["pred_vote_share"] for s in s_list]
        margins_128 = [s["loocv_128"]["margin"] for s in s_list]
        margins_256 = [s["loocv_256"]["margin"] for s in s_list]
        nbrs_128 = [s["loocv_128"]["true_neighbors"] for s in s_list]
        nbrs_256 = [s["loocv_256"]["true_neighbors"] for s in s_list]
        ndist_128 = [s["loocv_128"]["nearest_dist"] for s in s_list]
        ndist_256 = [s["loocv_256"]["nearest_dist"] for s in s_list]

        roi_covs = [s["roi_coverage"] for s in s_list]
        fg_128s = [s["fg_ratio_128"] for s in s_list]
        fg_256s = [s["fg_ratio_256"] for s in s_list]
        fg_rois = [s["fg_ratio_roi"] for s in s_list]
        orig_aspects = [s["orig_aspect"] for s in s_list]
        roi_aspects = [s["roi_aspect"] for s in s_list]
        fg_pixs = [s["fg_pix_count"] for s in s_list]
        hnorms_128 = [s["hog_norm_128"] for s in s_list]
        hnorms_256 = [s["hog_norm_256"] for s in s_list]

        pos_row = {
            "position": f"#{p:02d}",
            "pos_int": p,
            "total_samples": cnt,
            "correct_128": corr_128,
            "wrong_128": cnt - corr_128,
            "accuracy_128": acc_128,
            "correct_256": corr_256,
            "wrong_256": cnt - corr_256,
            "accuracy_256": acc_256,
            "delta_accuracy": delta_acc,
            "mean_vote_128": round(float(np.mean(votes_128)), 2),
            "median_vote_128": round(float(np.median(votes_128)), 2),
            "mean_vote_256": round(float(np.mean(votes_256)), 2),
            "median_vote_256": round(float(np.median(votes_256)), 2),
            "mean_nbr_128": round(float(np.mean(nbrs_128)), 2),
            "mean_nbr_256": round(float(np.mean(nbrs_256)), 2),
            "mean_margin_128": round(float(np.mean(margins_128)), 2),
            "mean_margin_256": round(float(np.mean(margins_256)), 2),
            "mean_nearest_dist_128": round(float(np.mean(ndist_128)), 4),
            "mean_nearest_dist_256": round(float(np.mean(ndist_256)), 4),
            "mean_roi_coverage": round(float(np.mean(roi_covs)), 4),
            "mean_fg_ratio_128": round(float(np.mean(fg_128s)), 4),
            "mean_fg_ratio_256": round(float(np.mean(fg_256s)), 4),
        }
        position_data.append(pos_row)

        vis_row = {
            "position": f"#{p:02d}",
            "pos_int": p,
            "accuracy_128": acc_128,
            "accuracy_256": acc_256,
            "mean_roi_coverage": round(float(np.mean(roi_covs)), 4),
            "mean_fg_ratio_orig": round(float(np.mean([s["fg_ratio_orig"] for s in s_list])), 4),
            "mean_fg_ratio_roi": round(float(np.mean(fg_rois)), 4),
            "mean_fg_ratio_128": round(float(np.mean(fg_128s)), 4),
            "mean_fg_ratio_256": round(float(np.mean(fg_256s)), 4),
            "mean_orig_aspect": round(float(np.mean(orig_aspects)), 4),
            "mean_roi_aspect": round(float(np.mean(roi_aspects)), 4),
            "mean_fg_pixels": round(float(np.mean(fg_pixs)), 1),
            "mean_hog_norm_128": round(float(np.mean(hnorms_128)), 4),
            "mean_hog_norm_256": round(float(np.mean(hnorms_256)), 4),
            "mean_nearest_dist_128": round(float(np.mean(ndist_128)), 4),
            "mean_nearest_dist_256": round(float(np.mean(ndist_256)), 4),
        }
        position_visual.append(vis_row)

    # ------------------------------------------------------------------
    # 5. Statistical Position Analysis
    # ------------------------------------------------------------------
    accs_128 = [p["accuracy_128"] for p in position_data]
    accs_256 = [p["accuracy_256"] for p in position_data]

    mean_acc_128 = round(float(np.mean(accs_128)), 2)
    median_acc_128 = round(float(np.median(accs_128)), 2)
    std_acc_128 = round(float(np.std(accs_128)), 2)
    min_acc_128 = round(float(np.min(accs_128)), 2)
    max_acc_128 = round(float(np.max(accs_128)), 2)

    mean_acc_256 = round(float(np.mean(accs_256)), 2)
    median_acc_256 = round(float(np.median(accs_256)), 2)
    std_acc_256 = round(float(np.std(accs_256)), 2)
    min_acc_256 = round(float(np.min(accs_256)), 2)
    max_acc_256 = round(float(np.max(accs_256)), 2)

    # Identify lowest & highest positions
    sorted_pos_128 = sorted(position_data, key=lambda x: x["accuracy_128"])
    sorted_pos_256 = sorted(position_data, key=lambda x: x["accuracy_256"])

    lowest_pos_128 = sorted_pos_128[0]["position"]
    highest_pos_128 = sorted_pos_128[-1]["position"]
    lowest_pos_256 = sorted_pos_256[0]["position"]
    highest_pos_256 = sorted_pos_256[-1]["position"]

    # Rank of #1 (1-indexed rank from lowest=1 to highest=20)
    rank_pos1_128 = [p["position"] for p in sorted_pos_128].index("#01") + 1
    rank_pos1_256 = [p["position"] for p in sorted_pos_256].index("#01") + 1

    # Compare #1 vs #2..20
    pos1_acc_128 = position_data[0]["accuracy_128"]
    pos2_20_accs_128 = [p["accuracy_128"] for p in position_data[1:]]
    pos2_20_mean_128 = round(float(np.mean(pos2_20_accs_128)), 2)
    pos1_diff_128 = round(pos1_acc_128 - pos2_20_mean_128, 2)

    pos1_acc_256 = position_data[0]["accuracy_256"]
    pos2_20_accs_256 = [p["accuracy_256"] for p in position_data[1:]]
    pos2_20_mean_256 = round(float(np.mean(pos2_20_accs_256)), 2)
    pos1_diff_256 = round(pos1_acc_256 - pos2_20_mean_256, 2)

    # Correlation between accuracy and visual metrics across 20 positions
    def calc_corr(x_vals, y_vals):
        x = np.array(x_vals)
        y = np.array(y_vals)
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return round(float(np.corrcoef(x, y)[0, 1]), 4)

    correlations_128 = {
        "roi_coverage": calc_corr(accs_128, [p["mean_roi_coverage"] for p in position_visual]),
        "fg_ratio": calc_corr(accs_128, [p["mean_fg_ratio_128"] for p in position_visual]),
        "orig_aspect": calc_corr(accs_128, [p["mean_orig_aspect"] for p in position_visual]),
        "roi_aspect": calc_corr(accs_128, [p["mean_roi_aspect"] for p in position_visual]),
        "hog_norm": calc_corr(accs_128, [p["mean_hog_norm_128"] for p in position_visual]),
        "nearest_dist": calc_corr(accs_128, [p["mean_nearest_dist_128"] for p in position_visual]),
    }

    correlations_256 = {
        "roi_coverage": calc_corr(accs_256, [p["mean_roi_coverage"] for p in position_visual]),
        "fg_ratio": calc_corr(accs_256, [p["mean_fg_ratio_256"] for p in position_visual]),
        "orig_aspect": calc_corr(accs_256, [p["mean_orig_aspect"] for p in position_visual]),
        "roi_aspect": calc_corr(accs_256, [p["mean_roi_aspect"] for p in position_visual]),
        "hog_norm": calc_corr(accs_256, [p["mean_hog_norm_256"] for p in position_visual]),
        "nearest_dist": calc_corr(accs_256, [p["mean_nearest_dist_256"] for p in position_visual]),
    }

    # ------------------------------------------------------------------
    # 6. Per-Student x Position Matrices (18 x 20)
    # ------------------------------------------------------------------
    correctness_matrix_128 = []
    correctness_matrix_256 = []
    prediction_matrix_128 = []
    prediction_matrix_256 = []

    student_samples = defaultdict(list)
    for s in samples_data:
        student_samples[s["student_name"]].append(s)

    for st in students:
        s_list = sorted(student_samples[st], key=lambda x: x["sample_num"])
        
        c_row_128 = {"student": st}
        c_row_256 = {"student": st}
        p_row_128 = {"student": st}
        p_row_256 = {"student": st}

        for s in s_list:
            pos_key = f"p{s['sample_num']:02d}"
            c_row_128[pos_key] = 1 if s["loocv_128"]["is_correct"] else 0
            c_row_256[pos_key] = 1 if s["loocv_256"]["is_correct"] else 0
            p_row_128[pos_key] = s["loocv_128"]["pred_label"]
            p_row_256[pos_key] = s["loocv_256"]["pred_label"]

        n_corr_128 = sum(1 for s in s_list if s["loocv_128"]["is_correct"])
        n_corr_256 = sum(1 for s in s_list if s["loocv_256"]["is_correct"])
        c_row_128["total_correct"] = n_corr_128
        c_row_128["accuracy_pct"] = round(n_corr_128 / len(s_list) * 100, 2)
        c_row_256["total_correct"] = n_corr_256
        c_row_256["accuracy_pct"] = round(n_corr_256 / len(s_list) * 100, 2)

        correctness_matrix_128.append(c_row_128)
        correctness_matrix_256.append(c_row_256)
        prediction_matrix_128.append(p_row_128)
        prediction_matrix_256.append(p_row_256)

    # ------------------------------------------------------------------
    # 7. Sample #1 Deep Audit (18 samples)
    # ------------------------------------------------------------------
    sample1_list = pos_samples[1]
    sample1_rows = []
    s1_transitions = {"CC": 0, "WC": 0, "CW": 0, "WW": 0}

    for s in sample1_list:
        c128 = s["loocv_128"]["is_correct"]
        c256 = s["loocv_256"]["is_correct"]
        if c128 and c256:
            trans = "CC"
        elif not c128 and c256:
            trans = "WC"
        elif c128 and not c256:
            trans = "CW"
        else:
            trans = "WW"
        s1_transitions[trans] += 1

        sample1_rows.append({
            "student": s["student_name"],
            "filename": s["filename"],
            "pred_128": s["loocv_128"]["pred_label"],
            "correct_128": c128,
            "pred_256": s["loocv_256"]["pred_label"],
            "correct_256": c256,
            "transition": trans,
            "roi_coverage": s["roi_coverage"],
            "fg_ratio_128": s["fg_ratio_128"],
            "fg_ratio_256": s["fg_ratio_256"],
            "orig_aspect": s["orig_aspect"],
            "roi_aspect": s["roi_aspect"],
            "nearest_dist_128": s["loocv_128"]["nearest_dist"],
            "nearest_dist_256": s["loocv_256"]["nearest_dist"],
            "vote_128": s["loocv_128"]["pred_vote_share"],
            "vote_256": s["loocv_256"]["pred_vote_share"],
            "margin_128": s["loocv_128"]["margin"],
            "margin_256": s["loocv_256"]["margin"],
            "neighbors_128": s["loocv_128"]["true_neighbors"],
            "neighbors_256": s["loocv_256"]["true_neighbors"],
        })

    # ------------------------------------------------------------------
    # 8. Confusion by Position
    # ------------------------------------------------------------------
    position_confusion = []
    for p in range(1, 21):
        s_list = pos_samples[p]
        wrong_128 = [s for s in s_list if not s["loocv_128"]["is_correct"]]
        wrong_256 = [s for s in s_list if not s["loocv_256"]["is_correct"]]

        preds_128 = [s["loocv_128"]["pred_label"] for s in wrong_128]
        preds_256 = [s["loocv_256"]["pred_label"] for s in wrong_256]

        cnts_128 = defaultdict(int)
        for pr in preds_128:
            cnts_128[pr] += 1
        dom_target_128 = max(cnts_128, key=cnts_128.get) if cnts_128 else "None"
        dom_cnt_128 = cnts_128[dom_target_128] if cnts_128 else 0

        cnts_256 = defaultdict(int)
        for pr in preds_256:
            cnts_256[pr] += 1
        dom_target_256 = max(cnts_256, key=cnts_256.get) if cnts_256 else "None"
        dom_cnt_256 = cnts_256[dom_target_256] if cnts_256 else 0

        ent_128 = calc_shannon_entropy(preds_128)
        ent_256 = calc_shannon_entropy(preds_256)

        position_confusion.append({
            "position": f"#{p:02d}",
            "pos_int": p,
            "errors_128": len(wrong_128),
            "dominant_target_128": dom_target_128,
            "dominant_count_128": dom_cnt_128,
            "unique_targets_128": len(cnts_128),
            "entropy_128": ent_128,
            "errors_256": len(wrong_256),
            "dominant_target_256": dom_target_256,
            "dominant_count_256": dom_cnt_256,
            "unique_targets_256": len(cnts_256),
            "entropy_256": ent_256,
        })

    # ------------------------------------------------------------------
    # 9. First-Position vs Student Effect (H1 vs H2)
    # ------------------------------------------------------------------
    student_effect = []
    cat_counts_128 = defaultdict(int)
    cat_counts_256 = defaultdict(int)

    for st in students:
        s_list = student_samples[st]
        s1 = [s for s in s_list if s["sample_num"] == 1][0]
        s_rest = [s for s in s_list if s["sample_num"] > 1]

        # 128
        total_corr_128 = sum(1 for s in s_list if s["loocv_128"]["is_correct"])
        overall_acc_128 = round(total_corr_128 / len(s_list) * 100, 2)
        pos1_ok_128 = s1["loocv_128"]["is_correct"]
        rest_corr_128 = sum(1 for s in s_rest if s["loocv_128"]["is_correct"])
        rest_acc_128 = round(rest_corr_128 / len(s_rest) * 100, 2)

        if not pos1_ok_128 and rest_acc_128 >= 50.0:
            cat_128 = "Cat 1: Wrong #1 but good on #2-20"
        elif not pos1_ok_128 and rest_acc_128 < 50.0:
            cat_128 = "Cat 2: Wrong #1 and poor across all"
        elif pos1_ok_128 and rest_acc_128 < 50.0:
            cat_128 = "Cat 3: Correct #1 but poor on #2-20"
        else:
            cat_128 = "Cat 4: Stable correct (both #1 and #2-20)"
        cat_counts_128[cat_128] += 1

        # 256
        total_corr_256 = sum(1 for s in s_list if s["loocv_256"]["is_correct"])
        overall_acc_256 = round(total_corr_256 / len(s_list) * 100, 2)
        pos1_ok_256 = s1["loocv_256"]["is_correct"]
        rest_corr_256 = sum(1 for s in s_rest if s["loocv_256"]["is_correct"])
        rest_acc_256 = round(rest_corr_256 / len(s_rest) * 100, 2)

        if not pos1_ok_256 and rest_acc_256 >= 50.0:
            cat_256 = "Cat 1: Wrong #1 but good on #2-20"
        elif not pos1_ok_256 and rest_acc_256 < 50.0:
            cat_256 = "Cat 2: Wrong #1 and poor across all"
        elif pos1_ok_256 and rest_acc_256 < 50.0:
            cat_256 = "Cat 3: Correct #1 but poor on #2-20"
        else:
            cat_256 = "Cat 4: Stable correct (both #1 and #2-20)"
        cat_counts_256[cat_256] += 1

        student_effect.append({
            "student": st,
            "overall_acc_128": overall_acc_128,
            "pos1_correct_128": pos1_ok_128,
            "rest_acc_128": rest_acc_128,
            "category_128": cat_128,
            "overall_acc_256": overall_acc_256,
            "pos1_correct_256": pos1_ok_256,
            "rest_acc_256": rest_acc_256,
            "category_256": cat_256,
        })

    # ------------------------------------------------------------------
    # 10. 128 -> 256 Position Transitions
    # ------------------------------------------------------------------
    pos_transitions = []
    for p in range(1, 21):
        s_list = pos_samples[p]
        cc, wc, cw, ww_same, ww_diff = 0, 0, 0, 0, 0
        for s in s_list:
            c128 = s["loocv_128"]["is_correct"]
            c256 = s["loocv_256"]["is_correct"]
            if c128 and c256:
                cc += 1
            elif not c128 and c256:
                wc += 1
            elif c128 and not c256:
                cw += 1
            elif s["loocv_128"]["pred_label"] == s["loocv_256"]["pred_label"]:
                ww_same += 1
            else:
                ww_diff += 1
        pos_transitions.append({
            "position": f"#{p:02d}",
            "pos_int": p,
            "total": len(s_list),
            "CC_both_correct": cc,
            "WC_improved": wc,
            "CW_regressed": cw,
            "WW_same": ww_same,
            "WW_diff": ww_diff,
            "net_gain": wc - cw,
        })

    # ------------------------------------------------------------------
    # 11. Generate Contact Sheets
    # ------------------------------------------------------------------
    print("\nGenerating representative contact sheets...")
    # 1. Contact sheet for position #1
    cs_pos1_path = os.path.join(CONTACT_DIR, "contact_sheet_position_01.png")
    create_contact_sheet(pos_samples[1], "Sample Position #1 (All 18 Students) - Comparison @128 vs @256", cs_pos1_path)

    # 2. Contact sheet for highest accuracy position (in 128)
    highest_p = sorted_pos_128[-1]["pos_int"]
    cs_high_path = os.path.join(CONTACT_DIR, f"contact_sheet_position_{highest_p:02d}_highest.png")
    create_contact_sheet(pos_samples[highest_p], f"Sample Position #{highest_p:02d} (Highest Accuracy: {sorted_pos_128[-1]['accuracy_128']}%)", cs_high_path)

    # 3. Contact sheet for representative median accuracy position
    median_p = sorted_pos_128[len(sorted_pos_128)//2]["pos_int"]
    cs_med_path = os.path.join(CONTACT_DIR, f"contact_sheet_position_{median_p:02d}_median.png")
    create_contact_sheet(pos_samples[median_p], f"Sample Position #{median_p:02d} (Median Accuracy: {sorted_pos_128[len(sorted_pos_128)//2]['accuracy_128']}%)", cs_med_path)

    # ------------------------------------------------------------------
    # 12. Save All Output Files
    # ------------------------------------------------------------------
    print("\nSaving CSV and JSON output files...")

    # position_accuracy_128_256.csv
    pacc_path = os.path.join(OUT_DIR, "position_accuracy_128_256.csv")
    with open(pacc_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(position_data[0].keys()))
        w.writeheader(); w.writerows(position_data)
    print(f"  Saved: {pacc_path}")

    # correctness_matrix_128.csv & 256
    c128_path = os.path.join(OUT_DIR, "correctness_matrix_128.csv")
    with open(c128_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(correctness_matrix_128[0].keys()))
        w.writeheader(); w.writerows(correctness_matrix_128)
    print(f"  Saved: {c128_path}")

    c256_path = os.path.join(OUT_DIR, "correctness_matrix_256.csv")
    with open(c256_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(correctness_matrix_256[0].keys()))
        w.writeheader(); w.writerows(correctness_matrix_256)
    print(f"  Saved: {c256_path}")

    # prediction_matrix_128.csv & 256
    p128_path = os.path.join(OUT_DIR, "prediction_matrix_128.csv")
    with open(p128_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(prediction_matrix_128[0].keys()))
        w.writeheader(); w.writerows(prediction_matrix_128)
    print(f"  Saved: {p128_path}")

    p256_path = os.path.join(OUT_DIR, "prediction_matrix_256.csv")
    with open(p256_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(prediction_matrix_256[0].keys()))
        w.writeheader(); w.writerows(prediction_matrix_256)
    print(f"  Saved: {p256_path}")

    # position_visual_metrics.csv
    pvis_path = os.path.join(OUT_DIR, "position_visual_metrics.csv")
    with open(pvis_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(position_visual[0].keys()))
        w.writeheader(); w.writerows(position_visual)
    print(f"  Saved: {pvis_path}")

    # position_confusion_analysis.csv
    pconf_path = os.path.join(OUT_DIR, "position_confusion_analysis.csv")
    with open(pconf_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(position_confusion[0].keys()))
        w.writeheader(); w.writerows(position_confusion)
    print(f"  Saved: {pconf_path}")

    # sample1_deep_audit.csv
    s1_path = os.path.join(OUT_DIR, "sample1_deep_audit.csv")
    with open(s1_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sample1_rows[0].keys()))
        w.writeheader(); w.writerows(sample1_rows)
    print(f"  Saved: {s1_path}")

    # student_position_effect.csv
    spe_path = os.path.join(OUT_DIR, "student_position_effect.csv")
    with open(spe_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(student_effect[0].keys()))
        w.writeheader(); w.writerows(student_effect)
    print(f"  Saved: {spe_path}")

    # resolution_position_transitions.csv
    rpt_path = os.path.join(OUT_DIR, "resolution_position_transitions.csv")
    with open(rpt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pos_transitions[0].keys()))
        w.writeheader(); w.writerows(pos_transitions)
    print(f"  Saved: {rpt_path}")

    # Master JSON Report
    master_report = {
        "experiment": "D — Page-Position / Sample-Position Bias Audit",
        "dataset_summary": {
            "total_samples": n,
            "total_students": n_students,
            "samples_per_student": 20,
            "filename_patterns": dict(filename_patterns),
        },
        "statistical_summary": {
            "128x128": {
                "mean_position_acc": mean_acc_128,
                "median_position_acc": median_acc_128,
                "std_position_acc": std_acc_128,
                "min_position_acc": min_acc_128,
                "max_position_acc": max_acc_128,
                "lowest_position": lowest_pos_128,
                "highest_position": highest_pos_128,
                "pos1_rank": rank_pos1_128,
                "pos1_accuracy": pos1_acc_128,
                "pos2_20_mean_accuracy": pos2_20_mean_128,
                "pos1_vs_pos2_20_delta": pos1_diff_128,
                "correlations_with_accuracy": correlations_128,
            },
            "256x256": {
                "mean_position_acc": mean_acc_256,
                "median_position_acc": median_acc_256,
                "std_position_acc": std_acc_256,
                "min_position_acc": min_acc_256,
                "max_position_acc": max_acc_256,
                "lowest_position": lowest_pos_256,
                "highest_position": highest_pos_256,
                "pos1_rank": rank_pos1_256,
                "pos1_accuracy": pos1_acc_256,
                "pos2_20_mean_accuracy": pos2_20_mean_256,
                "pos1_vs_pos2_20_delta": pos1_diff_256,
                "correlations_with_accuracy": correlations_256,
            },
        },
        "sample1_deep_audit": {
            "transitions_summary": s1_transitions,
            "sample1_details": sample1_rows,
        },
        "student_effect_categories": {
            "128x128": dict(cat_counts_128),
            "256x256": dict(cat_counts_256),
        },
        "position_data": position_data,
        "position_confusion": position_confusion,
        "position_transitions": pos_transitions,
    }

    json_path = os.path.join(OUT_DIR, "page_position_audit.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, (int, np.integer)) else
                                    float(o) if isinstance(o, (float, np.floating)) else
                                    bool(o) if isinstance(o, (bool, np.bool_)) else str(o))
    print(f"  Saved: {json_path}")

    # ------------------------------------------------------------------
    # 13. Console Output Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("POSITION ACCURACY SUMMARY (All 20 Positions)")
    print("=" * 80)
    hdr = "{:<10} {:>9} {:>9} {:>8} {:>10} {:>10} {:>10} {:>10}"
    print(hdr.format("Position", "Acc@128", "Acc@256", "Delta", "Vote@128", "Vote@256", "Margin@128", "Margin@256"))
    print("-" * 80)
    for p in position_data:
        sign = "+" if p["delta_accuracy"] >= 0 else ""
        print(hdr.format(
            p["position"],
            f"{p['accuracy_128']:.1f}%",
            f"{p['accuracy_256']:.1f}%",
            f"{sign}{p['delta_accuracy']:.1f}",
            f"{p['mean_vote_128']:.1f}%",
            f"{p['mean_vote_256']:.1f}%",
            f"{p['mean_margin_128']:.1f}%",
            f"{p['mean_margin_256']:.1f}%"
        ))

    print("-" * 80)
    print(f"Lowest position @128       : {lowest_pos_128} ({min_acc_128}%)")
    print(f"Lowest position @256       : {lowest_pos_256} ({min_acc_256}%)")
    print(f"Highest position @128      : {highest_pos_128} ({max_acc_128}%)")
    print(f"Highest position @256      : {highest_pos_256} ({max_acc_256}%)")
    print(f"Median position acc @128   : {median_acc_128}% (Mean: {mean_acc_128}%, Std: {std_acc_128}%)")
    print(f"Median position acc @256   : {median_acc_256}% (Mean: {mean_acc_256}%, Std: {std_acc_256}%)")
    print(f"Position #1 rank @128      : #{rank_pos1_128} of 20 (Accuracy: {pos1_acc_128}% vs #2-20 Mean: {pos2_20_mean_128}%, Delta: {pos1_diff_128}%)")
    print(f"Position #1 rank @256      : #{rank_pos1_256} of 20 (Accuracy: {pos1_acc_256}% vs #2-20 Mean: {pos2_20_mean_256}%, Delta: {pos1_diff_256}%)")

    print("\nSAMPLE #1 TRANSITIONS (128 -> 256):")
    print(f"  CC (Correct in both)       : {s1_transitions['CC']}")
    print(f"  WC (128 Wrong -> 256 Correct): {s1_transitions['WC']}")
    print(f"  CW (128 Correct -> 256 Wrong): {s1_transitions['CW']}")
    print(f"  WW (Wrong in both)         : {s1_transitions['WW']}")

    print("\nSTUDENT EFFECT BREAKDOWN (128x128):")
    for cat, c_val in cat_counts_128.items():
        print(f"  {cat:<45}: {c_val} students ({c_val/n_students*100:.1f}%)")

    print("\nSTUDENT EFFECT BREAKDOWN (256x256):")
    for cat, c_val in cat_counts_256.items():
        print(f"  {cat:<45}: {c_val} students ({c_val/n_students*100:.1f}%)")

    print("\nCORRELATIONS WITH POSITION ACCURACY:")
    print(f"  {'Metric':<25} {'Corr@128':>10} {'Corr@256':>10}")
    print("  " + "-" * 47)
    for k in correlations_128:
        print(f"  {k:<25} {correlations_128[k]:>10.4f} {correlations_256[k]:>10.4f}")

    print("=" * 80)
    print("EXPERIMENT D AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment_d()
