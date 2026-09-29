"""
tests/experiment_e_pure_crop.py — Experiment E: Pure Handwriting Crop Controlled Diagnostic
=============================================================================================
Evaluates whether spatial exclusion of document-layout/header/footer contamination
improves HOG+KNN writer verification across 360 images (18 students x 20 samples).

Three Controlled Branches:
  Branch A: Baseline 128 (Baseline preprocessing/ROI -> 128x128 -> HOG -> KNN)
  Branch B: Baseline 256 (Baseline preprocessing/ROI -> 256x256 -> HOG -> KNN)
  Branch C: PureCrop 256 (Pure handwriting spatial crop -> preprocessing -> 256x256 -> HOG -> KNN)

All HOG parameters (orientations=9, pixels_per_cell=8x8, cells_per_block=2x2, L2-Hys)
and KNN parameters (K=5, Euclidean, distance weights) are STRICTLY IDENTICAL across branches.

Outputs stored in: backend/tests/evaluation_results/pure_crop_experiment/
"""

import os
import sys
import json
import csv
import time
import sqlite3
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

OUT_DIR = os.path.join(BASE_DIR, "tests", "evaluation_results", "pure_crop_experiment")
os.makedirs(OUT_DIR, exist_ok=True)


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


def pure_handwriting_crop(image_bgr):
    """
    Deterministic spatial crop of pure handwriting body.
    Rule:
      1. Normalize orientation (portrait).
      2. Exclude top header zone (top 15% where titles/metadata reside).
      3. Exclude bottom footer zone (bottom 8% margin).
      4. Exclude side margins (6% on left and right).
      5. Standard grayscale -> blur -> Otsu -> noise removal -> baseline extract_roi().
      6. Preserves ALL handwriting strokes within the region (NO component masking).
    Returns: (cropped_roi_binary, (bx, by, bw, bh))
    """
    oriented = normalize_orientation(image_bgr)
    h, w = oriented.shape[:2]

    y1 = int(h * 0.15)
    y2 = int(h * 0.92)
    x1 = int(w * 0.06)
    x2 = int(w * 0.94)

    sub_bgr = oriented[y1:y2, x1:x2]

    gray = convert_to_grayscale(sub_bgr)
    blurred = apply_gaussian_blur(gray)
    binary = apply_otsu_threshold(blurred)
    denoised = remove_noise(binary)
    roi_binary = extract_roi(denoised)

    contours, _ = cv2.findContours(denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        all_pts = np.concatenate(contours, axis=0)
        bx, by, bw, bh = cv2.boundingRect(all_pts)
        pad = max(4, int(min(bw, bh) * 0.05))
        bx1 = max(0, bx - pad) + x1
        by1 = max(0, by - pad) + y1
        bx2 = min(w, bx + bw + pad) + x1
        by2 = min(h, by + bh + pad) + y1
    else:
        bx1, by1, bx2, by2 = x1, y1, x2, y2

    return roi_binary, (bx1, by1, bx2 - bx1, by2 - by1)


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
        })

    return results, dist_matrix


def compute_dist_metrics(X, y, dist_matrix=None):
    y_a = np.array(y)
    if dist_matrix is None:
        dist_matrix = pairwise_dist(X)
    unique_classes = sorted(set(y))
    cls_idx = {c: np.where(y_a == c)[0] for c in unique_classes}

    intra_all, inter_all = [], []
    per_class = {}

    for c in unique_classes:
        ci = cls_idx[c]
        oi = np.where(y_a != c)[0]
        sub = dist_matrix[np.ix_(ci, ci)]
        iu = np.triu_indices(len(ci), k=1)
        intra = sub[iu]
        inter = dist_matrix[np.ix_(ci, oi)].flatten()
        intra_all.extend(intra)
        inter_all.extend(inter)
        per_class[c] = {
            "intra_mean": round(float(np.mean(intra)), 4),
            "inter_mean": round(float(np.mean(inter)), 4),
            "sep_ratio": round(float(np.mean(inter) / np.mean(intra)) if np.mean(intra) > 0 else 0, 4),
        }

    return {
        "overall_intra_mean": round(float(np.mean(intra_all)), 4),
        "overall_inter_mean": round(float(np.mean(inter_all)), 4),
        "separability_ratio": round(float(np.mean(inter_all) / np.mean(intra_all)) if np.mean(intra_all) > 0 else 0, 4),
        "per_class": per_class,
    }, dist_matrix


def build_confusion_matrix(loocv_results, unique_classes):
    cls2i = {c: i for i, c in enumerate(unique_classes)}
    cm = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
    for r in loocv_results:
        cm[cls2i[r["true_label"]], cls2i[r["pred_label"]]] += 1
    return cm


def run_experiment_e():
    print("=" * 80)
    print("EXPERIMENT E: PURE HANDWRITING CROP — CONTROLLED DIAGNOSTIC")
    print("=" * 80)

    # 1. Load DB Records
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, student_name, original_filename, saved_filename, file_path "
        "FROM dataset ORDER BY student_name ASC, original_filename ASC, saved_filename ASC"
    ).fetchall()
    conn.close()

    n_records = len(rows)
    students = sorted(list(set(r["student_name"] for r in rows)))
    print(f"Loaded {n_records} records across {len(students)} students.")

    # 2. Extract Features for Branch A (Base128), Branch B (Base256), Branch C (PureCrop256)
    feat_base128 = []
    feat_base256 = []
    feat_crop256 = []
    labels = []
    samples_data = []

    t_hog_base128 = []
    t_hog_base256 = []
    t_hog_crop256 = []

    print("\nProcessing all 360 images across 3 branches...")
    for idx, r in enumerate(rows):
        student = r["student_name"]
        raw_name = r["saved_filename"] or os.path.basename(r["file_path"])
        raw_path = os.path.join(DATASET_RAW_DIR, raw_name)

        if not os.path.exists(raw_path):
            continue
        img_bgr = cv2.imread(raw_path)
        if img_bgr is None:
            continue

        # Branch A & B: Baseline Preprocessing & Baseline ROI
        oriented = normalize_orientation(img_bgr)
        orig_h, orig_w = oriented.shape[:2]
        gray = convert_to_grayscale(oriented)
        blurred = apply_gaussian_blur(gray)
        binary = apply_otsu_threshold(blurred)
        denoised = remove_noise(binary)
        base_roi = extract_roi(denoised)

        img_base128 = resize_with_aspect_ratio(base_roi, SIZE_128)
        img_base256 = resize_with_aspect_ratio(base_roi, SIZE_256)

        t0 = time.perf_counter()
        f_b128, _ = extract_hog(img_base128)
        t_hog_base128.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        f_b256, _ = extract_hog(img_base256)
        t_hog_base256.append(time.perf_counter() - t1)

        # Branch C: Pure Handwriting Crop -> Preprocessing -> 256x256
        t2 = time.perf_counter()
        crop_roi, (cx, cy, cw, ch) = pure_handwriting_crop(img_bgr)
        img_crop256 = resize_with_aspect_ratio(crop_roi, SIZE_256)
        f_c256, _ = extract_hog(img_crop256)
        t_hog_crop256.append(time.perf_counter() - t2)

        feat_base128.append(f_b128)
        feat_base256.append(f_b256)
        feat_crop256.append(f_c256)
        labels.append(student)

        sample_num = len([s for s in samples_data if s["student_name"] == student]) + 1

        # Visual metrics
        crop_cov = (cw * ch) / max(1, orig_w * orig_h)
        fg_crop256 = float(np.count_nonzero(img_crop256)) / (256 * 256)
        fg_base256 = float(np.count_nonzero(img_base256)) / (256 * 256)

        samples_data.append({
            "sample_idx": len(samples_data),
            "id": r["id"],
            "student_name": student,
            "filename": r["original_filename"],
            "sample_num": sample_num,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "crop_cov": round(crop_cov, 4),
            "fg_base256": round(fg_base256, 4),
            "fg_crop256": round(fg_crop256, 4),
            "crop_aspect": round(cw / max(1, ch), 4),
            "hog_norm_crop256": round(float(np.linalg.norm(f_c256)), 4),
            "hog_norm_base256": round(float(np.linalg.norm(f_b256)), 4),
        })

    n = len(labels)
    X_base128 = np.array(feat_base128)
    X_base256 = np.array(feat_base256)
    X_crop256 = np.array(feat_crop256)

    # 3. Distance Metrics & LOOCV
    print("\nRunning LOOCV on all 3 branches...")
    loocv_b128, dist_b128 = run_loocv(X_base128, labels)
    loocv_b256, dist_b256 = run_loocv(X_base256, labels)
    loocv_c256, dist_c256 = run_loocv(X_crop256, labels)

    dm_b128, _ = compute_dist_metrics(X_base128, labels, dist_b128)
    dm_b256, _ = compute_dist_metrics(X_base256, labels, dist_b256)
    dm_c256, _ = compute_dist_metrics(X_crop256, labels, dist_c256)

    for i in range(n):
        samples_data[i]["loocv_b128"] = loocv_b128[i]
        samples_data[i]["loocv_b256"] = loocv_b256[i]
        samples_data[i]["loocv_c256"] = loocv_c256[i]

    # Accuracy calculations
    acc_b128 = sum(1 for r in loocv_b128 if r["is_correct"]) / n * 100
    acc_b256 = sum(1 for r in loocv_b256 if r["is_correct"]) / n * 100
    acc_c256 = sum(1 for r in loocv_c256 if r["is_correct"]) / n * 100

    # Position subsets
    p1_samps = [s for s in samples_data if s["sample_num"] == 1]
    p16_samps = [s for s in samples_data if s["sample_num"] == 16]
    prest_samps = [s for s in samples_data if s["sample_num"] > 1]

    def sub_acc(loocv_key, s_list):
        return sum(1 for s in s_list if s[loocv_key]["is_correct"]) / len(s_list) * 100

    p1_acc_b128 = sub_acc("loocv_b128", p1_samps)
    p1_acc_b256 = sub_acc("loocv_b256", p1_samps)
    p1_acc_c256 = sub_acc("loocv_c256", p1_samps)

    p16_acc_b128 = sub_acc("loocv_b128", p16_samps)
    p16_acc_b256 = sub_acc("loocv_b256", p16_samps)
    p16_acc_c256 = sub_acc("loocv_c256", p16_samps)

    prest_acc_b128 = sub_acc("loocv_b128", prest_samps)
    prest_acc_b256 = sub_acc("loocv_b256", prest_samps)
    prest_acc_c256 = sub_acc("loocv_c256", prest_samps)

    # Neighbor vote stats
    def vote_stats(loocv_list):
        nbr_ge3 = sum(1 for r in loocv_list if r["true_neighbors"] >= 3) / n * 100
        mean_vote = float(np.mean([r["pred_vote_share"] for r in loocv_list]))
        mean_margin = float(np.mean([r["margin"] for r in loocv_list]))
        return round(nbr_ge3, 2), round(mean_vote, 2), round(mean_margin, 2)

    nbr_b128, vote_b128, margin_b128 = vote_stats(loocv_b128)
    nbr_b256, vote_b256, margin_b256 = vote_stats(loocv_b256)
    nbr_c256, vote_c256, margin_c256 = vote_stats(loocv_c256)

    # 4. Sample-Level Transitions (Baseline 256 -> PureCrop 256)
    transitions = {"CC": 0, "WC": 0, "CW": 0, "WW_same": 0, "WW_diff": 0}
    sample_trans_rows = []

    for s in samples_data:
        r_b = s["loocv_b256"]
        r_c = s["loocv_c256"]
        if r_b["is_correct"] and r_c["is_correct"]:
            cat = "CC"
        elif not r_b["is_correct"] and r_c["is_correct"]:
            cat = "WC"
        elif r_b["is_correct"] and not r_c["is_correct"]:
            cat = "CW"
        elif r_b["pred_label"] == r_c["pred_label"]:
            cat = "WW_same"
        else:
            cat = "WW_diff"
        transitions[cat] += 1

        sample_trans_rows.append({
            "sample_idx": s["sample_idx"],
            "student": s["student_name"],
            "sample_num": s["sample_num"],
            "filename": s["filename"],
            "correct_base256": r_b["is_correct"],
            "pred_base256": r_b["pred_label"],
            "correct_crop256": r_c["is_correct"],
            "pred_crop256": r_c["pred_label"],
            "transition": cat,
            "crop_cov": s["crop_cov"],
            "fg_crop256": s["fg_crop256"],
        })

    net_improvement = transitions["WC"] - transitions["CW"]

    # 5. Position Comparison Table (Positions 1..20)
    pos_comparison_rows = []
    for p in range(1, 21):
        s_list = [s for s in samples_data if s["sample_num"] == p]
        cnt = len(s_list)
        p_b128 = sum(1 for s in s_list if s["loocv_b128"]["is_correct"]) / cnt * 100
        p_b256 = sum(1 for s in s_list if s["loocv_b256"]["is_correct"]) / cnt * 100
        p_c256 = sum(1 for s in s_list if s["loocv_c256"]["is_correct"]) / cnt * 100
        pos_comparison_rows.append({
            "position": f"#{p:02d}",
            "pos_int": p,
            "acc_base128": round(p_b128, 2),
            "acc_base256": round(p_b256, 2),
            "acc_crop256": round(p_c256, 2),
            "delta_crop_vs_base256": round(p_c256 - p_b256, 2),
            "delta_crop_vs_base128": round(p_c256 - p_b128, 2),
        })

    # 6. Per-Student Comparison Table
    student_comparison_rows = []
    st_improved, st_unchanged, st_regressed = 0, 0, 0

    for st in students:
        s_list = [s for s in samples_data if s["student_name"] == st]
        cnt = len(s_list)
        st_b128 = sum(1 for s in s_list if s["loocv_b128"]["is_correct"]) / cnt * 100
        st_b256 = sum(1 for s in s_list if s["loocv_b256"]["is_correct"]) / cnt * 100
        st_c256 = sum(1 for s in s_list if s["loocv_c256"]["is_correct"]) / cnt * 100
        delta = round(st_c256 - st_b256, 2)

        if delta > 0:
            st_improved += 1
        elif delta == 0:
            st_unchanged += 1
        else:
            st_regressed += 1

        # Errors & main confusion
        wrong_b256 = [s for s in s_list if not s["loocv_b256"]["is_correct"]]
        wrong_c256 = [s for s in s_list if not s["loocv_c256"]["is_correct"]]

        cnt_b256 = defaultdict(int)
        for w in wrong_b256:
            cnt_b256[w["loocv_b256"]["pred_label"]] += 1
        main_conf_b256 = max(cnt_b256, key=cnt_b256.get) if cnt_b256 else "None"

        cnt_c256 = defaultdict(int)
        for w in wrong_c256:
            cnt_c256[w["loocv_c256"]["pred_label"]] += 1
        main_conf_c256 = max(cnt_c256, key=cnt_c256.get) if cnt_c256 else "None"

        vote_b = float(np.mean([s["loocv_b256"]["pred_vote_share"] for s in s_list]))
        vote_c = float(np.mean([s["loocv_c256"]["pred_vote_share"] for s in s_list]))
        margin_b = float(np.mean([s["loocv_b256"]["margin"] for s in s_list]))
        margin_c = float(np.mean([s["loocv_c256"]["margin"] for s in s_list]))

        student_comparison_rows.append({
            "student": st,
            "acc_base128": round(st_b128, 2),
            "acc_base256": round(st_b256, 2),
            "acc_crop256": round(st_c256, 2),
            "delta_crop_vs_base256": delta,
            "errors_base256": len(wrong_b256),
            "errors_crop256": len(wrong_c256),
            "main_confusion_base256": main_conf_b256,
            "main_confusion_crop256": main_conf_c256,
            "mean_vote_base256": round(vote_b, 2),
            "mean_vote_crop256": round(vote_c, 2),
            "mean_margin_base256": round(margin_b, 2),
            "mean_margin_crop256": round(margin_c, 2),
        })

    student_comparison_rows.sort(key=lambda x: -x["delta_crop_vs_base256"])

    # 7. Sample Position #1 & #16 Deep Analysis Tables
    def make_pos_deep_rows(pos_num):
        rows_out = []
        pos_samps = [s for s in samples_data if s["sample_num"] == pos_num]
        for s in pos_samps:
            rb = s["loocv_b256"]
            rc = s["loocv_c256"]
            r128 = s["loocv_b128"]
            if rb["is_correct"] and rc["is_correct"]:
                tr = "CC"
            elif not rb["is_correct"] and rc["is_correct"]:
                tr = "WC"
            elif rb["is_correct"] and not rc["is_correct"]:
                tr = "CW"
            elif rb["pred_label"] == rc["pred_label"]:
                tr = "WW_same"
            else:
                tr = "WW_diff"

            rows_out.append({
                "student": s["student_name"],
                "filename": s["filename"],
                "pred_128": r128["pred_label"],
                "pred_256": rb["pred_label"],
                "pred_crop256": rc["pred_label"],
                "correct_128": r128["is_correct"],
                "correct_256": rb["is_correct"],
                "correct_crop256": rc["is_correct"],
                "transition_256_to_crop": tr,
                "vote_128": r128["pred_vote_share"],
                "vote_256": rb["pred_vote_share"],
                "vote_crop256": rc["pred_vote_share"],
                "margin_128": r128["margin"],
                "margin_256": rb["margin"],
                "margin_crop256": rc["margin"],
                "crop_cov": s["crop_cov"],
                "fg_crop256": s["fg_crop256"],
            })
        return rows_out

    sample1_analysis_rows = make_pos_deep_rows(1)
    sample16_analysis_rows = make_pos_deep_rows(16)

    # 8. Confusion Matrices & Pairwise Analysis
    cm_b128 = build_confusion_matrix(loocv_b128, students)
    cm_b256 = build_confusion_matrix(loocv_b256, students)
    cm_c256 = build_confusion_matrix(loocv_c256, students)

    conf_pairs_rows = []
    cls2i = {c: i for i, c in enumerate(students)}
    pairs_improved, pairs_worsened, pairs_unchanged = 0, 0, 0

    for i in range(len(students)):
        for j in range(i + 1, len(students)):
            sa, sb = students[i], students[j]
            tot_b256 = cm_b256[i, j] + cm_b256[j, i]
            tot_c256 = cm_c256[i, j] + cm_c256[j, i]
            d_pair = tot_c256 - tot_b256

            if d_pair < 0:
                pairs_improved += 1
            elif d_pair > 0:
                pairs_worsened += 1
            else:
                pairs_unchanged += 1

            if tot_b256 > 0 or tot_c256 > 0:
                conf_pairs_rows.append({
                    "class_a": sa,
                    "class_b": sb,
                    "ab_base256": int(cm_b256[i, j]),
                    "ba_base256": int(cm_b256[j, i]),
                    "total_base256": int(tot_b256),
                    "ab_crop256": int(cm_c256[i, j]),
                    "ba_crop256": int(cm_c256[j, i]),
                    "total_crop256": int(tot_c256),
                    "delta_total": int(d_pair),
                })

    conf_pairs_rows.sort(key=lambda x: x["delta_total"])

    # 9. Case Studies: Dinnar & Fahim
    def get_student_case(name):
        row = [r for r in student_comparison_rows if r["student"] == name][0]
        return row

    dinnar_case = get_student_case("Febrian Dinnar Purnama")
    fahim_case = get_student_case("Fahim J Mujaddid")

    # 10. Computational Cost
    comp_cost = {
        "HOG_feature_dimension": {
            "Base128": X_base128.shape[1],
            "Base256": X_base256.shape[1],
            "Crop256": X_crop256.shape[1],
        },
        "mean_hog_extraction_ms_per_image": {
            "Base128": round(float(np.mean(t_hog_base128)) * 1000, 2),
            "Base256": round(float(np.mean(t_hog_base256)) * 1000, 2),
            "Crop256": round(float(np.mean(t_hog_crop256)) * 1000, 2),
        },
        "feature_matrix_RAM_MB": {
            "Base128": round(X_base128.nbytes / 1024 / 1024, 2),
            "Base256": round(X_base256.nbytes / 1024 / 1024, 2),
            "Crop256": round(X_crop256.nbytes / 1024 / 1024, 2),
        },
    }

    # 11. Master JSON Summary
    ab_summary = {
        "experiment": "E — Pure Handwriting Crop Controlled Diagnostic",
        "benchmark_accuracy": {
            "Base128": round(acc_b128, 2),
            "Base256": round(acc_b256, 2),
            "Crop256": round(acc_c256, 2),
            "delta_Crop256_vs_Base256": round(acc_c256 - acc_b256, 2),
            "delta_Crop256_vs_Base128": round(acc_c256 - acc_b128, 2),
        },
        "position_highlights": {
            "pos1_accuracy": {
                "Base128": round(p1_acc_b128, 2),
                "Base256": round(p1_acc_b256, 2),
                "Crop256": round(p1_acc_c256, 2),
                "delta": round(p1_acc_c256 - p1_acc_b256, 2),
            },
            "pos16_accuracy": {
                "Base128": round(p16_acc_b128, 2),
                "Base256": round(p16_acc_b256, 2),
                "Crop256": round(p16_acc_c256, 2),
                "delta": round(p16_acc_c256 - p16_acc_b256, 2),
            },
            "pos2_20_accuracy": {
                "Base128": round(prest_acc_b128, 2),
                "Base256": round(prest_acc_b256, 2),
                "Crop256": round(prest_acc_c256, 2),
                "delta": round(prest_acc_c256 - prest_acc_b256, 2),
            },
        },
        "transitions_Base256_to_Crop256": {
            "CC_both_correct": transitions["CC"],
            "WC_improved": transitions["WC"],
            "CW_regressed": transitions["CW"],
            "WW_same": transitions["WW_same"],
            "WW_diff": transitions["WW_diff"],
            "net_improvement": net_improvement,
        },
        "student_impact": {
            "students_improved": st_improved,
            "students_unchanged": st_unchanged,
            "students_regressed": st_regressed,
        },
        "case_studies": {
            "dinnar": dinnar_case,
            "fahim": fahim_case,
        },
        "separability_ratio": {
            "Base128": dm_b128["separability_ratio"],
            "Base256": dm_b256["separability_ratio"],
            "Crop256": dm_c256["separability_ratio"],
        },
        "decision_confidence": {
            "winning_nbr_ge3_pct": {"Base128": nbr_b128, "Base256": nbr_b256, "Crop256": nbr_c256},
            "mean_winning_vote_pct": {"Base128": vote_b128, "Base256": vote_b256, "Crop256": vote_c256},
            "mean_margin_pct": {"Base128": margin_b128, "Base256": margin_b256, "Crop256": margin_c256},
        },
    }

    # 12. Save CSVs and JSONs
    print("\nSaving evaluation outputs...")

    # position_comparison.csv
    with open(os.path.join(OUT_DIR, "position_comparison.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pos_comparison_rows[0].keys()))
        w.writeheader(); w.writerows(pos_comparison_rows)

    # per_student_comparison.csv
    with open(os.path.join(OUT_DIR, "per_student_comparison.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(student_comparison_rows[0].keys()))
        w.writeheader(); w.writerows(student_comparison_rows)

    # sample1_analysis.csv & sample16_analysis.csv
    with open(os.path.join(OUT_DIR, "sample1_analysis.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sample1_analysis_rows[0].keys()))
        w.writeheader(); w.writerows(sample1_analysis_rows)

    with open(os.path.join(OUT_DIR, "sample16_analysis.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sample16_analysis_rows[0].keys()))
        w.writeheader(); w.writerows(sample16_analysis_rows)

    # sample_transitions.csv
    with open(os.path.join(OUT_DIR, "sample_transitions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sample_trans_rows[0].keys()))
        w.writeheader(); w.writerows(sample_trans_rows)

    # confusion_matrix CSVs
    for name, cm_mat in [("confusion_matrix_128.csv", cm_b128),
                         ("confusion_matrix_256.csv", cm_b256),
                         ("confusion_matrix_crop256.csv", cm_c256)]:
        with open(os.path.join(OUT_DIR, name), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["actual \\ pred"] + students)
            for i, st in enumerate(students):
                writer.writerow([st] + list(cm_mat[i]))

    # confusion_pairs.csv
    if conf_pairs_rows:
        with open(os.path.join(OUT_DIR, "confusion_pairs.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(conf_pairs_rows[0].keys()))
            w.writeheader(); w.writerows(conf_pairs_rows)

    # visual_metrics.csv
    vis_rows = [{
        "sample_idx": s["sample_idx"],
        "student": s["student_name"],
        "sample_num": s["sample_num"],
        "crop_cov": s["crop_cov"],
        "fg_base256": s["fg_base256"],
        "fg_crop256": s["fg_crop256"],
        "crop_aspect": s["crop_aspect"],
        "hog_norm_base256": s["hog_norm_base256"],
        "hog_norm_crop256": s["hog_norm_crop256"],
    } for s in samples_data]
    with open(os.path.join(OUT_DIR, "visual_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(vis_rows[0].keys()))
        w.writeheader(); w.writerows(vis_rows)

    # computational_cost.json & ab_summary.json
    with open(os.path.join(OUT_DIR, "computational_cost.json"), "w", encoding="utf-8") as f:
        json.dump(comp_cost, f, indent=2)

    with open(os.path.join(OUT_DIR, "ab_summary.json"), "w", encoding="utf-8") as f:
        json.dump(ab_summary, f, indent=2)

    # 13. Console Output Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT E FINAL REPORT SUMMARY")
    print("=" * 80)
    fmt = "{:<28} {:>10} {:>10} {:>10}"
    print(fmt.format("Metric", "BASE128", "BASE256", "CROP256"))
    print("-" * 65)
    print(fmt.format("Overall accuracy", f"{acc_b128:.2f}%", f"{acc_b256:.2f}%", f"{acc_c256:.2f}%"))
    print(fmt.format("Position #1", f"{p1_acc_b128:.2f}%", f"{p1_acc_b256:.2f}%", f"{p1_acc_c256:.2f}%"))
    print(fmt.format("Position #16", f"{p16_acc_b128:.2f}%", f"{p16_acc_b256:.2f}%", f"{p16_acc_c256:.2f}%"))
    print(fmt.format("Position #2-20", f"{prest_acc_b128:.2f}%", f"{prest_acc_b256:.2f}%", f"{prest_acc_c256:.2f}%"))
    print(fmt.format("Feature dimension", str(X_base128.shape[1]), str(X_base256.shape[1]), str(X_crop256.shape[1])))
    print(fmt.format("Vote >=3/5", f"{nbr_b128:.2f}%", f"{nbr_b256:.2f}%", f"{nbr_c256:.2f}%"))
    print(fmt.format("Mean winning vote", f"{vote_b128:.2f}%", f"{vote_b256:.2f}%", f"{vote_c256:.2f}%"))
    print(fmt.format("Mean margin", f"{margin_b128:.2f}%", f"{margin_b256:.2f}%", f"{margin_c256:.2f}%"))
    print(fmt.format("Separability ratio", f"{dm_b128['separability_ratio']:.4f}", f"{dm_b256['separability_ratio']:.4f}", f"{dm_c256['separability_ratio']:.4f}"))

    print("\nBaseline256 -> Crop256 Transitions:")
    print(f"  WRONG -> CORRECT (WC)  : {transitions['WC']} samples ({transitions['WC']/n*100:.1f}%)")
    print(f"  CORRECT -> WRONG (CW)  : {transitions['CW']} samples ({transitions['CW']/n*100:.1f}%)")
    print(f"  Net improvement        : {net_improvement:+d} samples")
    print(f"  Both correct (CC)      : {transitions['CC']} samples")
    print(f"  Both wrong same (WW)   : {transitions['WW_same']} samples")
    print(f"  Both wrong diff (WW)   : {transitions['WW_diff']} samples")

    print("\nStudents Summary:")
    print(f"  Students improved      : {st_improved} / {len(students)}")
    print(f"  Students unchanged     : {st_unchanged} / {len(students)}")
    print(f"  Students regressed     : {st_regressed} / {len(students)}")

    print("\nCase Studies:")
    print(f"  Dinnar: {dinnar_case['acc_base128']:.1f}% -> {dinnar_case['acc_base256']:.1f}% -> {dinnar_case['acc_crop256']:.1f}% (Main conf: {dinnar_case['main_confusion_crop256']})")
    print(f"  Fahim : {fahim_case['acc_base128']:.1f}% -> {fahim_case['acc_base256']:.1f}% -> {fahim_case['acc_crop256']:.1f}% (Main conf: {fahim_case['main_confusion_crop256']})")

    print("=" * 80)
    print("EXPERIMENT E EVALUATION FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment_e()
