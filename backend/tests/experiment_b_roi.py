"""
tests/experiment_b_roi.py — Experiment B: Text-Density ROI A/B Comparison
==========================================================================
Baseline   : extract_roi()       — union bounding-box of all external contours
Experiment : extract_roi_text_density() — contour-filtered, text-only ROI
             with fallback to baseline when no sufficient text area is found.

SCOPE RULES (must NOT be violated):
  - Production extract_roi() is NOT modified.
  - config.py is NOT modified.
  - classifier.py / frontend / API are NOT changed.
  - No retraining of the production model.
  - No commit/push until user reviews A/B results.

Single variable changed: ROI SEGMENTATION only.
All other parameters are identical between baseline and experiment:
  IMAGE_SIZE=128x128, HOG orientations=9, pixels_per_cell=8x8,
  cells_per_block=2x2, block_norm=L2-Hys, K=5, Euclidean, weights=distance.

Output directory: backend/tests/evaluation_results/roi_experiment/
  ab_comparison_report.json
  page1_comparison.csv
  misclassification_comparison.csv
  fahim_fathur_analysis.json
  visualizations/<student>/sample_XX_idYY.png  (7-panel per image)
"""

import os
import sys
import json
import csv
import sqlite3
import numpy as np
import cv2
from collections import defaultdict
from skimage.feature import hog

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import (
    DB_PATH, DATASET_RAW_DIR, DATASET_PROCESSED_DIR,
    IMAGE_SIZE, GAUSSIAN_BLUR_KERNEL, MEDIAN_BLUR_KERNEL, MORPH_KERNEL_SIZE,
    HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK, HOG_BLOCK_NORM,
    KNN_N_NEIGHBORS,
)
from preprocessing.image_processor import (
    normalize_orientation, convert_to_grayscale,
    apply_gaussian_blur, apply_otsu_threshold, remove_noise,
    extract_roi, resize_with_aspect_ratio,
)
from model.classifier import euclidean_to_similarity

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
EXPR_OUT_DIR = os.path.join(BASE_DIR, "tests", "evaluation_results", "roi_experiment")
VIS_OUT_DIR  = os.path.join(EXPR_OUT_DIR, "visualizations")
os.makedirs(VIS_OUT_DIR,  exist_ok=True)

KNN_K = KNN_N_NEIGHBORS   # 5
EPS   = 1e-7


# ===========================================================================
# EXPERIMENT ROI: extract_roi_text_density()
# ===========================================================================
def extract_roi_text_density(image: np.ndarray, padding_ratio: float = 0.05):
    """
    Text-density–aware ROI extraction. Isolated experiment function.
    Does NOT overwrite production extract_roi().

    Strategy (CV-only, no OCR/CNN/YOLO):
    1. Find all external contours.
    2. Filter out:
       a. Very large contours likely to be border lines / page frames
          (area > 5% of total image area).
       b. Extremely elongated contours (aspect ratio > 15:1 or < 1:15)
          → horizontal/vertical lines, table rules.
       c. Very tiny isolated dots (area < min_dot threshold).
    3. From the remaining "text-likely" contours, compute a tight
       bounding-box union.
    4. Fallback to baseline extract_roi() if:
       - No contours survive filtering.
       - Surviving ROI area < 2% of image (too small).
       - Surviving foreground pixel ratio < 0.01 (too sparse, likely noise).

    Returns: (roi_crop, used_fallback: bool)
    """
    h_img, w_img = image.shape[:2]
    total_area = float(h_img * w_img)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return extract_roi(image, padding_ratio), True

    # --- Thresholds calibrated for A4-scale handwriting pages ---
    max_contour_area_ratio = 0.05   # > 5% of image → likely a page border / table
    min_dot_area           = 8      # < 8 px² → isolated noise dot
    max_aspect_line        = 15.0   # very elongated → line/table rule

    text_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_dot_area:
            continue
        if area / total_area > max_contour_area_ratio:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = bw / max(bh, 1)
        if aspect > max_aspect_line or aspect < (1.0 / max_aspect_line):
            continue
        text_contours.append(c)

    # --- Fallback condition 1: nothing survived ---
    if not text_contours:
        return extract_roi(image, padding_ratio), True

    # --- Union bounding-box of surviving text contours ---
    all_pts = np.concatenate(text_contours, axis=0)
    tx, ty, tw, th = cv2.boundingRect(all_pts)
    roi_area = float(tw * th)

    # --- Fallback condition 2: ROI too small ---
    if roi_area / total_area < 0.02:
        return extract_roi(image, padding_ratio), True

    # --- Fallback condition 3: foreground pixel ratio in proposed ROI too sparse ---
    roi_slice = image[ty:ty + th, tx:tx + tw]
    fg_ratio = float(np.count_nonzero(roi_slice)) / max(roi_area, 1)
    if fg_ratio < 0.01:
        return extract_roi(image, padding_ratio), True

    # --- Add padding ---
    padding = max(4, int(min(tw, th) * padding_ratio))
    x1 = max(0, tx - padding)
    y1 = max(0, ty - padding)
    x2 = min(w_img, tx + tw + padding)
    y2 = min(h_img, ty + th + padding)

    roi_crop = image[y1:y2, x1:x2]
    return roi_crop, False


# ===========================================================================
# HOG extraction helper
# ===========================================================================
def compute_hog(image_128: np.ndarray):
    feat, hog_vis = hog(
        image_128,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        visualize=True,
        feature_vector=True,
    )
    return feat, hog_vis


# ===========================================================================
# LOOCV KNN simulation (distance-weighted, K=5)
# ===========================================================================
def run_loocv(X, y, k=KNN_K):
    """
    Full Leave-One-Out Cross-Validation with distance-weighted KNN.
    Returns list of per-sample result dicts.
    """
    n = len(X)
    # Pairwise distance matrix
    dot  = np.dot(X, X.T)
    sq   = np.diag(dot)
    dsq  = np.maximum(0.0, sq[:, None] + sq[None, :] - 2 * dot)
    dist = np.sqrt(dsq)
    np.fill_diagonal(dist, 0.0)

    unique_classes = sorted(set(y))
    results = []

    for i in range(n):
        true_label = y[i]
        row   = np.delete(dist[i], i)
        lbls  = np.delete(y, i)

        sorted_idx  = np.argsort(row)
        topk_idx    = sorted_idx[:k]
        topk_dists  = row[topk_idx]
        topk_labels = lbls[topk_idx]

        weights_k   = 1.0 / np.maximum(topk_dists, EPS)
        total_w     = weights_k.sum()

        vote_weights    = defaultdict(float)
        vote_counts     = defaultdict(int)
        for d_val, lbl, wv in zip(topk_dists, topk_labels, weights_k):
            vote_weights[lbl] += wv
            vote_counts[lbl]  += 1

        sorted_votes = sorted(vote_weights.items(), key=lambda x: -x[1])
        pred_label   = sorted_votes[0][0]
        pred_share   = sorted_votes[0][1] / total_w * 100.0
        second_share = sorted_votes[1][1] / total_w * 100.0 if len(sorted_votes) > 1 else 0.0

        true_neighbors  = vote_counts.get(true_label, 0)
        true_vote_share = vote_weights.get(true_label, 0.0) / total_w * 100.0

        # Min distance to each class
        class_min_dists = {}
        for cls in unique_classes:
            mask = (lbls == cls)
            class_min_dists[cls] = float(row[mask].min()) if mask.any() else float("inf")

        neighbors_detail = [
            {
                "rank": rank + 1,
                "name": lbl,
                "distance": round(float(d), 4),
                "similarity_pct": round(euclidean_to_similarity(float(d)), 2),
                "vote_contrib_pct": round(float(wv / total_w * 100.0), 2),
            }
            for rank, (d, lbl, wv) in enumerate(zip(topk_dists, topk_labels, weights_k))
        ]

        results.append({
            "sample_idx":         i,
            "true_label":         true_label,
            "pred_label":         pred_label,
            "is_correct":         (pred_label == true_label),
            "pred_vote_share":    round(float(pred_share), 2),
            "second_vote_share":  round(float(second_share), 2),
            "margin":             round(float(pred_share - second_share), 2),
            "true_neighbors":     true_neighbors,
            "true_vote_share":    round(float(true_vote_share), 2),
            "min_dist_true":      round(class_min_dists.get(true_label, 999.0), 4),
            "min_dist_pred":      round(class_min_dists.get(pred_label, 999.0), 4),
            "nearest_dist":       round(float(topk_dists[0]), 4),
            "neighbors_detail":   neighbors_detail,
        })

    return results, dist


# ===========================================================================
# Distance-space metrics helper
# ===========================================================================
def compute_distance_metrics(X, y):
    dot  = np.dot(X, X.T)
    sq   = np.diag(dot)
    dsq  = np.maximum(0.0, sq[:, None] + sq[None, :] - 2 * dot)
    dist = np.sqrt(dsq)
    np.fill_diagonal(dist, 0.0)

    unique_classes = sorted(set(y))
    class_indices  = {c: np.where(np.array(y) == c)[0] for c in unique_classes}

    intra_all, inter_all = [], []
    class_stats = {}

    for c in unique_classes:
        ci  = class_indices[c]
        oi  = np.where(np.array(y) != c)[0]
        sub = dist[np.ix_(ci, ci)]
        iu  = np.triu_indices(len(ci), k=1)
        intra = sub[iu]
        inter = dist[np.ix_(ci, oi)].flatten()
        intra_all.extend(intra)
        inter_all.extend(inter)
        class_stats[c] = {
            "intra_mean":   float(np.mean(intra)),
            "intra_std":    float(np.std(intra)),
            "intra_min":    float(np.min(intra)),
            "intra_median": float(np.median(intra)),
            "intra_max":    float(np.max(intra)),
            "inter_mean":   float(np.mean(inter)),
            "inter_min":    float(np.min(inter)),
            "inter_median": float(np.median(inter)),
            "sep_ratio":    float(np.mean(inter) / np.mean(intra)) if np.mean(intra) > 0 else 0,
        }

    return {
        "overall_intra_mean":   float(np.mean(intra_all)),
        "overall_intra_std":    float(np.std(intra_all)),
        "overall_intra_min":    float(np.min(intra_all)),
        "overall_intra_max":    float(np.max(intra_all)),
        "overall_inter_mean":   float(np.mean(inter_all)),
        "overall_inter_std":    float(np.std(inter_all)),
        "overall_inter_min":    float(np.min(inter_all)),
        "overall_inter_max":    float(np.max(inter_all)),
        "separability_ratio":   float(np.mean(inter_all) / np.mean(intra_all)) if np.mean(intra_all) > 0 else 0,
        "per_class":            class_stats,
    }, intra_all, inter_all


# ===========================================================================
# Neighbor vote distribution helper
# ===========================================================================
def vote_distribution(loocv_results):
    dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for r in loocv_results:
        n = r["true_neighbors"]
        dist[n] = dist.get(n, 0) + 1
    total = len(loocv_results)
    return {
        "counts": dist,
        "pct_1_of_5": round(dist.get(1, 0) / total * 100, 2),
        "pct_2_of_5": round(dist.get(2, 0) / total * 100, 2),
        "pct_ge3_of_5": round(sum(dist.get(k, 0) for k in [3, 4, 5]) / total * 100, 2),
        "mean_winning_vote_share": round(float(np.mean([r["pred_vote_share"] for r in loocv_results])), 2),
        "mean_margin": round(float(np.mean([r["margin"] for r in loocv_results])), 2),
    }


# ===========================================================================
# Foreground-pixel ratio of 128×128 image
# ===========================================================================
def fg_ratio_128(img_128):
    return float(np.count_nonzero(img_128)) / (img_128.shape[0] * img_128.shape[1])


# ===========================================================================
# 7-panel visualization saver
# ===========================================================================
def make_7panel(
    img_bgr_oriented,
    baseline_bbox,      # (x, y, w, h) in full-res binary
    exp_bbox,           # (x, y, w, h) or None if fallback
    baseline_128,
    exp_128,
    baseline_hog_vis,
    exp_hog_vis,
    used_fallback,
    sample_info,
    save_path,
):
    h_orig, w_orig = img_bgr_oriented.shape[:2]
    scale = 256.0 / h_orig

    def thumb(img_any, tw, th):
        """Resize to (tw, th) preserving aspect within bounds."""
        if len(img_any.shape) == 2:
            img_any = cv2.cvtColor(img_any, cv2.COLOR_GRAY2BGR)
        return cv2.resize(img_any, (tw, th), interpolation=cv2.INTER_AREA)

    def add_header(panel, title, sub=""):
        h, w = panel.shape[:2]
        hdr = np.full((42, w, 3), 28, dtype=np.uint8)
        cv2.putText(hdr, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(hdr, sub,   (4, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)
        return np.vstack([hdr, panel])

    target_h = 256

    # Panel 1 — original + baseline bbox (red)
    p1 = cv2.resize(img_bgr_oriented, (int(w_orig * scale), target_h), interpolation=cv2.INTER_AREA)
    bx, by, bw, bh = baseline_bbox
    sx = p1.shape[1] / w_orig
    sy = target_h / h_orig
    cv2.rectangle(p1, (int(bx * sx), int(by * sy)), (int((bx + bw) * sx), int((by + bh) * sy)), (0, 0, 255), 2)
    p1 = add_header(p1, "1. Original + Baseline BBox",
                    f"{w_orig}x{h_orig}  cov={bw*bh/(w_orig*h_orig)*100:.0f}%")

    # Panel 2 — original + experiment bbox (cyan / yellow if fallback)
    p2 = cv2.resize(img_bgr_oriented, (int(w_orig * scale), target_h), interpolation=cv2.INTER_AREA)
    if exp_bbox is not None:
        ex, ey, ew, eh = exp_bbox
        color = (0, 200, 0) if not used_fallback else (0, 200, 255)
        cv2.rectangle(p2, (int(ex * sx), int(ey * sy)), (int((ex + ew) * sx), int((ey + eh) * sy)), color, 2)
        sub2 = f"cov={ew*eh/(w_orig*h_orig)*100:.0f}%  {'FALLBACK' if used_fallback else 'text-density'}"
    else:
        sub2 = "no bbox"
    p2 = add_header(p2, "2. Original + Experiment BBox", sub2)

    # Panels 3 & 4 — baseline and experiment 128×128
    b128_disp = thumb(baseline_128, 256, 256)
    e128_disp = thumb(exp_128, 256, 256)
    p3 = add_header(b128_disp, "3. Baseline 128x128",  f"fg={fg_ratio_128(baseline_128):.3f}")
    p4 = add_header(e128_disp, "4. Experiment 128x128", f"fg={fg_ratio_128(exp_128):.3f}")

    # Panels 5 & 6 — HOG visualizations
    def hog_display(hv):
        u8 = np.uint8(np.clip(hv * 255, 0, 255))
        colored = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
        return cv2.resize(colored, (256, 256), interpolation=cv2.INTER_NEAREST)

    p5 = add_header(hog_display(baseline_hog_vis), "5. Baseline HOG",    "")
    p6 = add_header(hog_display(exp_hog_vis),      "6. Experiment HOG",  "")

    # Panel 7 — student info text card
    p7_bg = np.full((target_h + 42, 256, 3), 20, dtype=np.uint8)
    lines = [
        sample_info["student_name"][:26],
        f"sample #{sample_info['sample_num']}  id={sample_info['db_id']}",
        f"file: {sample_info['filename'][:24]}",
        f"Fallback: {'YES' if used_fallback else 'NO'}",
    ]
    for li, txt in enumerate(lines):
        cv2.putText(p7_bg, txt, (6, 30 + li * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    p7 = p7_bg  # already has header space baked in

    # Ensure all panels have same height
    all_panels = [p1, p2, p3, p4, p5, p6, p7]
    max_h = max(p.shape[0] for p in all_panels)
    padded = []
    for p in all_panels:
        ph = p.shape[0]
        if ph < max_h:
            pad = np.zeros((max_h - ph, p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)

    montage = np.hstack(padded)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, montage)


# ===========================================================================
# MAIN EXPERIMENT
# ===========================================================================
def run_experiment_b():
    print("=" * 80)
    print("EXPERIMENT B: TEXT-DENSITY ROI — A/B COMPARISON")
    print("=" * 80)

    # ----- Load dataset from DB -----
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()
    rows = cur.execute(
        "SELECT id, student_name, original_filename, saved_filename, file_path "
        "FROM dataset ORDER BY student_name ASC, original_filename ASC, saved_filename ASC"
    ).fetchall()
    conn.close()

    n_records = len(rows)
    n_classes  = len(set(r["student_name"] for r in rows))
    print(f"Loaded {n_records} records across {n_classes} students.")

    # ----- Process all images -----
    baseline_features = []
    exp_features      = []
    labels            = []
    samples_meta      = []
    fallback_count    = 0
    fallback_details  = []

    print(f"\nProcessing all {n_records} images...")
    for idx, r in enumerate(rows):
        db_id    = r["id"]
        student  = r["student_name"]
        filename = r["original_filename"] or ""
        raw_name = r["saved_filename"] or os.path.basename(r["file_path"])
        raw_path = os.path.join(DATASET_RAW_DIR, raw_name)

        if not os.path.exists(raw_path):
            print(f"  [SKIP] Not found: {raw_path}")
            continue

        img_bgr = cv2.imread(raw_path)
        if img_bgr is None:
            print(f"  [SKIP] Cannot read: {raw_path}")
            continue

        # --- Shared pipeline steps (identical in both branches) ---
        oriented = normalize_orientation(img_bgr)
        gray     = convert_to_grayscale(oriented)
        blurred  = apply_gaussian_blur(gray)
        binary   = apply_otsu_threshold(blurred)
        denoised = remove_noise(binary)

        h_img, w_img = denoised.shape[:2]

        # ---- BASELINE branch ----
        baseline_roi = extract_roi(denoised)
        b128         = resize_with_aspect_ratio(baseline_roi, IMAGE_SIZE)
        b_feat, b_hog_vis = compute_hog(b128)

        # Baseline bbox (for visualization)
        b_contours, _ = cv2.findContours(denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if b_contours:
            all_pts = np.concatenate(b_contours, axis=0)
            bx, by, bw, bh = cv2.boundingRect(all_pts)
        else:
            bx, by, bw, bh = 0, 0, w_img, h_img
        baseline_bbox_cov = (bw * bh) / max(1, w_img * h_img)

        # ---- EXPERIMENT branch ----
        exp_roi, used_fallback = extract_roi_text_density(denoised)
        e128                   = resize_with_aspect_ratio(exp_roi, IMAGE_SIZE)
        e_feat, e_hog_vis      = compute_hog(e128)

        # Experiment bbox (for visualization): re-run to get coords
        e_contours, _ = cv2.findContours(denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = float(w_img * h_img)
        if not used_fallback and e_contours:
            text_cs = [
                c for c in e_contours
                if (8 <= cv2.contourArea(c) <= 0.05 * total_area)
                and (
                    (bw2 := cv2.boundingRect(c)[2]) / max(cv2.boundingRect(c)[3], 1) <= 15.0
                    and (bw2) / max(cv2.boundingRect(c)[3], 1) >= (1.0 / 15.0)
                )
            ]
            if text_cs:
                all_epts = np.concatenate(text_cs, axis=0)
                ex, ey, ew, eh = cv2.boundingRect(all_epts)
                exp_bbox = (ex, ey, ew, eh)
            else:
                exp_bbox = (bx, by, bw, bh)
        else:
            exp_bbox = (bx, by, bw, bh)   # fallback → same as baseline bbox

        exp_bbox_cov = (exp_bbox[2] * exp_bbox[3]) / max(1, total_area)

        if used_fallback:
            fallback_count += 1
            fallback_details.append({
                "sample_idx": idx,
                "db_id": db_id,
                "student_name": student,
                "filename": filename,
            })

        # Metadata for this sample
        sample_num = (idx % 20) + 1
        sample_meta = {
            "sample_idx":    idx,
            "db_id":         db_id,
            "student_name":  student,
            "filename":      filename,
            "sample_num":    sample_num,
            "baseline_cov":  round(float(baseline_bbox_cov), 4),
            "exp_cov":       round(float(exp_bbox_cov), 4),
            "baseline_fg":   round(fg_ratio_128(b128), 4),
            "exp_fg":        round(fg_ratio_128(e128), 4),
            "used_fallback": used_fallback,
        }
        samples_meta.append(sample_meta)
        baseline_features.append(b_feat)
        exp_features.append(e_feat)
        labels.append(student)

        # ---- 7-panel visualization ----
        safe_name = student.replace(" ", "_").replace("'", "")
        vis_dir   = os.path.join(VIS_OUT_DIR, safe_name)
        vis_path  = os.path.join(vis_dir, f"sample_{sample_num:02d}_id{db_id}.png")
        if not os.path.exists(vis_path):
            make_7panel(
                img_bgr_oriented=oriented,
                baseline_bbox=(bx, by, bw, bh),
                exp_bbox=exp_bbox,
                baseline_128=b128,
                exp_128=e128,
                baseline_hog_vis=b_hog_vis,
                exp_hog_vis=e_hog_vis,
                used_fallback=used_fallback,
                sample_info=sample_meta,
                save_path=vis_path,
            )

        if (idx + 1) % 36 == 0:
            print(f"  Processed {idx + 1}/{n_records} ...")

    print(f"\nAll {len(labels)} images processed. Fallback count: {fallback_count}/{len(labels)} "
          f"({fallback_count/len(labels)*100:.1f}%)")

    X_base = np.array(baseline_features)
    X_exp  = np.array(exp_features)
    y_arr  = np.array(labels)
    n      = len(y_arr)

    # =========== METRIC COMPUTATION ===========
    print("\nComputing distance metrics (baseline)...")
    base_dist_metrics, base_intra, base_inter = compute_distance_metrics(X_base, labels)
    print("Computing distance metrics (experiment)...")
    exp_dist_metrics,  exp_intra,  exp_inter  = compute_distance_metrics(X_exp,  labels)

    print("\nRunning LOOCV KNN (baseline)...")
    base_loocv, _ = run_loocv(X_base, y_arr)
    print("Running LOOCV KNN (experiment)...")
    exp_loocv,  _ = run_loocv(X_exp,  y_arr)

    # --- Accuracy metrics ---
    def accuracy_metrics(loocv_results, meta):
        correct_total   = sum(1 for r in loocv_results if r["is_correct"])
        page1_results   = [r for r in loocv_results if meta[r["sample_idx"]]["sample_num"] == 1]
        page_oth        = [r for r in loocv_results if meta[r["sample_idx"]]["sample_num"] > 1]
        p1_acc          = sum(1 for r in page1_results if r["is_correct"]) / max(1, len(page1_results)) * 100
        po_acc          = sum(1 for r in page_oth      if r["is_correct"]) / max(1, len(page_oth))      * 100
        return {
            "overall_accuracy":   round(correct_total / n * 100, 2),
            "page1_accuracy":     round(p1_acc, 2),
            "page2_20_accuracy":  round(po_acc, 2),
            "total_correct":      correct_total,
            "total_misclassified": n - correct_total,
        }

    base_acc = accuracy_metrics(base_loocv, samples_meta)
    exp_acc  = accuracy_metrics(exp_loocv,  samples_meta)

    # --- Neighbor vote distribution ---
    base_vote_dist = vote_distribution(base_loocv)
    exp_vote_dist  = vote_distribution(exp_loocv)

    # --- ROI coverage stats ---
    base_covs = [s["baseline_cov"] for s in samples_meta]
    exp_covs  = [s["exp_cov"]      for s in samples_meta]

    def cov_stats(arr):
        return {
            "mean":   round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "std":    round(float(np.std(arr)), 4),
            "pct_ge85": round(sum(1 for v in arr if v >= 0.85) / len(arr) * 100, 2),
        }

    # --- Confusion matrix helper ---
    unique_classes = sorted(set(labels))
    def confusion_matrix(loocv_results):
        cls2idx = {c: i for i, c in enumerate(unique_classes)}
        cm = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
        for r in loocv_results:
            cm[cls2idx[r["true_label"]], cls2idx[r["pred_label"]]] += 1
        return cm.tolist()

    base_cm = confusion_matrix(base_loocv)
    exp_cm  = confusion_matrix(exp_loocv)

    # =========== PAGE-1 COMPARISON TABLE ===========
    print("\nBuilding page-1 comparison table...")
    page1_rows = []
    for r_base, r_exp in zip(base_loocv, exp_loocv):
        meta = samples_meta[r_base["sample_idx"]]
        if meta["sample_num"] != 1:
            continue
        page1_rows.append({
            "student_name":        meta["student_name"],
            "filename":            meta["filename"],
            # Baseline
            "base_pred":           r_base["pred_label"],
            "base_correct":        r_base["is_correct"],
            "base_vote":           r_base["true_neighbors"],
            "base_nearest_dist":   r_base["nearest_dist"],
            "base_roi_cov":        meta["baseline_cov"],
            "base_pred_share":     r_base["pred_vote_share"],
            # Experiment
            "exp_pred":            r_exp["pred_label"],
            "exp_correct":         r_exp["is_correct"],
            "exp_vote":            r_exp["true_neighbors"],
            "exp_nearest_dist":    r_exp["nearest_dist"],
            "exp_roi_cov":         meta["exp_cov"],
            "exp_pred_share":      r_exp["pred_vote_share"],
            "used_fallback":       meta["used_fallback"],
            # Change analysis
            "base_fail_exp_fix":   (not r_base["is_correct"] and r_exp["is_correct"]),
            "base_ok_exp_broke":   (r_base["is_correct"] and not r_exp["is_correct"]),
        })

    base_p1_failures = sum(1 for r in page1_rows if not r["base_correct"])
    exp_p1_fixes     = sum(1 for r in page1_rows if r["base_fail_exp_fix"])
    exp_p1_regressions = sum(1 for r in page1_rows if r["base_ok_exp_broke"])
    print(f"  Page-1: baseline failures={base_p1_failures}/18, "
          f"experiment fixes={exp_p1_fixes}, regressions={exp_p1_regressions}")

    # =========== MISCLASSIFICATION COMPARISON ===========
    base_misclass_set = {r["sample_idx"] for r in base_loocv if not r["is_correct"]}
    exp_misclass_set  = {r["sample_idx"] for r in exp_loocv  if not r["is_correct"]}

    misclass_rows = []
    all_mc_idxs   = base_misclass_set | exp_misclass_set
    for idx in sorted(all_mc_idxs):
        meta   = samples_meta[idx]
        rb     = base_loocv[idx]
        re     = exp_loocv[idx]
        misclass_rows.append({
            "sample_idx":    idx,
            "student_name":  meta["student_name"],
            "sample_num":    meta["sample_num"],
            "filename":      meta["filename"],
            "base_correct":  rb["is_correct"],
            "base_pred":     rb["pred_label"],
            "base_share":    rb["pred_vote_share"],
            "exp_correct":   re["is_correct"],
            "exp_pred":      re["pred_label"],
            "exp_share":     re["pred_vote_share"],
            "used_fallback": meta["used_fallback"],
        })

    # Regression list (baseline correct → experiment wrong)
    regressions = [
        {"student_name": samples_meta[idx]["student_name"],
         "sample_num":   samples_meta[idx]["sample_num"],
         "filename":     samples_meta[idx]["filename"]}
        for idx in base_misclass_set.symmetric_difference(exp_misclass_set)
        if idx not in base_misclass_set and idx in exp_misclass_set
    ]

    # =========== FAHIM vs FATHUR CASE STUDY ===========
    fahim_name  = "Fahim J Mujaddid"
    fathur_name = "Fathurrahman Nugroho"

    def fahim_fathur_analysis(X, loocv_results, label="baseline"):
        y_np = np.array(labels)
        fi   = np.where(y_np == fahim_name)[0]
        fr   = np.where(y_np == fathur_name)[0]

        if len(fi) == 0 or len(fr) == 0:
            return {"note": "One or both classes not in dataset"}

        dot  = np.dot(X, X.T)
        sq   = np.diag(dot)
        dsq  = np.maximum(0.0, sq[:, None] + sq[None, :] - 2 * dot)
        dist = np.sqrt(dsq)

        fahim_sub  = dist[np.ix_(fi, fi)]; iu = np.triu_indices(len(fi), k=1)
        fathur_sub = dist[np.ix_(fr, fr)]; iu2 = np.triu_indices(len(fr), k=1)
        cross      = dist[np.ix_(fi, fr)].flatten()

        fahim_intra  = fahim_sub[iu]
        fathur_intra = fathur_sub[iu2]

        # Error counts
        f_to_r = sum(1 for r in loocv_results
                     if r["true_label"] == fahim_name and r["pred_label"] == fathur_name)
        r_to_f = sum(1 for r in loocv_results
                     if r["true_label"] == fathur_name and r["pred_label"] == fahim_name)

        # Nearest foreign class for Fahim
        fahim_all_loocv  = [r for r in loocv_results if r["true_label"] == fahim_name  and not r["is_correct"]]
        fathur_all_loocv = [r for r in loocv_results if r["true_label"] == fathur_name and not r["is_correct"]]

        return {
            "label":                     label,
            "fahim_intra_mean":          round(float(np.mean(fahim_intra)),  4),
            "fahim_intra_std":           round(float(np.std(fahim_intra)),   4),
            "fathur_intra_mean":         round(float(np.mean(fathur_intra)), 4),
            "fathur_intra_std":          round(float(np.std(fathur_intra)),  4),
            "cross_mean":                round(float(np.mean(cross)), 4),
            "cross_min":                 round(float(np.min(cross)),  4),
            "cross_max":                 round(float(np.max(cross)),  4),
            "cross_below_fahim_intra_mean_pct":  round(np.mean(cross < np.mean(fahim_intra))  * 100, 2),
            "cross_below_fathur_intra_mean_pct": round(np.mean(cross < np.mean(fathur_intra)) * 100, 2),
            "fahim_misclassified_as_fathur":  f_to_r,
            "fathur_misclassified_as_fahim":  r_to_f,
            "fahim_total_misclassified":  len(fahim_all_loocv),
            "fathur_total_misclassified": len(fathur_all_loocv),
        }

    fahim_fathur = {
        "baseline":   fahim_fathur_analysis(X_base, base_loocv, "baseline"),
        "experiment": fahim_fathur_analysis(X_exp,  exp_loocv,  "experiment"),
    }

    # =========== FINAL REPORT ===========
    report = {
        "experiment": "B — Text-Density ROI vs Baseline",
        "baseline": {
            "roi_method": "extract_roi(): union boundingRect of ALL external contours",
            "roi_coverage": cov_stats(base_covs),
            "foreground_ratio_128": {
                "mean": round(float(np.mean([s["baseline_fg"] for s in samples_meta])), 4),
                "std":  round(float(np.std( [s["baseline_fg"] for s in samples_meta])), 4),
            },
            "distance_metrics": base_dist_metrics,
            "accuracy": base_acc,
            "confusion_matrix": {"classes": unique_classes, "matrix": base_cm},
            "vote_distribution": base_vote_dist,
        },
        "experiment": {
            "roi_method": "extract_roi_text_density(): filtered contours (area, aspect, size)",
            "fallback_count": fallback_count,
            "fallback_pct":   round(fallback_count / n * 100, 2),
            "fallback_samples": fallback_details,
            "roi_coverage": cov_stats(exp_covs),
            "foreground_ratio_128": {
                "mean": round(float(np.mean([s["exp_fg"] for s in samples_meta])), 4),
                "std":  round(float(np.std( [s["exp_fg"] for s in samples_meta])), 4),
            },
            "distance_metrics": exp_dist_metrics,
            "accuracy": exp_acc,
            "confusion_matrix": {"classes": unique_classes, "matrix": exp_cm},
            "vote_distribution": exp_vote_dist,
        },
        "comparison_summary": {
            "delta_overall_accuracy":   round(exp_acc["overall_accuracy"] - base_acc["overall_accuracy"], 2),
            "delta_page1_accuracy":     round(exp_acc["page1_accuracy"]   - base_acc["page1_accuracy"],   2),
            "delta_page2_20_accuracy":  round(exp_acc["page2_20_accuracy"]- base_acc["page2_20_accuracy"],2),
            "delta_separability_ratio": round(exp_dist_metrics["separability_ratio"] - base_dist_metrics["separability_ratio"], 4),
            "delta_intra_mean":         round(exp_dist_metrics["overall_intra_mean"] - base_dist_metrics["overall_intra_mean"], 4),
            "delta_inter_mean":         round(exp_dist_metrics["overall_inter_mean"] - base_dist_metrics["overall_inter_mean"], 4),
            "page1_baseline_failures":  base_p1_failures,
            "page1_experiment_fixes":   exp_p1_fixes,
            "page1_experiment_regressions": exp_p1_regressions,
            "total_regressions_all_pages":  len(regressions),
            "regression_samples":           regressions,
        },
    }

    # =========== SAVE OUTPUTS ===========
    # 1. Main JSON report
    json_path = os.path.join(EXPR_OUT_DIR, "ab_comparison_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, (int, np.integer)) else
                                    float(o) if isinstance(o, (float, np.floating)) else str(o))
    print(f"\nSaved: {json_path}")

    # 2. Page-1 comparison CSV
    p1_csv_path = os.path.join(EXPR_OUT_DIR, "page1_comparison.csv")
    p1_fieldnames = list(page1_rows[0].keys()) if page1_rows else []
    with open(p1_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=p1_fieldnames)
        w.writeheader()
        w.writerows(page1_rows)
    print(f"Saved: {p1_csv_path}")

    # 3. Misclassification comparison CSV
    mc_csv_path = os.path.join(EXPR_OUT_DIR, "misclassification_comparison.csv")
    mc_fieldnames = list(misclass_rows[0].keys()) if misclass_rows else []
    with open(mc_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=mc_fieldnames)
        w.writeheader()
        w.writerows(misclass_rows)
    print(f"Saved: {mc_csv_path}")

    # 4. Fahim-Fathur analysis JSON
    ff_path = os.path.join(EXPR_OUT_DIR, "fahim_fathur_analysis.json")
    with open(ff_path, "w", encoding="utf-8") as f:
        json.dump(fahim_fathur, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, (int, np.integer)) else
                                    float(o) if isinstance(o, (float, np.floating)) else str(o))
    print(f"Saved: {ff_path}")

    # =========== PRINT SUMMARY ===========
    print("\n" + "=" * 80)
    print("A/B SUMMARY — BASELINE  vs  EXPERIMENT")
    print("=" * 80)
    fmt = "{:<35} {:>12}  {:>12}  {:>12}"
    print(fmt.format("Metric", "Baseline", "Experiment", "Delta"))
    print("-" * 75)
    rows_summary = [
        ("Overall Accuracy (%)",         base_acc["overall_accuracy"],    exp_acc["overall_accuracy"]),
        ("Page-1 Accuracy (%)",          base_acc["page1_accuracy"],      exp_acc["page1_accuracy"]),
        ("Page 2-20 Accuracy (%)",       base_acc["page2_20_accuracy"],   exp_acc["page2_20_accuracy"]),
        ("Intra-class Mean Dist",        base_dist_metrics["overall_intra_mean"], exp_dist_metrics["overall_intra_mean"]),
        ("Inter-class Mean Dist",        base_dist_metrics["overall_inter_mean"], exp_dist_metrics["overall_inter_mean"]),
        ("Separability Ratio",           base_dist_metrics["separability_ratio"], exp_dist_metrics["separability_ratio"]),
        ("ROI Coverage Mean",            float(np.mean(base_covs)),       float(np.mean(exp_covs))),
        ("FG Ratio @128x128 Mean",       float(np.mean([s["baseline_fg"] for s in samples_meta])),
                                          float(np.mean([s["exp_fg"] for s in samples_meta]))),
        ("Mean Winning Vote Share (%)",  base_vote_dist["mean_winning_vote_share"], exp_vote_dist["mean_winning_vote_share"]),
        ("Mean Margin (#1-#2 %)",        base_vote_dist["mean_margin"],   exp_vote_dist["mean_margin"]),
        ("Vote 1/5 (%)",                 base_vote_dist["pct_1_of_5"],    exp_vote_dist["pct_1_of_5"]),
        ("Vote 2/5 (%)",                 base_vote_dist["pct_2_of_5"],    exp_vote_dist["pct_2_of_5"]),
        ("Vote >=3/5 (%)",               base_vote_dist["pct_ge3_of_5"],  exp_vote_dist["pct_ge3_of_5"]),
    ]
    for label, bv, ev in rows_summary:
        delta = ev - bv
        sign  = "+" if delta >= 0 else ""
        print(fmt.format(label, f"{bv:.3f}", f"{ev:.3f}", f"{sign}{delta:.3f}"))

    print(f"\nFallback count (exp): {fallback_count}/{n} ({fallback_count/n*100:.1f}%)")
    print(f"Page-1 failures fixed by experiment: {exp_p1_fixes}/{base_p1_failures}")
    print(f"Page-1 regressions (baseline OK → exp FAIL): {exp_p1_regressions}")
    print(f"Total regressions (all pages): {len(regressions)}")
    if regressions:
        print("  Regression samples:")
        for rg in regressions:
            print(f"    {rg['student_name']}  sample#{rg['sample_num']}  ({rg['filename']})")

    print(f"\nViz output:  {VIS_OUT_DIR}")
    print(f"JSON report: {json_path}")
    print("=" * 80)
    print("EXPERIMENT B COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment_b()
