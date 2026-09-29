"""
tests/experiment_c_resolution.py — Experiment C: 128x128 vs 256x256 Resolution A/B
=====================================================================================
Baseline   : Identical preprocessing + baseline ROI + resize 128x128 + HOG + KNN
Experiment : Identical preprocessing + baseline ROI + resize 256x256 + HOG + KNN

Single variable changed: FINAL IMAGE SIZE only (128 vs 256).
All other parameters are EXACTLY identical:
  - same source images
  - same normalize_orientation
  - same grayscale
  - same Gaussian blur
  - same Otsu threshold
  - same morphology/noise removal
  - same extract_roi() (baseline, NOT text-density)
  - same HOG: orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm=L2-Hys
  - same KNN: K=5, Euclidean, weights=distance
  - same 360 images, 18 students x 20 samples
  - ROI is extracted ONCE and reused for both branches

STRICTLY NO changes to:
  config.py, production extract_roi(), HOG parameters, K, thresholds,
  frontend, API, production model, database.

Output: backend/tests/evaluation_results/resolution_experiment/
"""

import os
import sys
import json
import csv
import sqlite3
import time
import numpy as np
from collections import defaultdict
from skimage.feature import hog

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import (
    DB_PATH, DATASET_RAW_DIR,
    IMAGE_SIZE,  # 128x128 — production value, read-only
    HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK, HOG_BLOCK_NORM,
    KNN_N_NEIGHBORS, GAUSSIAN_BLUR_KERNEL, MEDIAN_BLUR_KERNEL, MORPH_KERNEL_SIZE,
)
from preprocessing.image_processor import (
    normalize_orientation, convert_to_grayscale,
    apply_gaussian_blur, apply_otsu_threshold, remove_noise,
    extract_roi, resize_with_aspect_ratio,
)
from model.classifier import euclidean_to_similarity

import cv2

# ---------------------------------------------------------------------------
# Experiment sizes
# ---------------------------------------------------------------------------
SIZE_128 = (128, 128)
SIZE_256 = (256, 256)
K        = KNN_N_NEIGHBORS   # 5
EPS      = 1e-7

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(BASE_DIR, "tests", "evaluation_results", "resolution_experiment")
os.makedirs(OUT_DIR, exist_ok=True)


# ===========================================================================
# HOG extraction  (identical params, variable image size)
# ===========================================================================
def extract_hog(image: np.ndarray):
    """Extract HOG feature vector. Returns (features, hog_vis)."""
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


def theoretical_hog_dim(size):
    h, w = size
    py, px = HOG_PIXELS_PER_CELL
    ch, cw = HOG_CELLS_PER_BLOCK
    n_blk_r = (h // py) - ch + 1
    n_blk_c = (w // px) - cw + 1
    return n_blk_r * n_blk_c * ch * cw * HOG_ORIENTATIONS


# ===========================================================================
# Distance matrix
# ===========================================================================
def pairwise_dist(X):
    dot = np.dot(X, X.T)
    sq  = np.diag(dot)
    dsq = np.maximum(0.0, sq[:, None] + sq[None, :] - 2 * dot)
    d   = np.sqrt(dsq)
    np.fill_diagonal(d, 0.0)
    return d


# ===========================================================================
# LOOCV KNN (distance-weighted, K=5)
# ===========================================================================
def run_loocv(X, y, dist_matrix=None):
    """Returns list of per-sample result dicts."""
    n   = len(X)
    y_a = np.array(y)
    if dist_matrix is None:
        dist_matrix = pairwise_dist(X)

    unique_classes = sorted(set(y))
    results = []

    for i in range(n):
        true_label = y_a[i]
        row        = np.delete(dist_matrix[i], i)
        lbls       = np.delete(y_a, i)

        sorted_idx  = np.argsort(row)
        topk_idx    = sorted_idx[:K]
        topk_dists  = row[topk_idx]
        topk_labels = lbls[topk_idx]

        wk      = 1.0 / np.maximum(topk_dists, EPS)
        total_w = wk.sum()

        vote_w  = defaultdict(float)
        vote_c  = defaultdict(int)
        for d_val, lbl, wv in zip(topk_dists, topk_labels, wk):
            vote_w[lbl] += wv
            vote_c[lbl] += 1

        sorted_votes = sorted(vote_w.items(), key=lambda x: -x[1])
        pred_label   = sorted_votes[0][0]
        pred_share   = sorted_votes[0][1] / total_w * 100.0
        second_share = sorted_votes[1][1] / total_w * 100.0 if len(sorted_votes) > 1 else 0.0

        true_neighbors  = vote_c.get(true_label, 0)
        true_vote_share = vote_w.get(true_label, 0.0) / total_w * 100.0

        # Per-class min distance
        class_min = {}
        for cls in unique_classes:
            mask = (lbls == cls)
            class_min[cls] = float(row[mask].min()) if mask.any() else float("inf")

        neighbors_detail = [
            {
                "rank":          rank + 1,
                "name":          lbl,
                "distance":      round(float(d), 4),
                "vote_contrib":  round(float(wv / total_w * 100.0), 2),
            }
            for rank, (d, lbl, wv) in enumerate(zip(topk_dists, topk_labels, wk))
        ]

        results.append({
            "sample_idx":        i,
            "true_label":        true_label,
            "pred_label":        pred_label,
            "is_correct":        bool(pred_label == true_label),
            "pred_vote_share":   round(float(pred_share),   2),
            "second_vote_share": round(float(second_share), 2),
            "margin":            round(float(pred_share - second_share), 2),
            "true_neighbors":    true_neighbors,
            "true_vote_share":   round(float(true_vote_share), 2),
            "min_dist_pred":     round(class_min.get(pred_label, 999.0), 4),
            "min_dist_true":     round(class_min.get(true_label, 999.0), 4),
            "nearest_dist":      round(float(topk_dists[0]), 4),
            "neighbors_detail":  neighbors_detail,
        })

    return results, dist_matrix


# ===========================================================================
# Distance-space metrics (intra / inter per class)
# ===========================================================================
def compute_dist_metrics(X, y, dist_matrix=None):
    y_a = np.array(y)
    if dist_matrix is None:
        dist_matrix = pairwise_dist(X)
    unique_classes = sorted(set(y))
    cls_idx = {c: np.where(y_a == c)[0] for c in unique_classes}

    intra_all, inter_all = [], []
    per_class = {}

    for c in unique_classes:
        ci  = cls_idx[c]
        oi  = np.where(y_a != c)[0]
        sub = dist_matrix[np.ix_(ci, ci)]
        iu  = np.triu_indices(len(ci), k=1)
        intra = sub[iu]
        inter = dist_matrix[np.ix_(ci, oi)].flatten()
        intra_all.extend(intra)
        inter_all.extend(inter)
        per_class[c] = {
            "intra_mean":   round(float(np.mean(intra)),   4),
            "intra_median": round(float(np.median(intra)), 4),
            "intra_std":    round(float(np.std(intra)),    4),
            "intra_min":    round(float(np.min(intra)),    4),
            "intra_max":    round(float(np.max(intra)),    4),
            "inter_mean":   round(float(np.mean(inter)),   4),
            "inter_min":    round(float(np.min(inter)),    4),
            "sep_ratio":    round(float(np.mean(inter) / np.mean(intra)) if np.mean(intra) > 0 else 0, 4),
        }

    return {
        "overall_intra_mean":   round(float(np.mean(intra_all)),  4),
        "overall_intra_std":    round(float(np.std(intra_all)),   4),
        "overall_intra_min":    round(float(np.min(intra_all)),   4),
        "overall_intra_max":    round(float(np.max(intra_all)),   4),
        "overall_inter_mean":   round(float(np.mean(inter_all)),  4),
        "overall_inter_std":    round(float(np.std(inter_all)),   4),
        "overall_inter_min":    round(float(np.min(inter_all)),   4),
        "overall_inter_max":    round(float(np.max(inter_all)),   4),
        "separability_ratio":   round(float(np.mean(inter_all) / np.mean(intra_all)) if np.mean(intra_all) > 0 else 0, 4),
        "per_class":            per_class,
    }, dist_matrix


# ===========================================================================
# Full 18x18 confusion matrix
# ===========================================================================
def build_confusion_matrix(loocv_results, unique_classes):
    cls2i = {c: i for i, c in enumerate(unique_classes)}
    cm    = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
    for r in loocv_results:
        cm[cls2i[r["true_label"]], cls2i[r["pred_label"]]] += 1
    return cm


# ===========================================================================
# Confusion-pair analysis (full NxN)
# ===========================================================================
def confusion_pair_analysis(cm_128, cm_256, unique_classes, dist128, dist256, y):
    """
    For every ordered pair (A, B) where A != B, compute:
      A→B @128, B→A @128, total @128
      A→B @256, B→A @256, total @256
      delta total confusion
      mean cross-class distance @128
      mean cross-class distance @256
    """
    n = len(unique_classes)
    cls2i = {c: i for i, c in enumerate(unique_classes)}
    y_a   = np.array(y)

    pairs = []
    for i, a in enumerate(unique_classes):
        for j, b in enumerate(unique_classes):
            if j <= i:
                continue
            ia = np.where(y_a == a)[0]
            ib = np.where(y_a == b)[0]

            ab128 = int(cm_128[cls2i[a], cls2i[b]])
            ba128 = int(cm_128[cls2i[b], cls2i[a]])
            ab256 = int(cm_256[cls2i[a], cls2i[b]])
            ba256 = int(cm_256[cls2i[b], cls2i[a]])

            total128 = ab128 + ba128
            total256 = ab256 + ba256

            cross_d128 = float(dist128[np.ix_(ia, ib)].mean())
            cross_d256 = float(dist256[np.ix_(ia, ib)].mean())

            intra_a128 = float(np.mean(dist128[np.ix_(ia, ia)][np.triu_indices(len(ia), k=1)]))
            intra_b128 = float(np.mean(dist128[np.ix_(ib, ib)][np.triu_indices(len(ib), k=1)]))
            intra_a256 = float(np.mean(dist256[np.ix_(ia, ia)][np.triu_indices(len(ia), k=1)]))
            intra_b256 = float(np.mean(dist256[np.ix_(ib, ib)][np.triu_indices(len(ib), k=1)]))

            pairs.append({
                "class_a":            a,
                "class_b":            b,
                "ab_128":             ab128,
                "ba_128":             ba128,
                "total_128":          total128,
                "ab_256":             ab256,
                "ba_256":             ba256,
                "total_256":          total256,
                "delta_total":        total256 - total128,
                "cross_mean_128":     round(cross_d128, 4),
                "cross_mean_256":     round(cross_d256, 4),
                "cross_vs_intra_a128": round(cross_d128 / intra_a128, 4) if intra_a128 > 0 else 0,
                "cross_vs_intra_b128": round(cross_d128 / intra_b128, 4) if intra_b128 > 0 else 0,
                "cross_vs_intra_a256": round(cross_d256 / intra_a256, 4) if intra_a256 > 0 else 0,
                "cross_vs_intra_b256": round(cross_d256 / intra_b256, 4) if intra_b256 > 0 else 0,
            })

    return pairs


# ===========================================================================
# Vote distribution summary
# ===========================================================================
def vote_dist_summary(loocv_results):
    n    = len(loocv_results)
    cnts = {k: 0 for k in range(6)}
    for r in loocv_results:
        cnts[r["true_neighbors"]] = cnts.get(r["true_neighbors"], 0) + 1
    return {
        "counts":               cnts,
        "pct_1_of_5":           round(cnts.get(1, 0) / n * 100, 2),
        "pct_2_of_5":           round(cnts.get(2, 0) / n * 100, 2),
        "pct_ge3_of_5":         round(sum(cnts.get(k, 0) for k in [3, 4, 5]) / n * 100, 2),
        "mean_winning_vote":    round(float(np.mean([r["pred_vote_share"] for r in loocv_results])), 2),
        "mean_margin":          round(float(np.mean([r["margin"] for r in loocv_results])), 2),
    }


# ===========================================================================
# Per-class accuracy + confusion summary
# ===========================================================================
def per_class_summary(loocv_results, unique_classes, sample_meta):
    grouped = defaultdict(list)
    for r in loocv_results:
        grouped[r["true_label"]].append(r)

    rows = []
    for cls in unique_classes:
        cls_results = grouped[cls]
        n_cls       = len(cls_results)
        n_correct   = sum(1 for r in cls_results if r["is_correct"])
        acc         = round(n_correct / n_cls * 100, 2) if n_cls > 0 else 0.0

        # Main confusion target
        wrong = [r for r in cls_results if not r["is_correct"]]
        if wrong:
            confusion_counter = defaultdict(int)
            for w in wrong:
                confusion_counter[w["pred_label"]] += 1
            main_conf = max(confusion_counter, key=confusion_counter.get)
            main_conf_cnt = confusion_counter[main_conf]
        else:
            main_conf     = ""
            main_conf_cnt = 0

        mean_vote   = round(float(np.mean([r["pred_vote_share"]  for r in cls_results])), 2)
        mean_margin = round(float(np.mean([r["margin"]           for r in cls_results])), 2)
        mean_nbr    = round(float(np.mean([r["true_neighbors"]   for r in cls_results])), 2)

        rows.append({
            "student":          cls,
            "n_samples":        n_cls,
            "n_correct":        n_correct,
            "n_errors":         n_cls - n_correct,
            "accuracy_pct":     acc,
            "main_confusion_to": main_conf,
            "main_confusion_cnt": main_conf_cnt,
            "mean_win_vote":    mean_vote,
            "mean_margin":      mean_margin,
            "mean_true_nbr":    mean_nbr,
        })
    return rows


# ===========================================================================
# Sample-level transition 128→256
# ===========================================================================
def sample_transitions(loocv128, loocv256, sample_meta):
    """Categorise each sample: CC, WC, CW, WW_same, WW_diff."""
    cats = {"CC": [], "WC": [], "CW": [], "WW_same": [], "WW_diff": []}
    for r128, r256 in zip(loocv128, loocv256):
        meta = sample_meta[r128["sample_idx"]]
        entry = {
            "sample_idx":    r128["sample_idx"],
            "student":       r128["true_label"],
            "sample_num":    meta["sample_num"],
            "filename":      meta["filename"],
            "pred_128":      r128["pred_label"],
            "pred_256":      r256["pred_label"],
            "vote_128":      r128["pred_vote_share"],
            "vote_256":      r256["pred_vote_share"],
            "margin_128":    r128["margin"],
            "margin_256":    r256["margin"],
        }
        if r128["is_correct"] and r256["is_correct"]:
            cats["CC"].append(entry)
        elif not r128["is_correct"] and r256["is_correct"]:
            cats["WC"].append(entry)
        elif r128["is_correct"] and not r256["is_correct"]:
            cats["CW"].append(entry)
        elif r128["pred_label"] == r256["pred_label"]:
            cats["WW_same"].append(entry)
        else:
            cats["WW_diff"].append(entry)
    return cats


# ===========================================================================
# Fahim–Fathur (and auto-detect top pairs) analysis
# ===========================================================================
def fahim_fathur_analysis(dist_matrix, y, loocv_results, label):
    y_a = np.array(y)
    fn  = "Fahim J Mujaddid"
    ft  = "Fathurrahman Nugroho"
    fi  = np.where(y_a == fn)[0]
    fr  = np.where(y_a == ft)[0]

    if len(fi) == 0 or len(fr) == 0:
        return {"note": "One or both classes not in dataset"}

    fahim_sub  = dist_matrix[np.ix_(fi, fi)]; iu = np.triu_indices(len(fi), k=1)
    fathur_sub = dist_matrix[np.ix_(fr, fr)]; iu2 = np.triu_indices(len(fr), k=1)
    cross      = dist_matrix[np.ix_(fi, fr)].flatten()

    fahim_intra  = fahim_sub[iu]
    fathur_intra = fathur_sub[iu2]

    f_to_r = sum(1 for r in loocv_results if r["true_label"] == fn and r["pred_label"] == ft)
    r_to_f = sum(1 for r in loocv_results if r["true_label"] == ft and r["pred_label"] == fn)

    return {
        "label":                             label,
        "fahim_intra_mean":                  round(float(np.mean(fahim_intra)),  4),
        "fahim_intra_std":                   round(float(np.std(fahim_intra)),   4),
        "fathur_intra_mean":                 round(float(np.mean(fathur_intra)), 4),
        "fathur_intra_std":                  round(float(np.std(fathur_intra)),  4),
        "cross_mean":                        round(float(np.mean(cross)), 4),
        "cross_min":                         round(float(np.min(cross)),  4),
        "cross_vs_fahim_intra":              round(float(np.mean(cross)) / float(np.mean(fahim_intra)),  4),
        "cross_vs_fathur_intra":             round(float(np.mean(cross)) / float(np.mean(fathur_intra)), 4),
        "cross_below_fahim_intra_mean_pct":  round(float(np.mean(cross < np.mean(fahim_intra)))  * 100, 2),
        "cross_below_fathur_intra_mean_pct": round(float(np.mean(cross < np.mean(fathur_intra))) * 100, 2),
        "fahim_to_fathur_errors":            f_to_r,
        "fathur_to_fahim_errors":            r_to_f,
        "fahim_total_misclassified":         sum(1 for r in loocv_results if r["true_label"] == fn and not r["is_correct"]),
        "fathur_total_misclassified":        sum(1 for r in loocv_results if r["true_label"] == ft and not r["is_correct"]),
    }


# ===========================================================================
# Accuracy helpers
# ===========================================================================
def accuracy_metrics(loocv_results, sample_meta, n):
    correct = sum(1 for r in loocv_results if r["is_correct"])
    p1  = [r for r in loocv_results if sample_meta[r["sample_idx"]]["sample_num"] == 1]
    p2p = [r for r in loocv_results if sample_meta[r["sample_idx"]]["sample_num"] >  1]
    return {
        "overall_accuracy":    round(correct / n * 100, 2),
        "page1_accuracy":      round(sum(1 for r in p1  if r["is_correct"]) / max(1, len(p1))  * 100, 2),
        "page2_20_accuracy":   round(sum(1 for r in p2p if r["is_correct"]) / max(1, len(p2p)) * 100, 2),
        "total_correct":       correct,
        "total_misclassified": n - correct,
    }


# ===========================================================================
# MAIN
# ===========================================================================
def run_experiment_c():
    print("=" * 76)
    print("EXPERIMENT C: 128x128 vs 256x256 RESOLUTION A/B COMPARISON")
    print("=" * 76)

    # ------------------------------------------------------------------
    # 0. Feature vector dimensions (theoretical + actual will be printed)
    # ------------------------------------------------------------------
    dim128_theory = theoretical_hog_dim(SIZE_128)
    dim256_theory = theoretical_hog_dim(SIZE_256)
    print(f"Theoretical HOG dim @ 128x128 : {dim128_theory}")
    print(f"Theoretical HOG dim @ 256x256 : {dim256_theory}")

    # ------------------------------------------------------------------
    # 1. Load dataset from DB
    # ------------------------------------------------------------------
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
    print(f"\nLoaded {n_records} records across {n_classes} students.\n")

    # ------------------------------------------------------------------
    # 2. Process all images — ROI extracted ONCE, reused for both branches
    # ------------------------------------------------------------------
    feat128_list = []
    feat256_list = []
    labels       = []
    sample_meta  = []

    t_preprocess_128 = []   # preprocessing + ROI + resize 128
    t_preprocess_256 = []   # resize 256 from same ROI
    t_hog128_list    = []
    t_hog256_list    = []
    skipped          = 0

    print(f"Processing all {n_records} images (shared ROI, dual resize)...")
    for idx, r in enumerate(rows):
        db_id    = r["id"]
        student  = r["student_name"]
        filename = r["original_filename"] or ""
        raw_name = r["saved_filename"] or os.path.basename(r["file_path"])
        raw_path = os.path.join(DATASET_RAW_DIR, raw_name)

        if not os.path.exists(raw_path):
            print(f"  [SKIP] Not found: {raw_path}")
            skipped += 1
            continue
        img_bgr = cv2.imread(raw_path)
        if img_bgr is None:
            print(f"  [SKIP] Cannot read: {raw_path}")
            skipped += 1
            continue

        # ---- Shared preprocessing (identical for both branches) ----
        t0 = time.perf_counter()
        oriented = normalize_orientation(img_bgr)
        gray     = convert_to_grayscale(oriented)
        blurred  = apply_gaussian_blur(gray)
        binary   = apply_otsu_threshold(blurred)
        denoised = remove_noise(binary)
        roi      = extract_roi(denoised)      # BASELINE extract_roi — NOT text-density
        img128   = resize_with_aspect_ratio(roi, SIZE_128)
        t_preprocess_128.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        img256 = resize_with_aspect_ratio(roi, SIZE_256)   # same ROI, different resize
        t_preprocess_256.append(time.perf_counter() - t1)

        # ---- HOG @ 128 ----
        t2 = time.perf_counter()
        feat128, _ = extract_hog(img128)
        t_hog128_list.append(time.perf_counter() - t2)

        # ---- HOG @ 256 ----
        t3 = time.perf_counter()
        feat256, _ = extract_hog(img256)
        t_hog256_list.append(time.perf_counter() - t3)

        feat128_list.append(feat128)
        feat256_list.append(feat256)
        labels.append(student)
        sample_meta.append({
            "sample_idx": idx - skipped,
            "db_id":      db_id,
            "student_name": student,
            "filename":   filename,
            "sample_num": (len([s for s in sample_meta if s["student_name"] == student]) + 1),
        })

        if (idx + 1) % 72 == 0:
            print(f"  Processed {idx + 1}/{n_records} ...")

    n = len(labels)
    print(f"\nAll {n} images processed (skipped: {skipped})")

    X128 = np.array(feat128_list)
    X256 = np.array(feat256_list)
    y_arr = np.array(labels)

    # Actual dims
    actual_dim128 = X128.shape[1]
    actual_dim256 = X256.shape[1]
    print(f"Actual HOG dim @ 128x128 : {actual_dim128}")
    print(f"Actual HOG dim @ 256x256 : {actual_dim256}")

    mem128_mb = X128.nbytes / 1024 / 1024
    mem256_mb = X256.nbytes / 1024 / 1024
    print(f"Feature matrix memory 128: {mem128_mb:.2f} MB")
    print(f"Feature matrix memory 256: {mem256_mb:.2f} MB")

    # ------------------------------------------------------------------
    # 3. Distance metrics
    # ------------------------------------------------------------------
    print("\nComputing distance metrics (128)...")
    t_dm128_start = time.perf_counter()
    dist_metrics128, dm128 = compute_dist_metrics(X128, labels)
    t_dm128 = time.perf_counter() - t_dm128_start

    print("Computing distance metrics (256)...")
    t_dm256_start = time.perf_counter()
    dist_metrics256, dm256 = compute_dist_metrics(X256, labels)
    t_dm256 = time.perf_counter() - t_dm256_start

    # ------------------------------------------------------------------
    # 4. LOOCV KNN
    # ------------------------------------------------------------------
    print("\nRunning LOOCV KNN (128)...")
    t_loocv128_start = time.perf_counter()
    loocv128, _ = run_loocv(X128, labels, dm128)
    t_loocv128   = time.perf_counter() - t_loocv128_start

    print("Running LOOCV KNN (256)...")
    t_loocv256_start = time.perf_counter()
    loocv256, _ = run_loocv(X256, labels, dm256)
    t_loocv256   = time.perf_counter() - t_loocv256_start

    # ------------------------------------------------------------------
    # 5. Accuracy
    # ------------------------------------------------------------------
    acc128 = accuracy_metrics(loocv128, sample_meta, n)
    acc256 = accuracy_metrics(loocv256, sample_meta, n)

    # ------------------------------------------------------------------
    # 6. Vote distributions
    # ------------------------------------------------------------------
    vote128 = vote_dist_summary(loocv128)
    vote256 = vote_dist_summary(loocv256)

    # ------------------------------------------------------------------
    # 7. Confusion matrices
    # ------------------------------------------------------------------
    unique_classes = sorted(set(labels))
    cm128 = build_confusion_matrix(loocv128, unique_classes)
    cm256 = build_confusion_matrix(loocv256, unique_classes)

    # ------------------------------------------------------------------
    # 8. Per-class summaries
    # ------------------------------------------------------------------
    per_cls128 = per_class_summary(loocv128, unique_classes, sample_meta)
    per_cls256 = per_class_summary(loocv256, unique_classes, sample_meta)

    # Build lookup by student name
    cls128_by_name = {r["student"]: r for r in per_cls128}
    cls256_by_name = {r["student"]: r for r in per_cls256}

    per_student_combined = []
    for cls in unique_classes:
        r128 = cls128_by_name[cls]
        r256 = cls256_by_name[cls]
        per_student_combined.append({
            "student":            cls,
            "accuracy_128":       r128["accuracy_pct"],
            "accuracy_256":       r256["accuracy_pct"],
            "delta_accuracy":     round(r256["accuracy_pct"] - r128["accuracy_pct"], 2),
            "errors_128":         r128["n_errors"],
            "errors_256":         r256["n_errors"],
            "main_confusion_128": r128["main_confusion_to"],
            "main_conf_cnt_128":  r128["main_confusion_cnt"],
            "main_confusion_256": r256["main_confusion_to"],
            "main_conf_cnt_256":  r256["main_confusion_cnt"],
            "mean_vote_128":      r128["mean_win_vote"],
            "mean_vote_256":      r256["mean_win_vote"],
            "mean_margin_128":    r128["mean_margin"],
            "mean_margin_256":    r256["mean_margin"],
        })
    per_student_combined.sort(key=lambda x: x["delta_accuracy"], reverse=True)

    # ------------------------------------------------------------------
    # 9. Confusion pair analysis
    # ------------------------------------------------------------------
    print("\nBuilding confusion pair analysis...")
    conf_pairs = confusion_pair_analysis(cm128, cm256, unique_classes, dm128, dm256, labels)

    # Sort by total confusion @128 for top pairs
    top_confusing_128 = sorted(conf_pairs, key=lambda x: -x["total_128"])[:10]
    top_confusing_256 = sorted(conf_pairs, key=lambda x: -x["total_256"])[:10]
    most_improved     = sorted(conf_pairs, key=lambda x:  x["delta_total"])[:10]
    most_degraded     = sorted(conf_pairs, key=lambda x: -x["delta_total"])[:10]

    # ------------------------------------------------------------------
    # 10. Page-1 comparison table
    # ------------------------------------------------------------------
    print("Building page-1 comparison table...")
    page1_rows = []
    for r128, r256 in zip(loocv128, loocv256):
        meta = sample_meta[r128["sample_idx"]]
        if meta["sample_num"] != 1:
            continue
        page1_rows.append({
            "student":         meta["student_name"],
            "filename":        meta["filename"],
            "pred_128":        r128["pred_label"],
            "pred_256":        r256["pred_label"],
            "correct_128":     r128["is_correct"],
            "correct_256":     r256["is_correct"],
            "vote_128":        r128["pred_vote_share"],
            "vote_256":        r256["pred_vote_share"],
            "neighbors_128":   r128["true_neighbors"],
            "neighbors_256":   r256["true_neighbors"],
            "margin_128":      r128["margin"],
            "margin_256":      r256["margin"],
            "fail128_fix256":  (not r128["is_correct"] and r256["is_correct"]),
            "ok128_fail256":   (r128["is_correct"] and not r256["is_correct"]),
        })

    p1_fails128  = sum(1 for r in page1_rows if not r["correct_128"])
    p1_fixed256  = sum(1 for r in page1_rows if r["fail128_fix256"])
    p1_regressed = sum(1 for r in page1_rows if r["ok128_fail256"])
    print(f"  Page-1: baseline failures={p1_fails128}/18, 256 fixes={p1_fixed256}, 256 regressions={p1_regressed}")

    # ------------------------------------------------------------------
    # 11. Sample-level transitions
    # ------------------------------------------------------------------
    transitions = sample_transitions(loocv128, loocv256, sample_meta)
    print(f"\nSample transitions: CC={len(transitions['CC'])}, WC={len(transitions['WC'])}, "
          f"CW={len(transitions['CW'])}, WW_same={len(transitions['WW_same'])}, WW_diff={len(transitions['WW_diff'])}")

    # ------------------------------------------------------------------
    # 12. Fahim–Fathur case study
    # ------------------------------------------------------------------
    ff128 = fahim_fathur_analysis(dm128, labels, loocv128, "128x128")
    ff256 = fahim_fathur_analysis(dm256, labels, loocv256, "256x256")

    # ------------------------------------------------------------------
    # 13. Computational cost summary
    # ------------------------------------------------------------------
    def ms(t_list): return round(float(np.mean(t_list)) * 1000, 3)

    comp_cost = {
        "n_samples": n,
        "128x128": {
            "mean_preprocess_ms":     ms(t_preprocess_128),
            "mean_hog_extract_ms":    ms(t_hog128_list),
            "total_feature_extract_s": round(sum(t_preprocess_128) + sum(t_hog128_list), 3),
            "loocv_eval_s":           round(t_loocv128, 3),
            "dist_matrix_s":          round(t_dm128, 3),
            "feature_dim":            actual_dim128,
            "feature_matrix_MB":      round(mem128_mb, 3),
        },
        "256x256": {
            "mean_preprocess_ms":     ms(t_preprocess_256),
            "mean_hog_extract_ms":    ms(t_hog256_list),
            "total_feature_extract_s": round(sum(t_preprocess_256) + sum(t_hog256_list), 3),
            "loocv_eval_s":           round(t_loocv256, 3),
            "dist_matrix_s":          round(t_dm256, 3),
            "feature_dim":            actual_dim256,
            "feature_matrix_MB":      round(mem256_mb, 3),
        },
    }

    # ------------------------------------------------------------------
    # 14. Compile misclassification comparison rows
    # ------------------------------------------------------------------
    mis128_set = {r["sample_idx"] for r in loocv128 if not r["is_correct"]}
    mis256_set = {r["sample_idx"] for r in loocv256 if not r["is_correct"]}
    all_mc     = mis128_set | mis256_set

    misclass_rows = []
    for idx in sorted(all_mc):
        meta = sample_meta[idx]
        r128 = loocv128[idx]
        r256 = loocv256[idx]
        misclass_rows.append({
            "sample_idx":    idx,
            "student":       meta["student_name"],
            "sample_num":    meta["sample_num"],
            "filename":      meta["filename"],
            "correct_128":   r128["is_correct"],
            "pred_128":      r128["pred_label"],
            "vote_128":      r128["pred_vote_share"],
            "correct_256":   r256["is_correct"],
            "pred_256":      r256["pred_label"],
            "vote_256":      r256["pred_vote_share"],
        })

    # ------------------------------------------------------------------
    # 15. Compile final JSON report
    # ------------------------------------------------------------------
    report = {
        "experiment": "C — 128x128 vs 256x256 Resolution Comparison",
        "dataset": {"n_samples": n, "n_classes": n_classes, "students": unique_classes},
        "feature_dimensions": {
            "128x128_theoretical": dim128_theory,
            "128x128_actual":      actual_dim128,
            "256x256_theoretical": dim256_theory,
            "256x256_actual":      actual_dim256,
        },
        "baseline_128": {
            "accuracy":        acc128,
            "distance_metrics": dist_metrics128,
            "vote_distribution": vote128,
            "confusion_matrix": cm128.tolist(),
        },
        "experiment_256": {
            "accuracy":        acc256,
            "distance_metrics": dist_metrics256,
            "vote_distribution": vote256,
            "confusion_matrix": cm256.tolist(),
        },
        "comparison_summary": {
            "delta_overall_accuracy":    round(acc256["overall_accuracy"]  - acc128["overall_accuracy"],  2),
            "delta_page1_accuracy":      round(acc256["page1_accuracy"]    - acc128["page1_accuracy"],    2),
            "delta_page2_20_accuracy":   round(acc256["page2_20_accuracy"] - acc128["page2_20_accuracy"], 2),
            "delta_separability_ratio":  round(dist_metrics256["separability_ratio"] - dist_metrics128["separability_ratio"], 4),
            "delta_intra_mean":          round(dist_metrics256["overall_intra_mean"] - dist_metrics128["overall_intra_mean"], 4),
            "delta_inter_mean":          round(dist_metrics256["overall_inter_mean"] - dist_metrics128["overall_inter_mean"], 4),
            "page1_baseline_failures":   p1_fails128,
            "page1_fixed_by_256":        p1_fixed256,
            "page1_regressed_by_256":    p1_regressed,
            "WC_improvements":           len(transitions["WC"]),
            "CW_regressions":            len(transitions["CW"]),
            "CC_both_correct":           len(transitions["CC"]),
            "WW_same":                   len(transitions["WW_same"]),
            "WW_diff":                   len(transitions["WW_diff"]),
        },
        "fahim_fathur": {
            "baseline_128": ff128,
            "experiment_256": ff256,
        },
        "confusion_pairs": {
            "top_confusing_128":  top_confusing_128,
            "top_confusing_256":  top_confusing_256,
            "most_improved_256":  most_improved,
            "most_degraded_256":  most_degraded,
        },
        "computational_cost": comp_cost,
    }

    # ------------------------------------------------------------------
    # 16. Save outputs
    # ------------------------------------------------------------------

    # Main JSON
    json_path = os.path.join(OUT_DIR, "resolution_ab_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, (int, np.integer)) else
                                    float(o) if isinstance(o, (float, np.floating)) else
                                    bool(o) if isinstance(o, (bool, np.bool_)) else str(o))
    print(f"\nSaved: {json_path}")

    # Page-1 CSV
    p1_csv = os.path.join(OUT_DIR, "page1_comparison.csv")
    with open(p1_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(page1_rows[0].keys()) if page1_rows else [])
        w.writeheader(); w.writerows(page1_rows)
    print(f"Saved: {p1_csv}")

    # Per-student accuracy CSV
    ps_csv = os.path.join(OUT_DIR, "per_student_accuracy.csv")
    with open(ps_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_student_combined[0].keys()) if per_student_combined else [])
        w.writeheader(); w.writerows(per_student_combined)
    print(f"Saved: {ps_csv}")

    # Misclassification comparison CSV
    mc_csv = os.path.join(OUT_DIR, "misclassification_comparison.csv")
    with open(mc_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(misclass_rows[0].keys()) if misclass_rows else [])
        w.writeheader(); w.writerows(misclass_rows)
    print(f"Saved: {mc_csv}")

    # Fahim–Fathur JSON
    ff_path = os.path.join(OUT_DIR, "fahim_fathur_analysis.json")
    with open(ff_path, "w", encoding="utf-8") as f:
        json.dump({"baseline_128": ff128, "experiment_256": ff256}, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, (int, np.integer)) else
                                    float(o) if isinstance(o, (float, np.floating)) else str(o))
    print(f"Saved: {ff_path}")

    # Computational cost JSON
    cc_path = os.path.join(OUT_DIR, "computational_cost.json")
    with open(cc_path, "w", encoding="utf-8") as f:
        json.dump(comp_cost, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, (int, np.integer)) else
                                    float(o) if isinstance(o, (float, np.floating)) else str(o))
    print(f"Saved: {cc_path}")

    # Confusion matrix 128 CSV
    cm128_csv = os.path.join(OUT_DIR, "confusion_matrix_128.csv")
    with open(cm128_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["actual \\ pred"] + unique_classes)
        for i, cls in enumerate(unique_classes):
            w.writerow([cls] + cm128[i].tolist())
    print(f"Saved: {cm128_csv}")

    # Confusion matrix 256 CSV
    cm256_csv = os.path.join(OUT_DIR, "confusion_matrix_256.csv")
    with open(cm256_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["actual \\ pred"] + unique_classes)
        for i, cls in enumerate(unique_classes):
            w.writerow([cls] + cm256[i].tolist())
    print(f"Saved: {cm256_csv}")

    # Confusion pairs comparison CSV
    cp_csv = os.path.join(OUT_DIR, "confusion_pairs_comparison.csv")
    cp_fields = list(conf_pairs[0].keys()) if conf_pairs else []
    with open(cp_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cp_fields)
        w.writeheader()
        for p in sorted(conf_pairs, key=lambda x: -(x["total_128"] + x["total_256"])):
            w.writerow(p)
    print(f"Saved: {cp_csv}")

    # Per-student detailed analysis CSV (per-class metrics both resolutions)
    psd_csv = os.path.join(OUT_DIR, "per_student_detailed_analysis.csv")
    psd_fields = ["student","accuracy_128","accuracy_256","delta_accuracy",
                  "errors_128","errors_256",
                  "intra_mean_128","intra_mean_256","sep_ratio_128","sep_ratio_256",
                  "main_confusion_128","main_conf_cnt_128","main_confusion_256","main_conf_cnt_256",
                  "mean_vote_128","mean_vote_256","mean_margin_128","mean_margin_256"]
    with open(psd_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=psd_fields)
        w.writeheader()
        dm128_pc = dist_metrics128["per_class"]
        dm256_pc = dist_metrics256["per_class"]
        for row in per_student_combined:
            cls = row["student"]
            w.writerow({
                "student":            cls,
                "accuracy_128":       row["accuracy_128"],
                "accuracy_256":       row["accuracy_256"],
                "delta_accuracy":     row["delta_accuracy"],
                "errors_128":         row["errors_128"],
                "errors_256":         row["errors_256"],
                "intra_mean_128":     dm128_pc.get(cls, {}).get("intra_mean", ""),
                "intra_mean_256":     dm256_pc.get(cls, {}).get("intra_mean", ""),
                "sep_ratio_128":      dm128_pc.get(cls, {}).get("sep_ratio",  ""),
                "sep_ratio_256":      dm256_pc.get(cls, {}).get("sep_ratio",  ""),
                "main_confusion_128": row["main_confusion_128"],
                "main_conf_cnt_128":  row["main_conf_cnt_128"],
                "main_confusion_256": row["main_confusion_256"],
                "main_conf_cnt_256":  row["main_conf_cnt_256"],
                "mean_vote_128":      row["mean_vote_128"],
                "mean_vote_256":      row["mean_vote_256"],
                "mean_margin_128":    row["mean_margin_128"],
                "mean_margin_256":    row["mean_margin_256"],
            })
    print(f"Saved: {psd_csv}")

    # Sample transition CSV
    st_csv = os.path.join(OUT_DIR, "sample_transition_analysis.csv")
    all_trans_rows = []
    for cat, items in transitions.items():
        for item in items:
            all_trans_rows.append({"category": cat, **item})
    if all_trans_rows:
        with open(st_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_trans_rows[0].keys()))
            w.writeheader(); w.writerows(all_trans_rows)
    print(f"Saved: {st_csv}")

    # Feature-space class analysis JSON
    fs_path = os.path.join(OUT_DIR, "feature_space_class_analysis.json")
    with open(fs_path, "w", encoding="utf-8") as f:
        json.dump({
            "128x128": dist_metrics128["per_class"],
            "256x256": dist_metrics256["per_class"],
        }, f, indent=2,
        default=lambda o: int(o) if isinstance(o, (int, np.integer)) else
                          float(o) if isinstance(o, (float, np.floating)) else str(o))
    print(f"Saved: {fs_path}")

    # ------------------------------------------------------------------
    # 17. Console summary
    # ------------------------------------------------------------------
    C = report["comparison_summary"]
    dm128_ov = dist_metrics128
    dm256_ov = dist_metrics256

    print("\n" + "=" * 76)
    print("EXPERIMENT C FINAL SUMMARY  --  128x128  vs  256x256")
    print("=" * 76)
    fmt = "{:<40} {:>11}  {:>11}  {:>9}"
    print(fmt.format("Metric", "128x128", "256x256", "Delta"))
    print("-" * 76)
    summary_rows = [
        ("Feature dimension",           actual_dim128,                   actual_dim256),
        ("Overall Accuracy (%)",        acc128["overall_accuracy"],       acc256["overall_accuracy"]),
        ("Page-1 Accuracy (%)",         acc128["page1_accuracy"],         acc256["page1_accuracy"]),
        ("Page 2-20 Accuracy (%)",      acc128["page2_20_accuracy"],      acc256["page2_20_accuracy"]),
        ("Intra-class Mean Dist",       dm128_ov["overall_intra_mean"],   dm256_ov["overall_intra_mean"]),
        ("Inter-class Mean Dist",       dm128_ov["overall_inter_mean"],   dm256_ov["overall_inter_mean"]),
        ("Separability Ratio",          dm128_ov["separability_ratio"],   dm256_ov["separability_ratio"]),
        ("Mean Win Vote Share (%)",     vote128["mean_winning_vote"],     vote256["mean_winning_vote"]),
        ("Mean Margin (#1-#2 %)",       vote128["mean_margin"],           vote256["mean_margin"]),
        ("Vote 1/5 (%)",                vote128["pct_1_of_5"],            vote256["pct_1_of_5"]),
        ("Vote 2/5 (%)",                vote128["pct_2_of_5"],            vote256["pct_2_of_5"]),
        ("Vote >=3/5 (%)",              vote128["pct_ge3_of_5"],          vote256["pct_ge3_of_5"]),
        ("Feature matrix MB",           mem128_mb,                        mem256_mb),
    ]
    for label, bv, ev in summary_rows:
        d    = ev - bv
        sign = "+" if d >= 0 else ""
        print(fmt.format(label, f"{bv:.3f}", f"{ev:.3f}", f"{sign}{d:.3f}"))

    print()
    print(f"Baseline wrong -> 256 correct (WC) : {len(transitions['WC'])} samples")
    print(f"Baseline correct -> 256 wrong (CW) : {len(transitions['CW'])} samples")
    print(f"Both correct (CC)                  : {len(transitions['CC'])} samples")
    print(f"Both wrong same pred (WW_same)     : {len(transitions['WW_same'])} samples")
    print(f"Both wrong diff pred (WW_diff)     : {len(transitions['WW_diff'])} samples")
    print()
    print(f"Page-1 failures @128               : {p1_fails128}/18")
    print(f"Page-1 fixed by 256                : {p1_fixed256}")
    print(f"Page-1 regressions by 256          : {p1_regressed}")

    print()
    print("FAHIM-FATHUR CONFUSION:")
    ff_fmt = "{:<40} {:>11}  {:>11}"
    print(ff_fmt.format("Metric", "128x128", "256x256"))
    print("-" * 65)
    for key, label in [
        ("fahim_intra_mean",                  "Fahim intra mean dist"),
        ("fathur_intra_mean",                 "Fathur intra mean dist"),
        ("cross_mean",                        "Cross mean dist"),
        ("cross_vs_fahim_intra",              "Cross / Fahim intra (ratio)"),
        ("cross_vs_fathur_intra",             "Cross / Fathur intra (ratio)"),
        ("cross_below_fahim_intra_mean_pct",  "Cross < Fahim intra mean (%)"),
        ("fahim_to_fathur_errors",            "Fahim->Fathur errors"),
        ("fathur_to_fahim_errors",            "Fathur->Fahim errors"),
        ("fahim_total_misclassified",         "Fahim total misclassified"),
        ("fathur_total_misclassified",        "Fathur total misclassified"),
    ]:
        print(ff_fmt.format(label, str(ff128.get(key, "")), str(ff256.get(key, ""))))

    print()
    print("TOP 5 MOST CONFUSING PAIRS @ 128x128:")
    for p in top_confusing_128[:5]:
        print(f"  {p['class_a'][:20]:<22} <-> {p['class_b'][:20]:<22} "
              f"total={p['total_128']}  (A->B={p['ab_128']}, B->A={p['ba_128']})")

    print()
    print("TOP 5 MOST CONFUSING PAIRS @ 256x256:")
    for p in top_confusing_256[:5]:
        print(f"  {p['class_a'][:20]:<22} <-> {p['class_b'][:20]:<22} "
              f"total={p['total_256']}  (A->B={p['ab_256']}, B->A={p['ba_256']})")

    print()
    print("PER-STUDENT ACCURACY CHANGE (sorted by delta):")
    hdr = "{:<30} {:>8}  {:>8}  {:>7}  {:>7}"
    print(hdr.format("Student", "Acc@128", "Acc@256", "Delta", "Errors@256"))
    print("-" * 68)
    for row in per_student_combined:
        sign = "+" if row["delta_accuracy"] >= 0 else ""
        print(hdr.format(row["student"][:30],
                         f"{row['accuracy_128']:.1f}%",
                         f"{row['accuracy_256']:.1f}%",
                         f"{sign}{row['delta_accuracy']:.1f}",
                         str(row["errors_256"])))

    print()
    print("COMPUTATIONAL COST:")
    cc_fmt = "{:<38} {:>11}  {:>11}"
    print(cc_fmt.format("Metric", "128x128", "256x256"))
    print("-" * 64)
    c128 = comp_cost["128x128"]; c256 = comp_cost["256x256"]
    print(cc_fmt.format("Feature dimension",           c128["feature_dim"],          c256["feature_dim"]))
    print(cc_fmt.format("Feature matrix (MB)",         f"{c128['feature_matrix_MB']:.2f}", f"{c256['feature_matrix_MB']:.2f}"))
    print(cc_fmt.format("Mean preprocess+resize (ms)", f"{c128['mean_preprocess_ms']:.2f}", f"{c256['mean_preprocess_ms']:.2f}"))
    print(cc_fmt.format("Mean HOG extraction (ms)",    f"{c128['mean_hog_extract_ms']:.2f}", f"{c256['mean_hog_extract_ms']:.2f}"))
    print(cc_fmt.format("Total feature extract (s)",   f"{c128['total_feature_extract_s']:.2f}", f"{c256['total_feature_extract_s']:.2f}"))
    print(cc_fmt.format("Distance matrix time (s)",    f"{c128['dist_matrix_s']:.2f}", f"{c256['dist_matrix_s']:.2f}"))
    print(cc_fmt.format("LOOCV eval time (s)",         f"{c128['loocv_eval_s']:.2f}", f"{c256['loocv_eval_s']:.2f}"))

    print()
    print("=" * 76)
    print("EXPERIMENT C COMPLETE — No production code modified")
    print("=" * 76)


if __name__ == "__main__":
    run_experiment_c()
