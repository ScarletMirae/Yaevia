"""
tests/experiment_f_hog_scale.py — Experiment F: HOG Scale Sensitivity Analysis
================================================================================
Evaluates HOG spatial scale sensitivity across 360 images (18 students x 20 samples)
using 4 strictly controlled configurations:

  F0: 128x128, HOG PPC=(8,8),   CPB=(2,2), Ori=9 -> 8,100-dim (Reference Baseline)
  F1: 256x256, HOG PPC=(8,8),   CPB=(2,2), Ori=9 -> 34,596-dim (Current Best Resolution)
  F2: 256x256, HOG PPC=(16,16), CPB=(2,2), Ori=9 -> 8,100-dim (Relative-Scale Matched)
  F3: 256x256, HOG PPC=(12,12), CPB=(2,2), Ori=9 -> 14,400-dim (Mid-Scale Sensitivity)

All preprocessing (orientation norm, gray, blur, Otsu, noise removal, baseline extract_roi()),
classifier configuration (K=5, Euclidean, distance weighting), and LOOCV methodology are
STRICTLY IDENTICAL across all configurations.

Outputs saved in: backend/tests/evaluation_results/hog_scale_experiment/
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
    HOG_ORIENTATIONS, HOG_CELLS_PER_BLOCK, HOG_BLOCK_NORM,
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

OUT_DIR = os.path.join(BASE_DIR, "tests", "evaluation_results", "hog_scale_experiment")
os.makedirs(OUT_DIR, exist_ok=True)


def extract_hog_config(image: np.ndarray, ppc: tuple):
    img_f = image.astype(np.float64) / 255.0 if image.dtype != np.float64 else image
    feat = hog(
        img_f,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=ppc,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        visualize=False,
        feature_vector=True,
    )
    return feat


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


def run_experiment_f():
    print("=" * 80)
    print("EXPERIMENT F: HOG SCALE SENSITIVITY ANALYSIS (F0 vs F1 vs F2 vs F3)")
    print("=" * 80)

    # 1. Load Dataset
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, student_name, original_filename, saved_filename, file_path "
        "FROM dataset ORDER BY student_name ASC, original_filename ASC, saved_filename ASC"
    ).fetchall()
    conn.close()

    students = sorted(list(set(r["student_name"] for r in rows)))
    n_records = len(rows)
    print(f"Loaded {n_records} records across {len(students)} students.")

    # 2. Extract Features for F0, F1, F2, F3
    feat_f0, feat_f1, feat_f2, feat_f3 = [], [], [], []
    t_f0, t_f1, t_f2, t_f3 = [], [], [], []
    labels = []
    samples_data = []

    print("\nProcessing images and extracting HOG features across 4 configurations...")
    for idx, r in enumerate(rows):
        student = r["student_name"]
        raw_name = r["saved_filename"] or os.path.basename(r["file_path"])
        raw_path = os.path.join(DATASET_RAW_DIR, raw_name)

        if not os.path.exists(raw_path):
            continue
        img_bgr = cv2.imread(raw_path)
        if img_bgr is None:
            continue

        oriented = normalize_orientation(img_bgr)
        gray = convert_to_grayscale(oriented)
        blurred = apply_gaussian_blur(gray)
        binary = apply_otsu_threshold(blurred)
        denoised = remove_noise(binary)
        base_roi = extract_roi(denoised)

        img128 = resize_with_aspect_ratio(base_roi, SIZE_128)
        img256 = resize_with_aspect_ratio(base_roi, SIZE_256)

        # F0: 128x128, PPC=(8,8)
        t0 = time.perf_counter()
        f0 = extract_hog_config(img128, (8, 8))
        t_f0.append(time.perf_counter() - t0)

        # F1: 256x256, PPC=(8,8)
        t1 = time.perf_counter()
        f1 = extract_hog_config(img256, (8, 8))
        t_f1.append(time.perf_counter() - t1)

        # F2: 256x256, PPC=(16,16)
        t2 = time.perf_counter()
        f2 = extract_hog_config(img256, (16, 16))
        t_f2.append(time.perf_counter() - t2)

        # F3: 256x256, PPC=(12,12)
        t3 = time.perf_counter()
        f3 = extract_hog_config(img256, (12, 12))
        t_f3.append(time.perf_counter() - t3)

        feat_f0.append(f0)
        feat_f1.append(f1)
        feat_f2.append(f2)
        feat_f3.append(f3)
        labels.append(student)

        sample_num = len([s for s in samples_data if s["student_name"] == student]) + 1
        samples_data.append({
            "sample_idx": len(samples_data),
            "id": r["id"],
            "student_name": student,
            "filename": r["original_filename"],
            "sample_num": sample_num,
        })

    n = len(labels)
    X_f0 = np.array(feat_f0)
    X_f1 = np.array(feat_f1)
    X_f2 = np.array(feat_f2)
    X_f3 = np.array(feat_f3)

    # 3. Distance Metrics & LOOCV
    print("\nRunning LOOCV evaluation for F0, F1, F2, F3...")
    t_eval_f0_start = time.perf_counter()
    loocv_f0, dist_f0 = run_loocv(X_f0, labels)
    t_eval_f0 = time.perf_counter() - t_eval_f0_start

    t_eval_f1_start = time.perf_counter()
    loocv_f1, dist_f1 = run_loocv(X_f1, labels)
    t_eval_f1 = time.perf_counter() - t_eval_f1_start

    t_eval_f2_start = time.perf_counter()
    loocv_f2, dist_f2 = run_loocv(X_f2, labels)
    t_eval_f2 = time.perf_counter() - t_eval_f2_start

    t_eval_f3_start = time.perf_counter()
    loocv_f3, dist_f3 = run_loocv(X_f3, labels)
    t_eval_f3 = time.perf_counter() - t_eval_f3_start

    dm_f0, _ = compute_dist_metrics(X_f0, labels, dist_f0)
    dm_f1, _ = compute_dist_metrics(X_f1, labels, dist_f1)
    dm_f2, _ = compute_dist_metrics(X_f2, labels, dist_f2)
    dm_f3, _ = compute_dist_metrics(X_f3, labels, dist_f3)

    for i in range(n):
        samples_data[i]["loocv_f0"] = loocv_f0[i]
        samples_data[i]["loocv_f1"] = loocv_f1[i]
        samples_data[i]["loocv_f2"] = loocv_f2[i]
        samples_data[i]["loocv_f3"] = loocv_f3[i]

    # Accuracy calculations
    def calc_acc(loocv_list):
        return sum(1 for r in loocv_list if r["is_correct"]) / n * 100

    acc_f0 = calc_acc(loocv_f0)
    acc_f1 = calc_acc(loocv_f1)
    acc_f2 = calc_acc(loocv_f2)
    acc_f3 = calc_acc(loocv_f3)

    p1_samps = [s for s in samples_data if s["sample_num"] == 1]
    p16_samps = [s for s in samples_data if s["sample_num"] == 16]
    prest_samps = [s for s in samples_data if s["sample_num"] > 1]

    def sub_acc(loocv_key, s_list):
        return sum(1 for s in s_list if s[loocv_key]["is_correct"]) / len(s_list) * 100

    p1_f0 = sub_acc("loocv_f0", p1_samps)
    p1_f1 = sub_acc("loocv_f1", p1_samps)
    p1_f2 = sub_acc("loocv_f2", p1_samps)
    p1_f3 = sub_acc("loocv_f3", p1_samps)

    p16_f0 = sub_acc("loocv_f0", p16_samps)
    p16_f1 = sub_acc("loocv_f1", p16_samps)
    p16_f2 = sub_acc("loocv_f2", p16_samps)
    p16_f3 = sub_acc("loocv_f3", p16_samps)

    prest_f0 = sub_acc("loocv_f0", prest_samps)
    prest_f1 = sub_acc("loocv_f1", prest_samps)
    prest_f2 = sub_acc("loocv_f2", prest_samps)
    prest_f3 = sub_acc("loocv_f3", prest_samps)

    def vote_stats(loocv_list):
        nbr_ge3 = sum(1 for r in loocv_list if r["true_neighbors"] >= 3) / n * 100
        mean_vote = float(np.mean([r["pred_vote_share"] for r in loocv_list]))
        mean_margin = float(np.mean([r["margin"] for r in loocv_list]))
        return round(nbr_ge3, 2), round(mean_vote, 2), round(mean_margin, 2)

    nbr_f0, vote_f0, margin_f0 = vote_stats(loocv_f0)
    nbr_f1, vote_f1, margin_f1 = vote_stats(loocv_f1)
    nbr_f2, vote_f2, margin_f2 = vote_stats(loocv_f2)
    nbr_f3, vote_f3, margin_f3 = vote_stats(loocv_f3)

    # 4. Transitions (F1 -> F2 primary, also F1 -> F3)
    def calc_transitions(key_from, key_to):
        trans = {"CC": 0, "WC": 0, "CW": 0, "WW_same": 0, "WW_diff": 0}
        rows_out = []
        for s in samples_data:
            rf = s[key_from]
            rt = s[key_to]
            if rf["is_correct"] and rt["is_correct"]:
                c = "CC"
            elif not rf["is_correct"] and rt["is_correct"]:
                c = "WC"
            elif rf["is_correct"] and not rt["is_correct"]:
                c = "CW"
            elif rf["pred_label"] == rt["pred_label"]:
                c = "WW_same"
            else:
                c = "WW_diff"
            trans[c] += 1
            rows_out.append({
                "sample_idx": s["sample_idx"],
                "student": s["student_name"],
                "sample_num": s["sample_num"],
                "filename": s["filename"],
                "pred_from": rf["pred_label"],
                "correct_from": rf["is_correct"],
                "pred_to": rt["pred_label"],
                "correct_to": rt["is_correct"],
                "transition": c,
            })
        return trans, rows_out

    trans_f1_f2, trans_f1_f2_rows = calc_transitions("loocv_f1", "loocv_f2")
    trans_f1_f3, _ = calc_transitions("loocv_f1", "loocv_f3")
    trans_f2_f3, _ = calc_transitions("loocv_f2", "loocv_f3")

    net_f1_f2 = trans_f1_f2["WC"] - trans_f1_f2["CW"]

    # 5. Position Comparison Table (Positions 1..20)
    pos_comparison_rows = []
    for p in range(1, 21):
        s_list = [s for s in samples_data if s["sample_num"] == p]
        cnt = len(s_list)
        p_f0 = sum(1 for s in s_list if s["loocv_f0"]["is_correct"]) / cnt * 100
        p_f1 = sum(1 for s in s_list if s["loocv_f1"]["is_correct"]) / cnt * 100
        p_f2 = sum(1 for s in s_list if s["loocv_f2"]["is_correct"]) / cnt * 100
        p_f3 = sum(1 for s in s_list if s["loocv_f3"]["is_correct"]) / cnt * 100
        pos_comparison_rows.append({
            "position": f"#{p:02d}",
            "pos_int": p,
            "acc_f0": round(p_f0, 2),
            "acc_f1": round(p_f1, 2),
            "acc_f2": round(p_f2, 2),
            "acc_f3": round(p_f3, 2),
            "delta_f2_vs_f1": round(p_f2 - p_f1, 2),
            "delta_f3_vs_f1": round(p_f3 - p_f1, 2),
        })

    # 6. Per-Student Comparison Table
    student_comparison_rows = []
    st_improved_f1_f2, st_unchanged_f1_f2, st_regressed_f1_f2 = 0, 0, 0

    for st in students:
        s_list = [s for s in samples_data if s["student_name"] == st]
        cnt = len(s_list)
        st_f0 = sum(1 for s in s_list if s["loocv_f0"]["is_correct"]) / cnt * 100
        st_f1 = sum(1 for s in s_list if s["loocv_f1"]["is_correct"]) / cnt * 100
        st_f2 = sum(1 for s in s_list if s["loocv_f2"]["is_correct"]) / cnt * 100
        st_f3 = sum(1 for s in s_list if s["loocv_f3"]["is_correct"]) / cnt * 100
        d_f2_f1 = round(st_f2 - st_f1, 2)

        if d_f2_f1 > 0:
            st_improved_f1_f2 += 1
        elif d_f2_f1 == 0:
            st_unchanged_f1_f2 += 1
        else:
            st_regressed_f1_f2 += 1

        # Determine diagnostic best configuration
        acc_dict = {"F0": st_f0, "F1": st_f1, "F2": st_f2, "F3": st_f3}
        max_acc = max(acc_dict.values())
        best_cfgs = [k for k, v in acc_dict.items() if v == max_acc]
        best_cfg_str = "/".join(best_cfgs)

        student_comparison_rows.append({
            "student": st,
            "acc_f0": round(st_f0, 2),
            "acc_f1": round(st_f1, 2),
            "acc_f2": round(st_f2, 2),
            "acc_f3": round(st_f3, 2),
            "delta_f2_vs_f1": d_f2_f1,
            "best_cfg_diagnostic": best_cfg_str,
        })

    student_comparison_rows.sort(key=lambda x: -x["delta_f2_vs_f1"])

    # 7. Confusion Matrices
    cm_f0 = build_confusion_matrix(loocv_f0, students)
    cm_f1 = build_confusion_matrix(loocv_f1, students)
    cm_f2 = build_confusion_matrix(loocv_f2, students)
    cm_f3 = build_confusion_matrix(loocv_f3, students)

    # Pairwise Confusion Comparison (F1 vs F2)
    conf_pairs_rows = []
    pairs_improved, pairs_worsened, pairs_unchanged = 0, 0, 0
    for i in range(len(students)):
        for j in range(i + 1, len(students)):
            sa, sb = students[i], students[j]
            tot_f1 = cm_f1[i, j] + cm_f1[j, i]
            tot_f2 = cm_f2[i, j] + cm_f2[j, i]
            d = tot_f2 - tot_f1
            if d < 0:
                pairs_improved += 1
            elif d > 0:
                pairs_worsened += 1
            else:
                pairs_unchanged += 1
            if tot_f1 > 0 or tot_f2 > 0:
                conf_pairs_rows.append({
                    "class_a": sa,
                    "class_b": sb,
                    "ab_f1": int(cm_f1[i, j]),
                    "ba_f1": int(cm_f1[j, i]),
                    "total_f1": int(tot_f1),
                    "ab_f2": int(cm_f2[i, j]),
                    "ba_f2": int(cm_f2[j, i]),
                    "total_f2": int(tot_f2),
                    "delta_f2_vs_f1": int(d),
                })

    conf_pairs_rows.sort(key=lambda x: x["delta_f2_vs_f1"])

    # 8. Known Diagnostic Cases
    def get_case(name):
        return [r for r in student_comparison_rows if r["student"] == name][0]

    case_dinnar = get_case("Febrian Dinnar Purnama")
    case_fahim = get_case("Fahim J Mujaddid")
    case_brama = get_case("Bramasetya Raka Purnama")
    case_dimas = get_case("Dimas Wahyu Prasetyo")
    case_rakha = get_case("Rakha Burhannudin Majid")

    # 9. Computational Cost
    cost_data = [
        {
            "config": "F0 (128x128, PPC=8)",
            "dim": X_f0.shape[1],
            "dim_ratio_vs_f1": round(X_f0.shape[1] / X_f1.shape[1], 4),
            "ram_mb": round(X_f0.nbytes / 1024 / 1024, 2),
            "mean_hog_ms": round(float(np.mean(t_f0)) * 1000, 2),
            "total_hog_s": round(float(np.sum(t_f0)), 2),
            "loocv_eval_s": round(t_eval_f0, 3),
        },
        {
            "config": "F1 (256x256, PPC=8)",
            "dim": X_f1.shape[1],
            "dim_ratio_vs_f1": 1.0,
            "ram_mb": round(X_f1.nbytes / 1024 / 1024, 2),
            "mean_hog_ms": round(float(np.mean(t_f1)) * 1000, 2),
            "total_hog_s": round(float(np.sum(t_f1)), 2),
            "loocv_eval_s": round(t_eval_f1, 3),
        },
        {
            "config": "F2 (256x256, PPC=16)",
            "dim": X_f2.shape[1],
            "dim_ratio_vs_f1": round(X_f2.shape[1] / X_f1.shape[1], 4),
            "ram_mb": round(X_f2.nbytes / 1024 / 1024, 2),
            "mean_hog_ms": round(float(np.mean(t_f2)) * 1000, 2),
            "total_hog_s": round(float(np.sum(t_f2)), 2),
            "loocv_eval_s": round(t_eval_f2, 3),
        },
        {
            "config": "F3 (256x256, PPC=12)",
            "dim": X_f3.shape[1],
            "dim_ratio_vs_f1": round(X_f3.shape[1] / X_f1.shape[1], 4),
            "ram_mb": round(X_f3.nbytes / 1024 / 1024, 2),
            "mean_hog_ms": round(float(np.mean(t_f3)) * 1000, 2),
            "total_hog_s": round(float(np.sum(t_f3)), 2),
            "loocv_eval_s": round(t_eval_f3, 3),
        },
    ]

    # 10. Configuration Summary
    cfg_summary = [
        {
            "config": "F0",
            "image_size": "128x128",
            "ppc": "(8,8)",
            "dim": X_f0.shape[1],
            "overall_acc": round(acc_f0, 2),
            "pos1_acc": round(p1_f0, 2),
            "pos16_acc": round(p16_f0, 2),
            "pos2_20_acc": round(prest_f0, 2),
            "winning_vote_ge3_pct": nbr_f0,
            "mean_vote": vote_f0,
            "mean_margin": margin_f0,
            "sep_ratio": dm_f0["separability_ratio"],
            "mean_hog_ms": round(float(np.mean(t_f0)) * 1000, 2),
            "ram_mb": round(X_f0.nbytes / 1024 / 1024, 2),
        },
        {
            "config": "F1",
            "image_size": "256x256",
            "ppc": "(8,8)",
            "dim": X_f1.shape[1],
            "overall_acc": round(acc_f1, 2),
            "pos1_acc": round(p1_f1, 2),
            "pos16_acc": round(p16_f1, 2),
            "pos2_20_acc": round(prest_f1, 2),
            "winning_vote_ge3_pct": nbr_f1,
            "mean_vote": vote_f1,
            "mean_margin": margin_f1,
            "sep_ratio": dm_f1["separability_ratio"],
            "mean_hog_ms": round(float(np.mean(t_f1)) * 1000, 2),
            "ram_mb": round(X_f1.nbytes / 1024 / 1024, 2),
        },
        {
            "config": "F2",
            "image_size": "256x256",
            "ppc": "(16,16)",
            "dim": X_f2.shape[1],
            "overall_acc": round(acc_f2, 2),
            "pos1_acc": round(p1_f2, 2),
            "pos16_acc": round(p16_f2, 2),
            "pos2_20_acc": round(prest_f2, 2),
            "winning_vote_ge3_pct": nbr_f2,
            "mean_vote": vote_f2,
            "mean_margin": margin_f2,
            "sep_ratio": dm_f2["separability_ratio"],
            "mean_hog_ms": round(float(np.mean(t_f2)) * 1000, 2),
            "ram_mb": round(X_f2.nbytes / 1024 / 1024, 2),
        },
        {
            "config": "F3",
            "image_size": "256x256",
            "ppc": "(12,12)",
            "dim": X_f3.shape[1],
            "overall_acc": round(acc_f3, 2),
            "pos1_acc": round(p1_f3, 2),
            "pos16_acc": round(p16_f3, 2),
            "pos2_20_acc": round(prest_f3, 2),
            "winning_vote_ge3_pct": nbr_f3,
            "mean_vote": vote_f3,
            "mean_margin": margin_f3,
            "sep_ratio": dm_f3["separability_ratio"],
            "mean_hog_ms": round(float(np.mean(t_f3)) * 1000, 2),
            "ram_mb": round(X_f3.nbytes / 1024 / 1024, 2),
        },
    ]

    # 11. Save all CSVs and JSONs
    print("\nSaving Experiment F outputs...")

    # configuration_summary.csv
    with open(os.path.join(OUT_DIR, "configuration_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cfg_summary[0].keys()))
        w.writeheader(); w.writerows(cfg_summary)

    # per_student_comparison.csv
    with open(os.path.join(OUT_DIR, "per_student_comparison.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(student_comparison_rows[0].keys()))
        w.writeheader(); w.writerows(student_comparison_rows)

    # position_comparison.csv
    with open(os.path.join(OUT_DIR, "position_comparison.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pos_comparison_rows[0].keys()))
        w.writeheader(); w.writerows(pos_comparison_rows)

    # sample_transitions_f1_f2.csv
    with open(os.path.join(OUT_DIR, "sample_transitions_f1_f2.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trans_f1_f2_rows[0].keys()))
        w.writeheader(); w.writerows(trans_f1_f2_rows)

    # confusion matrices
    for name, cm_mat in [("confusion_matrix_f0.csv", cm_f0),
                         ("confusion_matrix_f1.csv", cm_f1),
                         ("confusion_matrix_f2.csv", cm_f2),
                         ("confusion_matrix_f3.csv", cm_f3)]:
        with open(os.path.join(OUT_DIR, name), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["actual \\ pred"] + students)
            for i, st in enumerate(students):
                writer.writerow([st] + list(cm_mat[i]))

    # confusion_pair_comparison.csv
    if conf_pairs_rows:
        with open(os.path.join(OUT_DIR, "confusion_pair_comparison.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(conf_pairs_rows[0].keys()))
            w.writeheader(); w.writerows(conf_pairs_rows)

    # computational_cost.csv
    with open(os.path.join(OUT_DIR, "computational_cost.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cost_data[0].keys()))
        w.writeheader(); w.writerows(cost_data)

    # experiment_f_summary.json
    master_json = {
        "experiment": "F — HOG Scale Sensitivity Analysis",
        "configurations": cfg_summary,
        "transitions": {
            "F1_to_F2": trans_f1_f2,
            "F1_to_F3": trans_f1_f3,
            "F2_to_F3": trans_f2_f3,
        },
        "student_transitions_f1_f2": {
            "improved": st_improved_f1_f2,
            "unchanged": st_unchanged_f1_f2,
            "regressed": st_regressed_f1_f2,
        },
        "diagnostic_cases": {
            "dinnar": case_dinnar,
            "fahim": case_fahim,
            "brama": case_brama,
            "dimas": case_dimas,
            "rakha": case_rakha,
        },
        "pairwise_confusion_f1_vs_f2": {
            "improved": pairs_improved,
            "unchanged": pairs_unchanged,
            "worsened": pairs_worsened,
        },
        "computational_cost": cost_data,
    }

    with open(os.path.join(OUT_DIR, "experiment_f_summary.json"), "w", encoding="utf-8") as f:
        json.dump(master_json, f, indent=2)

    # 12. Console Output
    print("\n" + "=" * 80)
    print("EXPERIMENT F FINAL REPORT SUMMARY")
    print("=" * 80)
    fmt = "{:<28} {:>10} {:>10} {:>10} {:>10}"
    print(fmt.format("Metric", "F0 (128/8)", "F1 (256/8)", "F2 (256/16)", "F3 (256/12)"))
    print("-" * 72)
    print(fmt.format("Overall Accuracy", f"{acc_f0:.2f}%", f"{acc_f1:.2f}%", f"{acc_f2:.2f}%", f"{acc_f3:.2f}%"))
    print(fmt.format("Position #1", f"{p1_f0:.2f}%", f"{p1_f1:.2f}%", f"{p1_f2:.2f}%", f"{p1_f3:.2f}%"))
    print(fmt.format("Position #16", f"{p16_f0:.2f}%", f"{p16_f1:.2f}%", f"{p16_f2:.2f}%", f"{p16_f3:.2f}%"))
    print(fmt.format("Position #2-20", f"{prest_f0:.2f}%", f"{prest_f1:.2f}%", f"{prest_f2:.2f}%", f"{prest_f3:.2f}%"))
    print(fmt.format("Feature Dimension", str(X_f0.shape[1]), str(X_f1.shape[1]), str(X_f2.shape[1]), str(X_f3.shape[1])))
    print(fmt.format("Vote >=3/5", f"{nbr_f0:.2f}%", f"{nbr_f1:.2f}%", f"{nbr_f2:.2f}%", f"{nbr_f3:.2f}%"))
    print(fmt.format("Mean Winning Vote", f"{vote_f0:.2f}%", f"{vote_f1:.2f}%", f"{vote_f2:.2f}%", f"{vote_f3:.2f}%"))
    print(fmt.format("Mean Margin", f"{margin_f0:.2f}%", f"{margin_f1:.2f}%", f"{margin_f2:.2f}%", f"{margin_f3:.2f}%"))
    print(fmt.format("Separability Ratio", f"{dm_f0['separability_ratio']:.4f}", f"{dm_f1['separability_ratio']:.4f}", f"{dm_f2['separability_ratio']:.4f}", f"{dm_f3['separability_ratio']:.4f}"))
    print(fmt.format("HOG Time (ms/img)", f"{np.mean(t_f0)*1000:.2f}", f"{np.mean(t_f1)*1000:.2f}", f"{np.mean(t_f2)*1000:.2f}", f"{np.mean(t_f3)*1000:.2f}"))
    print(fmt.format("Memory (MB)", f"{X_f0.nbytes/1024/1024:.2f}", f"{X_f1.nbytes/1024/1024:.2f}", f"{X_f2.nbytes/1024/1024:.2f}", f"{X_f3.nbytes/1024/1024:.2f}"))

    print("\nPrimary F1 -> F2 Transition:")
    print(f"  WRONG -> CORRECT (WC)  : {trans_f1_f2['WC']} samples ({trans_f1_f2['WC']/n*100:.1f}%)")
    print(f"  CORRECT -> WRONG (CW)  : {trans_f1_f2['CW']} samples ({trans_f1_f2['CW']/n*100:.1f}%)")
    print(f"  Net improvement        : {net_f1_f2:+d} samples")
    print(f"  Students improved      : {st_improved_f1_f2} / {len(students)}")
    print(f"  Students unchanged     : {st_unchanged_f1_f2} / {len(students)}")
    print(f"  Students regressed     : {st_regressed_f1_f2} / {len(students)}")

    print("\nDiagnostic Cases:")
    print(f"  Dinnar : F0={case_dinnar['acc_f0']:.1f}% -> F1={case_dinnar['acc_f1']:.1f}% -> F2={case_dinnar['acc_f2']:.1f}% -> F3={case_dinnar['acc_f3']:.1f}% (Best: {case_dinnar['best_cfg_diagnostic']})")
    print(f"  Fahim  : F0={case_fahim['acc_f0']:.1f}% -> F1={case_fahim['acc_f1']:.1f}% -> F2={case_fahim['acc_f2']:.1f}% -> F3={case_fahim['acc_f3']:.1f}% (Best: {case_fahim['best_cfg_diagnostic']})")
    print(f"  Brama  : F0={case_brama['acc_f0']:.1f}% -> F1={case_brama['acc_f1']:.1f}% -> F2={case_brama['acc_f2']:.1f}% -> F3={case_brama['acc_f3']:.1f}% (Best: {case_brama['best_cfg_diagnostic']})")
    print(f"  Dimas  : F0={case_dimas['acc_f0']:.1f}% -> F1={case_dimas['acc_f1']:.1f}% -> F2={case_dimas['acc_f2']:.1f}% -> F3={case_dimas['acc_f3']:.1f}% (Best: {case_dimas['best_cfg_diagnostic']})")
    print(f"  Rakha  : F0={case_rakha['acc_f0']:.1f}% -> F1={case_rakha['acc_f1']:.1f}% -> F2={case_rakha['acc_f2']:.1f}% -> F3={case_rakha['acc_f3']:.1f}% (Best: {case_rakha['best_cfg_diagnostic']})")

    print("=" * 80)
    print("EXPERIMENT F EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment_f()
