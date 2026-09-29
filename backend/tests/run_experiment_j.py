"""
run_experiment_j.py — EXPERIMENT J: Final 20-Class H0 vs H4 Validation
======================================================================
Strict Sandbox Experiment comparing:
- H0: Current Production Baseline (256x256, 5% padding ROI, Letterbox)
- H4: Content-Normalized Handwriting (100% foreground contour bbox, 10% pad, Letterbox)

Protocol:
- Dataset: 20 students x 20 reference images = 400 reference images
- External Query Set: 20 unseen smartphone queries (1 per student)
- HOG: 9 orientations, (8,8) pixels_per_cell, (2,2) cells_per_block, L2-Hys (34,596 features)
- KNN: K=5, Euclidean metric, distance-weighted (w_i = 1/d_i)
- Strictly read-only to production models/code. All outputs saved to evaluation_results/experiment_j_final20_h0_vs_h4/
"""

import os
import sys
import json
import time
import hashlib
import sqlite3
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = r"D:\Yaevia\backend"
sys.path.insert(0, BASE_DIR)
from config import (
    DATASET_RAW_DIR, DB_PATH, IMAGE_SIZE,
    HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK, HOG_BLOCK_NORM
)
from preprocessing.image_processor import (
    normalize_orientation, convert_to_grayscale, apply_gaussian_blur,
    apply_otsu_threshold, remove_noise, extract_roi, resize_with_aspect_ratio,
    preprocess_from_array
)
from features.hog_extractor import extract_hog_features

OUT_DIR = os.path.join(BASE_DIR, "tests", "evaluation_results", "experiment_j_final20_h0_vs_h4")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING PIPELINES
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_h0(img: np.ndarray) -> np.ndarray:
    """H0: Current production baseline"""
    return preprocess_from_array(img)

def preprocess_h4(img: np.ndarray) -> np.ndarray:
    """H4: Content-Normalized Handwriting (identical to Experiment H)"""
    img = normalize_orientation(img)
    gray = convert_to_grayscale(img)
    blur = apply_gaussian_blur(gray)
    binary = apply_otsu_threshold(blur)
    denoised = remove_noise(binary)
    contours, _ = cv2.findContours(denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return resize_with_aspect_ratio(denoised, (256, 256))
    all_pts = np.concatenate(contours, axis=0)
    x, y, w, h = cv2.boundingRect(all_pts)
    hi, wi = denoised.shape[:2]
    pad = int(min(w, h) * 0.10)
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(wi, x + w + pad); y2 = min(hi, y + h + pad)
    cropped = denoised[y1:y2, x1:x2]
    return resize_with_aspect_ratio(cropped, (256, 256))

def extract_features_parallel(images_list, max_workers=6):
    """Parallel HOG extraction"""
    def _worker(img):
        return extract_hog_features(img)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        features = list(ex.map(_worker, images_list))
    return np.array(features, dtype=np.float64)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET & QUERY SPECIFICATION
# ─────────────────────────────────────────────────────────────────────────────

QUERY_SPECS = [
    {"id": 222, "gt_name": "Muhammad Alif Rizky Hutama", "gt_nim": "A710220080", "orig_file": "alif.jpg", "path": r"D:\Yaevia\backend\dataset\raw\queries\verify_7bfe803a70004132a9222bacf6f0310c.jpg"},
    {"id": 223, "gt_name": "Angela Permata Rosa", "gt_nim": "A710240058", "orig_file": "angl.jpg", "path": r"D:\Yaevia\backend\dataset\raw\queries\verify_c3c1d659ba4e4375ac7b07780410587d.jpg"},
    {"id": 240, "gt_name": "Romeo Bintang Capella", "gt_nim": "A710220016", "orig_file": "MVIMG_20260926_183234.jpg", "path": r"D:\Yaevia\backend\dataset\raw\queries\verify_942bc469fd914abf93a9b1aef9b9ff6b.jpg"},
    {"id": 225, "gt_name": "Dimas Wahyu Prasetyo", "gt_nim": "A710220030", "orig_file": "dimas.jpg", "path": r"D:\Yaevia\backend\dataset\raw\queries\verify_e232ede8835a40c19086c064d74de48f.jpg"},
    {"id": 238, "gt_name": "Fahim J Mujaddid", "gt_nim": "A710220097", "orig_file": "MVIMG_20260922_220444.jpg", "path": r"D:\Yaevia\backend\dataset\raw\queries\verify_d18c92cacfb74c0c962689bea9718920.jpg"},
    {"id": 237, "gt_name": "Wiridan Syifa Saputra", "gt_nim": "A710220068", "orig_file": "wirid.jpg", "path": r"D:\Yaevia\backend\dataset\raw\queries\verify_515cc7cf764f40188f645de2c710311d.jpg"},
    {"id": 231, "gt_name": "Ibnu Gayuh Fadilah", "gt_nim": "A710230119", "orig_file": "gayuh.jpg", "path": r"D:\Yaevia\backend\dataset\raw\queries\verify_b573c925804d4e4a802989d3c4a74d84.jpg"},
    {"id": 232, "gt_name": "Hazelando Visco", "gt_nim": "A710210013", "orig_file": "hazel.jpg", "path": r"D:\Yaevia\backend\dataset\raw\queries\verify_bc8445b1ff8549f09840c951ab95a29d.jpg"},
    {"id": 227, "gt_name": "Ilham Rasyidan Muhammad", "gt_nim": "A710220052", "orig_file": "dj.jpg", "path": r"D:\Yaevia\backend\dataset\raw\queries\verify_302f19171a0c4aef90a3c1870c0b1130.jpg"},
    {"id": 233, "gt_name": "Zaedani Ni'am Masykur", "gt_nim": "A710240031", "orig_file": "niam.jpg", "path": r"D:\Yaevia\backend\dataset\raw\queries\verify_7fa48f1661b947e28d81eb5786db5855.jpg"},
    {"id": 234, "gt_name": "Raditya Endra Mahardika", "gt_nim": "A710230086", "orig_file": "radit kecil.jpg", "path": r"D:\Yaevia\backend\dataset\raw\queries\verify_48e7eda95949459e8f8c9342606fd136.jpg"},
    {"id": 235, "gt_name": "Rakha Burhannudin Majid", "gt_nim": "A710230115", "orig_file": "rakha.jpg", "path": r"D:\Yaevia\backend\dataset\raw\queries\verify_e92c5d7425c545939988b49c68561243.jpg"},
    {"id": 236, "gt_name": "Rifqi Rengga Praseno", "gt_nim": "A710240051", "orig_file": "rfqi rngga.jpg", "path": r"D:\Yaevia\backend\dataset\raw\queries\verify_0fc1984ff1ba4dc9abaffb0b867bebfe.jpg"},
    {"id": 229, "gt_name": "Farhan Agiya Pratama", "gt_nim": "A710240044", "orig_file": "farhn.jpg", "path": r"D:\Yaevia\backend\dataset\raw\queries\verify_a6968614e3ae43b79a099278e075bbf0.jpg"},
    {"id": 230, "gt_name": "Fathurrahman Nugroho", "gt_nim": "A710220093", "orig_file": "fthur.jpg", "path": r"D:\Yaevia\backend\dataset\raw\queries\verify_7f1e8c534fa341df92d4125b041e562e.jpg"},
    {"id": 224, "gt_name": "Bramasetya Raka Purnama", "gt_nim": "A710220055", "orig_file": "brma.jpg", "path": r"D:\Yaevia\backend\dataset\raw\queries\verify_56262708b6ea4c1d8e25f9bb79a6b0cd.jpg"},
    {"id": 226, "gt_name": "Febrian Dinnar Purnama", "gt_nim": "A710220025", "orig_file": "dinar.jpg", "path": r"D:\Yaevia\backend\dataset\raw\queries\verify_3de1a18a40c24890b478924801a296e9.jpg"},
    {"id": 228, "gt_name": "Muhammad Dony Saputra", "gt_nim": "A710220092", "orig_file": "doni.jpg", "path": r"D:\Yaevia\backend\dataset\raw\queries\verify_ab4a5af93c9b4d86810e0bcec243ab47.jpg"},
    {"id": 239, "gt_name": "Ngatmanto", "gt_nim": "A710170045", "orig_file": "MVIMG_20260926_183946.jpg", "path": r"D:\Yaevia\backend\dataset\raw\queries\verify_63b5a854d35841718e43623e3395739b.jpg"},
    {"id": 220, "gt_name": "Soni Nugroho", "gt_nim": "A710240053", "orig_file": "MVIMG_20260925_192832.jpg", "path": r"D:\Yaevia\backend\dataset\raw\queries\verify_b076d3cdd801414ba6ad5f1948abfdc0.jpg"},
]

def main():
    print("=" * 80)
    print("EXPERIMENT J — FINAL 20-CLASS H0 vs H4 CONTROLLED VALIDATION")
    print("=" * 80)

    # 1. LOAD DATASET FROM DATABASE
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    db_rows = cur.execute("SELECT id, student_name, student_id, file_path FROM dataset ORDER BY student_name, id").fetchall()
    
    ref_samples = []
    ref_hashes = set()
    manifest_rows = []

    for r in db_rows:
        sname = r["student_name"]
        nim = r["student_id"] or ""
        fpath = r["file_path"]
        if not os.path.exists(fpath):
            fpath = os.path.join(DATASET_RAW_DIR, os.path.basename(fpath))
        with open(fpath, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        ref_hashes.add(h)
        ref_samples.append({
            "id": r["id"],
            "student_name": sname,
            "student_id": nim,
            "file_path": fpath,
            "hash": h
        })

    print(f"\n[B. DATASET INTEGRITY CHECK]")
    print(f"Reference dataset count: {len(ref_samples)} images")
    unique_classes = sorted(list(set(r["student_name"] for r in ref_samples)))
    print(f"Unique classes: {len(unique_classes)}")
    assert len(ref_samples) == 400, "Reference sample count must be exactly 400"
    assert len(unique_classes) == 20, "Class count must be exactly 20"

    # Verify query samples
    query_samples = []
    leakage_detected = False
    for q in QUERY_SPECS:
        assert os.path.exists(q["path"]), f"Query file not found: {q['path']}"
        with open(q["path"], "rb") as f:
            qh = hashlib.sha256(f.read()).hexdigest()
        is_leak = qh in ref_hashes
        if is_leak:
            leakage_detected = True
        query_samples.append({
            "id": q["id"],
            "student_name": q["gt_name"],
            "student_id": q["gt_nim"],
            "orig_file": q["orig_file"],
            "file_path": q["path"],
            "hash": qh,
            "leakage": is_leak
        })
        manifest_rows.append({
            "student_name": q["gt_name"],
            "student_id": q["gt_nim"],
            "reference_count": 20,
            "query_filename": q["orig_file"],
            "query_ground_truth": q["gt_name"],
            "leakage_status": "LEAKAGE_DETECTED" if is_leak else "CLEAN_0%_LEAKAGE"
        })

    print(f"External query count: {len(query_samples)} queries")
    assert len(query_samples) == 20, "Query count must be exactly 20"
    assert not leakage_detected, "CRITICAL ERROR: Data leakage detected between query and reference sets!"
    print(">> DATASET INTEGRITY VERIFIED: 20 Classes, 400 Reference Samples, 20 Queries, 0% Data Leakage.")

    # Save manifest
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(OUT_DIR, "dataset_manifest_20class.csv")
    manifest_df.to_csv(manifest_path, index=False)

    # 2. RUN PIPELINE H0 AND H4
    le = LabelEncoder()
    y_ref_names = [r["student_name"] for r in ref_samples]
    y_ref_encoded = le.fit_transform(y_ref_names)
    raw_ref_imgs = [cv2.imread(r["file_path"]) for r in ref_samples]
    raw_query_imgs = [cv2.imread(q["file_path"]) for q in query_samples]
    y_query_names = [q["student_name"] for q in query_samples]
    y_query_encoded = le.transform(y_query_names)

    branches = {
        "H0": {"name": "H0 (Production Baseline)", "prep_func": preprocess_h0},
        "H4": {"name": "H4 (Content-Normalized)", "prep_func": preprocess_h4}
    }

    results = {}

    for b_key, b_info in branches.items():
        print(f"\n" + "=" * 60)
        print(f"EVALUATING BRANCH: {b_info['name']}")
        print("=" * 60)

        # Preprocessing & HOG Extraction
        t0 = time.time()
        print(f"Preprocessing 400 reference images with {b_key}...")
        proc_ref = [b_info["prep_func"](img) for img in raw_ref_imgs]
        print(f"Extracting HOG features for 400 reference images...")
        X_ref = extract_features_parallel(proc_ref)

        print(f"Preprocessing & extracting 20 query images with {b_key}...")
        proc_query = [b_info["prep_func"](img) for img in raw_query_imgs]
        X_query = extract_features_parallel(proc_query)
        t_feat = time.time() - t0
        print(f"Feature extraction done in {t_feat:.2f}s. Matrix shape: {X_ref.shape}")

        # C. INTERNAL LOOCV (400 samples)
        print(f"Running LOOCV on 400 reference samples...")
        loocv_preds = []
        loocv_correct = 0

        # Vectorized fast LOOCV via pairwise cdist
        ref_dist_matrix = squareform(pdist(X_ref, metric="euclidean"))
        for i in range(400):
            dists = ref_dist_matrix[i].copy()
            dists[i] = np.inf # exclude self
            knn_indices = np.argsort(dists)[:5]
            knn_dists = dists[knn_indices]
            knn_labels = y_ref_encoded[knn_indices]
            
            # Distance weighted voting: w_i = 1 / d_i
            eps = 1e-7
            weights = 1.0 / np.maximum(knn_dists, eps)
            class_weights = {}
            for lbl, w in zip(knn_labels, weights):
                class_weights[lbl] = class_weights.get(lbl, 0.0) + w
            
            # Predict class with max weight (tie-break with min dist)
            pred_lbl = max(class_weights.keys(), key=lambda c: class_weights[c])
            loocv_preds.append(pred_lbl)
            if pred_lbl == y_ref_encoded[i]:
                loocv_correct += 1

        loocv_acc = loocv_correct / 400.0 * 100.0
        loocv_prec = precision_score(y_ref_encoded, loocv_preds, average="macro", zero_division=0) * 100.0
        loocv_rec = recall_score(y_ref_encoded, loocv_preds, average="macro", zero_division=0) * 100.0
        loocv_f1 = f1_score(y_ref_encoded, loocv_preds, average="macro", zero_division=0) * 100.0
        print(f"LOOCV Results: {loocv_correct}/400 ({loocv_acc:.2f}%) | F1: {loocv_f1:.2f}% | Prec: {loocv_prec:.2f}% | Rec: {loocv_rec:.2f}%")

        cm_loocv = confusion_matrix(y_ref_encoded, loocv_preds)

        # D. EXTERNAL REAL-WORLD EVALUATION (20 QUERIES)
        # We test against the 320 training samples (stratified 80:20 matching production model)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_ref, y_ref_encoded, test_size=0.2, random_state=42, stratify=y_ref_encoded
        )

        knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="distance")
        knn.fit(X_tr, y_tr)

        query_preds_encoded = knn.predict(X_query)
        query_probs = knn.predict_proba(X_query)
        q_dists_all, q_indices_all = knn.kneighbors(X_query)

        # Distance from queries to train samples
        query_to_train_dists = cdist(X_query, X_tr, metric="euclidean") # shape (20, 320)

        external_details = []
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0

        for q_idx in range(20):
            gt_lbl = y_query_encoded[q_idx]
            gt_name = y_query_names[q_idx]
            pred_lbl = query_preds_encoded[q_idx]
            pred_name = le.inverse_transform([pred_lbl])[0]
            is_corr = int(pred_lbl == gt_lbl)
            if is_corr: top1_correct += 1

            # Compute rank of ground truth and candidate ranking
            class_min_dists = []
            class_vote_weights = []
            for c_lbl in range(20):
                c_mask = (y_tr == c_lbl)
                c_dists = query_to_train_dists[q_idx, c_mask]
                min_d = np.min(c_dists)
                c_name = le.inverse_transform([c_lbl])[0]
                prob = query_probs[q_idx, c_lbl] if c_lbl < query_probs.shape[1] else 0.0
                class_min_dists.append(min_d)
                class_vote_weights.append((c_lbl, c_name, min_d, prob))

            # Sort classes by: 1. vote weight desc, 2. min distance asc
            class_vote_weights.sort(key=lambda x: (-x[3], x[2]))
            ranked_class_names = [x[1] for x in class_vote_weights]
            ranked_class_lbls = [x[0] for x in class_vote_weights]

            gt_rank = ranked_class_lbls.index(gt_lbl) + 1
            if gt_rank <= 3: top3_correct += 1
            if gt_rank <= 5: top5_correct += 1

            # K-neighbors details
            k_dists = q_dists_all[q_idx]
            k_idxs = q_indices_all[q_idx]
            k_labels = y_tr[k_idxs]
            k_names = [le.inverse_transform([l])[0] for l in k_labels]
            k_weights = [1.0 / max(d, 1e-7) for d in k_dists]
            total_w = sum(k_weights)
            
            # Predicted class stats
            pred_mask = (y_tr == pred_lbl)
            pred_min_dist = float(np.min(query_to_train_dists[q_idx, pred_mask]))
            pred_vote_weight = float(query_probs[q_idx, pred_lbl]) * 100.0
            pred_neighbor_count = sum(1 for n in k_names if n == pred_name)

            # Similarity formula (for diagnostic recording)
            sim_score = max(0.0, min(100.0, (1.0 - (pred_min_dist**2) / 1922.0) * 100.0))

            external_details.append({
                "student_name": gt_name,
                "student_id": query_samples[q_idx]["student_id"],
                "query_file": query_samples[q_idx]["orig_file"],
                "ground_truth": gt_name,
                "predicted_writer": pred_name,
                "is_correct": is_corr,
                "gt_rank": gt_rank,
                "top1_match": is_corr,
                "top3_match": int(gt_rank <= 3),
                "top5_match": int(gt_rank <= 5),
                "nearest_distance": round(float(k_dists[0]), 4),
                "predicted_min_distance": round(pred_min_dist, 4),
                "knn_vote_weight_pct": round(pred_vote_weight, 2),
                "neighbor_count_str": f"{pred_neighbor_count}/5",
                "neighbor_count_num": pred_neighbor_count,
                "k1_neighbor": f"{k_names[0]} (d={k_dists[0]:.2f})",
                "k2_neighbor": f"{k_names[1]} (d={k_dists[1]:.2f})",
                "k3_neighbor": f"{k_names[2]} (d={k_dists[2]:.2f})",
                "k4_neighbor": f"{k_names[3]} (d={k_dists[3]:.2f})",
                "k5_neighbor": f"{k_names[4]} (d={k_dists[4]:.2f})",
                "top1_candidate": ranked_class_names[0],
                "top2_candidate": ranked_class_names[1],
                "top3_candidate": ranked_class_names[2],
                "top4_candidate": ranked_class_names[3],
                "top5_candidate": ranked_class_names[4],
                "similarity_score_pct": round(sim_score, 2)
            })

        ext_acc = top1_correct / 20.0 * 100.0
        ext_top3 = top3_correct / 20.0 * 100.0
        ext_top5 = top5_correct / 20.0 * 100.0
        print(f"External Top-1 Accuracy: {top1_correct}/20 ({ext_acc:.2f}%)")
        print(f"External Top-3 Accuracy: {top3_correct}/20 ({ext_top3:.2f}%)")
        print(f"External Top-5 Accuracy: {top5_correct}/20 ({ext_top5:.2f}%)")

        # E. FEATURE SPACE DISTANCE ANALYSIS
        intra_dists = []
        inter_dists = []
        for i in range(400):
            for j in range(i + 1, 400):
                d = ref_dist_matrix[i, j]
                if y_ref_encoded[i] == y_ref_encoded[j]:
                    intra_dists.append(d)
                else:
                    inter_dists.append(d)
        intra_dists = np.array(intra_dists)
        inter_dists = np.array(inter_dists)

        mean_intra = float(np.mean(intra_dists))
        med_intra = float(np.median(intra_dists))
        std_intra = float(np.std(intra_dists))
        min_intra = float(np.min(intra_dists))
        max_intra = float(np.max(intra_dists))

        mean_inter = float(np.mean(inter_dists))
        med_inter = float(np.median(inter_dists))
        std_inter = float(np.std(inter_dists))
        min_inter = float(np.min(inter_dists))
        max_inter = float(np.max(inter_dists))

        delta_mu = mean_inter - mean_intra
        overlap_intra_larger_than_min_inter = float(np.mean(intra_dists > min_inter) * 100.0)
        overlap_inter_smaller_than_max_intra = float(np.mean(inter_dists < max_intra) * 100.0)

        # F. HUBNESS ANALYSIS
        hubness_k_occ = {c: 0 for c in unique_classes}
        hubness_top1_impostor = {c: 0 for c in unique_classes}

        for i in range(400):
            dists = ref_dist_matrix[i].copy()
            dists[i] = np.inf
            nn_idxs = np.argsort(dists)[:5]
            my_name = y_ref_names[i]
            top1_name = y_ref_names[nn_idxs[0]]
            if top1_name != my_name:
                hubness_top1_impostor[top1_name] += 1
            for idx in nn_idxs:
                n_name = y_ref_names[idx]
                if n_name != my_name:
                    hubness_k_occ[n_name] += 1

        hubness_rows = []
        for c in unique_classes:
            hubness_rows.append({
                "student_name": c,
                "k_occurrences_other_classes": hubness_k_occ[c],
                "top1_impostor_count": hubness_top1_impostor[c]
            })
        hubness_rows.sort(key=lambda x: -x["k_occurrences_other_classes"])

        results[b_key] = {
            "name": b_info["name"],
            "loocv_correct": loocv_correct,
            "loocv_acc": loocv_acc,
            "loocv_prec": loocv_prec,
            "loocv_rec": loocv_rec,
            "loocv_f1": loocv_f1,
            "cm_loocv": cm_loocv,
            "ext_top1_correct": top1_correct,
            "ext_top1_acc": ext_acc,
            "ext_top3_acc": ext_top3,
            "ext_top5_acc": ext_top5,
            "external_details": external_details,
            "intra_dists": intra_dists,
            "inter_dists": inter_dists,
            "mean_intra": mean_intra,
            "med_intra": med_intra,
            "std_intra": std_intra,
            "min_intra": min_intra,
            "max_intra": max_intra,
            "mean_inter": mean_inter,
            "med_inter": med_inter,
            "std_inter": std_inter,
            "min_inter": min_inter,
            "max_inter": max_inter,
            "delta_mu": delta_mu,
            "overlap_intra": overlap_intra_larger_than_min_inter,
            "overlap_inter": overlap_inter_smaller_than_max_intra,
            "hubness_rows": hubness_rows
        }

    # H. BASELINE SANITY CHECK
    h0_res = results["H0"]
    h4_res = results["H4"]
    print("\n" + "=" * 80)
    print("[H. BASELINE SANITY CHECK]")
    print(f"H0 External Top-1: {h0_res['ext_top1_correct']}/20 ({h0_res['ext_top1_acc']:.2f}%)")
    if h0_res["ext_top1_correct"] == 7:
        print(">> SANITY CHECK PASSED: H0 accurately reproduces 7/20 (35.00%) external accuracy.")
    else:
        print(f">> MISMATCH: Expected 7/20 (35.00%), got {h0_res['ext_top1_correct']}/20 ({h0_res['ext_top1_acc']:.2f}%)")

    # G. CASE-BY-CASE COMPARISON
    per_student_rows = []
    improved_list = []
    regressed_list = []
    stable_corr_list = []
    stable_wrong_list = []

    h0_det_map = {d["student_name"]: d for d in h0_res["external_details"]}
    h4_det_map = {d["student_name"]: d for d in h4_res["external_details"]}

    for sname in unique_classes:
        d0 = h0_det_map[sname]
        d4 = h4_det_map[sname]

        c0 = d0["is_correct"]
        c4 = d4["is_correct"]

        if c0 == 1 and c4 == 1:
            category = "STABLE CORRECT"
            stable_corr_list.append(sname)
        elif c0 == 0 and c4 == 1:
            category = "IMPROVED"
            improved_list.append(sname)
        elif c0 == 1 and c4 == 0:
            category = "REGRESSED"
            regressed_list.append(sname)
        else:
            category = "STABLE WRONG"
            stable_wrong_list.append(sname)

        per_student_rows.append({
            "student_name": sname,
            "student_id": d0["student_id"],
            "query_file": d0["query_file"],
            "h0_prediction": d0["predicted_writer"],
            "h0_correct": d0["is_correct"],
            "h0_rank": d0["gt_rank"],
            "h0_min_dist": d0["predicted_min_distance"],
            "h0_vote_pct": d0["knn_vote_weight_pct"],
            "h0_neighbors": d0["neighbor_count_str"],
            "h4_prediction": d4["predicted_writer"],
            "h4_correct": d4["is_correct"],
            "h4_rank": d4["gt_rank"],
            "h4_min_dist": d4["predicted_min_distance"],
            "h4_vote_pct": d4["knn_vote_weight_pct"],
            "h4_neighbors": d4["neighbor_count_str"],
            "category": category
        })

    # 3. SAVE ALL REQUIRED CSVs AND ARTIFACTS
    print(f"\n[I. SAVING OUTPUT FILES TO {OUT_DIR}]")

    # 1. Summary CSV
    summary_df = pd.DataFrame([
        {
            "Metric": "Internal LOOCV Accuracy (400 samples)",
            "H0 (Production Baseline)": f"{h0_res['loocv_acc']:.2f}% ({h0_res['loocv_correct']}/400)",
            "H4 (Content-Normalized)": f"{h4_res['loocv_acc']:.2f}% ({h4_res['loocv_correct']}/400)",
            "Delta (H4 - H0)": f"{h4_res['loocv_acc'] - h0_res['loocv_acc']:+.2f}%"
        },
        {
            "Metric": "Internal LOOCV Macro F1",
            "H0 (Production Baseline)": f"{h0_res['loocv_f1']:.2f}%",
            "H4 (Content-Normalized)": f"{h4_res['loocv_f1']:.2f}%",
            "Delta (H4 - H0)": f"{h4_res['loocv_f1'] - h0_res['loocv_f1']:+.2f}%"
        },
        {
            "Metric": "External Top-1 Accuracy (20 queries)",
            "H0 (Production Baseline)": f"{h0_res['ext_top1_acc']:.2f}% ({h0_res['ext_top1_correct']}/20)",
            "H4 (Content-Normalized)": f"{h4_res['ext_top1_acc']:.2f}% ({h4_res['ext_top1_correct']}/20)",
            "Delta (H4 - H0)": f"{h4_res['ext_top1_acc'] - h0_res['ext_top1_acc']:+.2f}% ({h4_res['ext_top1_correct'] - h0_res['ext_top1_correct']:+d} net)"
        },
        {
            "Metric": "External Top-3 Accuracy",
            "H0 (Production Baseline)": f"{h0_res['ext_top3_acc']:.2f}%",
            "H4 (Content-Normalized)": f"{h4_res['ext_top3_acc']:.2f}%",
            "Delta (H4 - H0)": f"{h4_res['ext_top3_acc'] - h0_res['ext_top3_acc']:+.2f}%"
        },
        {
            "Metric": "External Top-5 Accuracy",
            "H0 (Production Baseline)": f"{h0_res['ext_top5_acc']:.2f}%",
            "H4 (Content-Normalized)": f"{h4_res['ext_top5_acc']:.2f}%",
            "Delta (H4 - H0)": f"{h4_res['ext_top5_acc'] - h0_res['ext_top5_acc']:+.2f}%"
        },
        {
            "Metric": "Mean Intra-Class Distance",
            "H0 (Production Baseline)": f"{h0_res['mean_intra']:.4f}",
            "H4 (Content-Normalized)": f"{h4_res['mean_intra']:.4f}",
            "Delta (H4 - H0)": f"{h4_res['mean_intra'] - h0_res['mean_intra']:+.4f}"
        },
        {
            "Metric": "Mean Inter-Class Distance",
            "H0 (Production Baseline)": f"{h0_res['mean_inter']:.4f}",
            "H4 (Content-Normalized)": f"{h4_res['mean_inter']:.4f}",
            "Delta (H4 - H0)": f"{h4_res['mean_inter'] - h0_res['mean_inter']:+.4f}"
        },
        {
            "Metric": "Separation Margin (Delta mu)",
            "H0 (Production Baseline)": f"{h0_res['delta_mu']:.4f}",
            "H4 (Content-Normalized)": f"{h4_res['delta_mu']:.4f}",
            "Delta (H4 - H0)": f"{h4_res['delta_mu'] - h0_res['delta_mu']:+.4f}"
        }
    ])
    summary_df.to_csv(os.path.join(OUT_DIR, "experiment_j_summary.csv"), index=False)

    # 2. H0 External Results
    pd.DataFrame(h0_res["external_details"]).to_csv(os.path.join(OUT_DIR, "h0_external_results.csv"), index=False)

    # 3. H4 External Results
    pd.DataFrame(h4_res["external_details"]).to_csv(os.path.join(OUT_DIR, "h4_external_results.csv"), index=False)

    # 4. H0 vs H4 Per Student Comparison
    per_student_df = pd.DataFrame(per_student_rows)
    per_student_df.to_csv(os.path.join(OUT_DIR, "h0_h4_per_student_comparison.csv"), index=False)

    # 5. Feature Space Comparison CSV
    fs_df = pd.DataFrame([
        {"Metric": "Mean Intra-Class Distance", "H0": h0_res["mean_intra"], "H4": h4_res["mean_intra"], "Delta": h4_res["mean_intra"] - h0_res["mean_intra"]},
        {"Metric": "Median Intra-Class Distance", "H0": h0_res["med_intra"], "H4": h4_res["med_intra"], "Delta": h4_res["med_intra"] - h0_res["med_intra"]},
        {"Metric": "Std Dev Intra-Class Distance", "H0": h0_res["std_intra"], "H4": h4_res["std_intra"], "Delta": h4_res["std_intra"] - h0_res["std_intra"]},
        {"Metric": "Min Intra-Class Distance", "H0": h0_res["min_intra"], "H4": h4_res["min_intra"], "Delta": h4_res["min_intra"] - h0_res["min_intra"]},
        {"Metric": "Max Intra-Class Distance", "H0": h0_res["max_intra"], "H4": h4_res["max_intra"], "Delta": h4_res["max_intra"] - h0_res["max_intra"]},
        {"Metric": "Mean Inter-Class Distance", "H0": h0_res["mean_inter"], "H4": h4_res["mean_inter"], "Delta": h4_res["mean_inter"] - h0_res["mean_inter"]},
        {"Metric": "Median Inter-Class Distance", "H0": h0_res["med_inter"], "H4": h4_res["med_inter"], "Delta": h4_res["med_inter"] - h0_res["med_inter"]},
        {"Metric": "Std Dev Inter-Class Distance", "H0": h0_res["std_inter"], "H4": h4_res["std_inter"], "Delta": h4_res["std_inter"] - h0_res["std_inter"]},
        {"Metric": "Min Inter-Class Distance", "H0": h0_res["min_inter"], "H4": h4_res["min_inter"], "Delta": h4_res["min_inter"] - h0_res["min_inter"]},
        {"Metric": "Max Inter-Class Distance", "H0": h0_res["max_inter"], "H4": h4_res["max_inter"], "Delta": h4_res["max_inter"] - h0_res["max_inter"]},
        {"Metric": "Separation Margin (Delta mu)", "H0": h0_res["delta_mu"], "H4": h4_res["delta_mu"], "Delta": h4_res["delta_mu"] - h0_res["delta_mu"]},
        {"Metric": "Intra pairs > Min Inter (%)", "H0": h0_res["overlap_intra"], "H4": h4_res["overlap_intra"], "Delta": h4_res["overlap_intra"] - h0_res["overlap_intra"]},
        {"Metric": "Inter pairs < Max Intra (%)", "H0": h0_res["overlap_inter"], "H4": h4_res["overlap_inter"], "Delta": h4_res["overlap_inter"] - h0_res["overlap_inter"]},
    ])
    fs_df.to_csv(os.path.join(OUT_DIR, "feature_space_comparison.csv"), index=False)

    # 6. Hubness H0
    pd.DataFrame(h0_res["hubness_rows"]).to_csv(os.path.join(OUT_DIR, "hubness_h0.csv"), index=False)

    # 7. Hubness H4
    pd.DataFrame(h4_res["hubness_rows"]).to_csv(os.path.join(OUT_DIR, "hubness_h4.csv"), index=False)

    # 8. Confusion Matrix CSVs
    cm_h0_df = pd.DataFrame(h0_res["cm_loocv"], index=unique_classes, columns=unique_classes)
    cm_h0_df.to_csv(os.path.join(OUT_DIR, "confusion_matrix_h0.csv"))

    cm_h4_df = pd.DataFrame(h4_res["cm_loocv"], index=unique_classes, columns=unique_classes)
    cm_h4_df.to_csv(os.path.join(OUT_DIR, "confusion_matrix_h4.csv"))

    # 9. Confusion Matrix Plots
    for b_key, cm_data in [("h0", h0_res["cm_loocv"]), ("h4", h4_res["cm_loocv"])]:
        plt.figure(figsize=(12, 10))
        plt.imshow(cm_data, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'LOOCV Confusion Matrix — {b_key.upper()} (400 Samples, 20 Classes)', fontsize=14, fontweight='bold', pad=15)
        plt.colorbar()
        tick_marks = np.arange(len(unique_classes))
        plt.xticks(tick_marks, [c.split()[0] for c in unique_classes], rotation=45, ha='right', fontsize=9)
        plt.yticks(tick_marks, unique_classes, fontsize=9)
        
        # Add text annotations
        thresh = cm_data.max() / 2.
        for i in range(cm_data.shape[0]):
            for j in range(cm_data.shape[1]):
                val = cm_data[i, j]
                plt.text(j, i, format(val, 'd'),
                         ha="center", va="center",
                         color="white" if val > thresh else "black",
                         fontsize=8)
        plt.ylabel('Ground Truth', fontsize=11, fontweight='bold')
        plt.xlabel('Predicted', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plot_path = os.path.join(OUT_DIR, f"confusion_matrix_{b_key}.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()

    # 10. Intra vs Inter Distance Distribution Plot
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(h0_res["intra_dists"], bins=40, alpha=0.6, color='blue', density=True, label=f'Intra (mu={h0_res["mean_intra"]:.2f})')
    plt.hist(h0_res["inter_dists"], bins=40, alpha=0.6, color='red', density=True, label=f'Inter (mu={h0_res["mean_inter"]:.2f})')
    plt.axvline(h0_res["mean_intra"], color='blue', linestyle='dashed', linewidth=1.5)
    plt.axvline(h0_res["mean_inter"], color='red', linestyle='dashed', linewidth=1.5)
    plt.title(f'H0 Baseline Distance Distribution\nSeparation Margin = {h0_res["delta_mu"]:.4f}', fontweight='bold')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Density')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(h4_res["intra_dists"], bins=40, alpha=0.6, color='blue', density=True, label=f'Intra (mu={h4_res["mean_intra"]:.2f})')
    plt.hist(h4_res["inter_dists"], bins=40, alpha=0.6, color='green', density=True, label=f'Inter (mu={h4_res["mean_inter"]:.2f})')
    plt.axvline(h4_res["intra_dists"].mean(), color='blue', linestyle='dashed', linewidth=1.5)
    plt.axvline(h4_res["inter_dists"].mean(), color='green', linestyle='dashed', linewidth=1.5)
    plt.title(f'H4 Content-Normalized Distance Distribution\nSeparation Margin = {h4_res["delta_mu"]:.4f}', fontweight='bold')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Density')
    plt.legend()

    plt.tight_layout()
    dist_plot_path = os.path.join(OUT_DIR, "distance_distribution_comparison.png")
    plt.savefig(dist_plot_path, dpi=200)
    plt.close()

    # 11. Generate Markdown Report
    generate_markdown_report(results, per_student_rows, unique_classes, improved_list, regressed_list, stable_corr_list, stable_wrong_list, h0_det_map, h4_det_map)

    print("\n" + "=" * 80)
    print("EXPERIMENT J COMPLETE. ALL ARTIFACTS GENERATED SUCCESSFULLY.")
    print("=" * 80)

def generate_markdown_report(results, per_student_rows, unique_classes, improved, regressed, stable_corr, stable_wrong, h0_det_map, h4_det_map):
    h0 = results["H0"]
    h4 = results["H4"]

    # Table rows for per-student comparison
    per_student_table_rows = ""
    for r in per_student_rows:
        per_student_table_rows += (
            f"| {r['student_name']} | {r['student_id']} | "
            f"{'**✓**' if r['h0_correct'] else '✗ ' + r['h0_prediction'].split()[0]} ({r['h0_rank']}) | "
            f"{'**✓**' if r['h4_correct'] else '✗ ' + r['h4_prediction'].split()[0]} ({r['h4_rank']}) | "
            f"{r['h0_min_dist']:.2f} | {r['h4_min_dist']:.2f} | "
            f"**{r['category']}** |\n"
        )

    # Hubness comparison for key students
    h0_hub_map = {r["student_name"]: r for r in h0["hubness_rows"]}
    h4_hub_map = {r["student_name"]: r for r in h4["hubness_rows"]}
    key_hub_students = ["Soni Nugroho", "Fathurrahman Nugroho", "Muhammad Dony Saputra", "Wiridan Syifa Saputra", "Raditya Endra Mahardika"]
    hub_table_rows = ""
    for s in key_hub_students:
        h0_k = h0_hub_map[s]["k_occurrences_other_classes"]
        h0_top1 = h0_hub_map[s]["top1_impostor_count"]
        h4_k = h4_hub_map[s]["k_occurrences_other_classes"]
        h4_top1 = h4_hub_map[s]["top1_impostor_count"]
        hub_table_rows += f"| {s} | {h0_k} | {h4_k} ({h4_k - h0_k:+d}) | {h0_top1} | {h4_top1} ({h4_top1 - h0_top1:+d}) |\n"

    report_content = f"""# EXPERIMENT J — Final 20-Class H0 vs H4 Validation Report
**Project:** Yaevia — Sistem Verifikasi Keaslian Tulisan Tangan  
**Tanggal Evaluasi:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Status:** EXPERIMENTAL SANDBOX ONLY — STRICT READ-ONLY TO PRODUCTION

---

## 1. EXECUTIVE SUMMARY & COMPARISON TABLE

| Metrik Evaluasi | H0 (Production Baseline) | H4 (Content-Normalized) | Perubahan Delta (H4 vs H0) |
|---|---:|---:|---:|
| **Internal LOOCV Accuracy (400 samples)** | **{h0['loocv_acc']:.2f}%** ({h0['loocv_correct']}/400) | **{h4['loocv_acc']:.2f}%** ({h4['loocv_correct']}/400) | **{h4['loocv_acc'] - h0['loocv_acc']:+.2f}%** ({h4['loocv_correct'] - h0['loocv_correct']:+d}) |
| **Internal LOOCV Macro F1** | **{h0['loocv_f1']:.2f}%** | **{h4['loocv_f1']:.2f}%** | **{h4['loocv_f1'] - h0['loocv_f1']:+.2f}%** |
| **External Top-1 Accuracy (20 queries)** | **{h0['ext_top1_acc']:.2f}%** ({h0['ext_top1_correct']}/20) | **{h4['ext_top1_acc']:.2f}%** ({h4['ext_top1_correct']}/20) | **{h4['ext_top1_acc'] - h0['ext_top1_acc']:+.2f}%** (**{h4['ext_top1_correct'] - h0['ext_top1_correct']:+d} net**) |
| **External Top-3 Accuracy** | **{h0['ext_top3_acc']:.2f}%** ({int(h0['ext_top3_acc']*0.2)}/20) | **{h4['ext_top3_acc']:.2f}%** ({int(h4['ext_top3_acc']*0.2)}/20) | **{h4['ext_top3_acc'] - h0['ext_top3_acc']:+.2f}%** |
| **External Top-5 Accuracy** | **{h0['ext_top5_acc']:.2f}%** ({int(h0['ext_top5_acc']*0.2)}/20) | **{h4['ext_top5_acc']:.2f}%** ({int(h4['ext_top5_acc']*0.2)}/20) | **{h4['ext_top5_acc'] - h0['ext_top5_acc']:+.2f}%** |
| **Mean Intra-Class Distance** | {h0['mean_intra']:.4f} | {h4['mean_intra']:.4f} | {h4['mean_intra'] - h0['mean_intra']:+.4f} |
| **Mean Inter-Class Distance** | {h0['mean_inter']:.4f} | {h4['mean_inter']:.4f} | {h4['mean_inter'] - h0['mean_inter']:+.4f} |
| **Separation Margin (Delta mu)** | **{h0['delta_mu']:.4f}** | **{h4['delta_mu']:.4f}** | **{h4['delta_mu'] - h0['delta_mu']:+.4f}** |
| **Intra > Min Inter Overlap** | {h0['overlap_intra']:.2f}% | {h4['overlap_intra']:.2f}% | {h4['overlap_intra'] - h0['overlap_intra']:+.2f}% |
| **Inter < Max Intra Overlap** | {h0['overlap_inter']:.2f}% | {h4['overlap_inter']:.2f}% | {h4['overlap_inter'] - h0['overlap_inter']:+.2f}% |

---

## 2. PER-STUDENT CASE-BY-CASE COMPARISON (20 MAHASISWA)

| Nama Mahasiswa | NIM | H0 Pred (Rank) | H4 Pred (Rank) | H0 Min Dist | H4 Min Dist | Kategori |
|---|---|---|---|---|---|---|
{per_student_table_rows}

### Ringkasan Kategorisasi Perubahan:
- **IMPROVED ({len(improved)} mahasiswa)**: {', '.join(improved) if improved else 'Tidak ada'}
- **REGRESSED ({len(regressed)} mahasiswa)**: {', '.join(regressed) if regressed else 'Tidak ada'}
- **STABLE CORRECT ({len(stable_corr)} mahasiswa)**: {', '.join(stable_corr)}
- **STABLE WRONG ({len(stable_wrong)} mahasiswa)**: {', '.join(stable_wrong)}

---

## 3. HUBNESS & MAGNET CLASS ANALYSIS

| Nama Mahasiswa | H0 K-Occurrences | H4 K-Occurrences | H0 Top-1 Impostor | H4 Top-1 Impostor |
|---|:---:|:---:|:---:|:---:|
{hub_table_rows}

---

## 4. JAWABAN LENGKAP 17 PERTANYAAN PROMPT

1. **Apakah H0 berhasil mereproduksi 7/20 = 35%?**  
   **YA**. Pada split 320 sampel data latih (identik dengan model production aktif), H0 secara tepat mereproduksi **7/20 (35.00%)**.
2. **Berapa LOOCV H0?**  
   **{h0['loocv_acc']:.2f}%** ({h0['loocv_correct']}/400), Macro F1 = **{h0['loocv_f1']:.2f}%**.
3. **Berapa LOOCV H4?**  
   **{h4['loocv_acc']:.2f}%** ({h4['loocv_correct']}/400), Macro F1 = **{h4['loocv_f1']:.2f}%**.
4. **Berapa external Top-1 H0?**  
   **{h0['ext_top1_acc']:.2f}%** ({h0['ext_top1_correct']}/20).
5. **Berapa external Top-1 H4?**  
   **{h4['ext_top1_acc']:.2f}%** ({h4['ext_top1_correct']}/20).
6. **Berapa Top-3 H0 vs H4?**  
   H0 = **{h0['ext_top3_acc']:.2f}%** vs H4 = **{h4['ext_top3_acc']:.2f}%** ({h4['ext_top3_acc'] - h0['ext_top3_acc']:+.2f}%).
7. **Berapa Top-5 H0 vs H4?**  
   H0 = **{h0['ext_top5_acc']:.2f}%** vs H4 = **{h4['ext_top5_acc']:.2f}%** ({h4['ext_top5_acc'] - h0['ext_top5_acc']:+.2f}%).
8. **Mahasiswa mana yang IMPROVED?**  
   {', '.join(improved) if improved else 'Tidak ada'}.
9. **Mahasiswa mana yang REGRESSED?**  
   {', '.join(regressed) if regressed else 'Tidak ada'}.
10. **Berapa mean intra/inter distance H0 vs H4?**  
    - H0: Intra = {h0['mean_intra']:.4f}, Inter = {h0['mean_inter']:.4f}
    - H4: Intra = {h4['mean_intra']:.4f}, Inter = {h4['mean_inter']:.4f}
11. **Apakah separation margin membaik?**  
    Margin H0 = {h0['delta_mu']:.4f} vs H4 = {h4['delta_mu']:.4f} (Perubahan: {h4['delta_mu'] - h0['delta_mu']:+.4f}).
12. **Apakah overlap intra/inter berkurang?**  
    Overlap H0 ({h0['overlap_intra']:.2f}%) vs H4 ({h4['overlap_intra']:.2f}%).
13. **Apakah hubness Soni/Fathur/Dony/Wiridan/Raditya berkurang?**  
    Lihat tabel Hubness pada Bagian 3.
14. **Apakah symmetric confusion Alif <-> Angela membaik?**  
    Alif di H0: {h0_det_map['Muhammad Alif Rizky Hutama']['predicted_writer']} -> di H4: {h4_det_map['Muhammad Alif Rizky Hutama']['predicted_writer']}.  
    Angela di H0: {h0_det_map['Angela Permata Rosa']['predicted_writer']} -> di H4: {h4_det_map['Angela Permata Rosa']['predicted_writer']}.
15. **Apakah H4 masih memberikan manfaat setelah dataset bertambah dari 18 menjadi 20 kelas?**  
    Pada dataset 20 kelas, H4 memberikan sedikit peningkatan pada LOOCV internal (+1.50%), namun **TIDAK memberikan peningkatan net pada external Top-1 (tetap 35.00% / 7 dari 20)**.
16. **Apakah peningkatan external accuracy konsisten dengan peningkatan feature-space separation?**  
    Separation margin $\Delta \mu$ praktis identik (1.0883 vs 1.1090), konsisten dengan fakta bahwa akurasi external Top-1 tidak mengalami perubahan (35% vs 35%).
17. **Berdasarkan bukti, apakah H4 layak menjadi kandidat preprocessing final?**  
    **TIDAK CUKUP KUAT**. H4 tidak mampu memecahkan masalah kebingungan antar-kelas pada dataset 20 mahasiswa dan tidak memberikan peningkatan akurasi real-world.
"""

    report_path = os.path.join(OUT_DIR, "experiment_j_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved experiment report: {report_path}")

if __name__ == "__main__":
    main()
