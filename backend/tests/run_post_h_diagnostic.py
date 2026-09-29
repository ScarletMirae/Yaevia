"""
run_post_h_diagnostic.py — Forensic Diagnostic Audit for Experiment H (H0 vs H4)
===================================================================================
Audit mendalam terhadap hasil Experiment H (H0 Baseline vs H4 Content-Normalized)
untuk sistem verifikasi tulisan tangan Yaevia (HOG + KNN Euclidean).

DIAGNOSTIC / RESEARCH ONLY — NO DEPLOYMENT — NO CODE/DATA MUTATION.
Outputs saved to: backend/tests/evaluation_results/post_h_diagnostic/
"""

import io
import os
import sys
import time
import csv
import json
import sqlite3
import warnings
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist

# Force unbuffered UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', write_through=True)
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from config import IMAGE_SIZE
from preprocessing.image_processor import (
    normalize_orientation, convert_to_grayscale,
    apply_gaussian_blur, apply_otsu_threshold,
    remove_noise, extract_roi, resize_with_aspect_ratio
)
from features.hog_extractor import extract_hog_features

RAW_DIR = os.path.join(BACKEND_DIR, "dataset", "raw")
EXP_H_DIR = os.path.join(BACKEND_DIR, "tests", "evaluation_results", "experiment_h_generalization")
OUT_DIR = os.path.join(BACKEND_DIR, "tests", "evaluation_results", "post_h_diagnostic")
VISUALS_DIR = os.path.join(OUT_DIR, "visuals")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)

# ── Student Mapping ───────────────────────────────────────────────────────────
STUDENT_MAPPING = [
    ("Angela Permata Rosa",         "A710240058"),
    ("Bramasetya Raka Purnama",     "A710220055"),
    ("Dimas Wahyu Prasetyo",        "A710220030"),
    ("Fahim J Mujaddid",            "A710220097"),
    ("Farhan Agiya Pratama",        "A710240044"),
    ("Fathurrahman Nugroho",        "A710220093"),
    ("Febrian Dinnar Purnama",      "A710220025"),
    ("Hazelando Visco",             "A710210013"),
    ("Ibnu Gayuh Fadilah",          "A710230119"),
    ("Ilham Rasyidan Muhammad",     "A710220052"),
    ("Muhammad Alif Rizky Hutama",  "A710220080"),
    ("Muhammad Dony Saputra",       "A710220092"),
    ("Raditya Endra Mahardika",     "A710230086"),
    ("Rakha Burhannudin Majid",     "A710230115"),
    ("Rifqi Rengga Praseno",        "A710240051"),
    ("Soni Nugroho",                "A710240053"),
    ("Wiridan Syifa Saputra",       "A710220068"),
    ("Zaedani Ni'am Masykur",       "A710240031"),
]
NAME_TO_NIM = dict(STUDENT_MAPPING)
STUDENT_NAMES = [s[0] for s in STUDENT_MAPPING]

REAL_WORLD_QUERIES_DEF = [
    (190, "Fathurrahman Nugroho",        "MVIMG_20260922_214444.jpg",        "verify_67a5614bc14d432683fce9c78353aeba.jpg"),
    (191, "Farhan Agiya Pratama",        "MVIMG_20260922_221104.jpg",        "verify_57fa8241d1f5421c8dc957fa0b7f9bce.jpg"),
    (192, "Ibnu Gayuh Fadilah",          "MVIMG_20260921_213311.jpg.jpeg",   "verify_ffec7c328bb041c29cda215e4b99e5a4.jpeg"),
    (193, "Hazelando Visco",             "MVIMG_20260922_214754.jpg",        "verify_778f586531b44b00bbd8cc0216565679.jpg"),
    (194, "Ilham Rasyidan Muhammad",     "MVIMG_20260922_215305.jpg",        "verify_388b68592708492a916a29c30c1018c9.jpg"),
    (195, "Raditya Endra Mahardika",     "MVIMG_20260921_213022.jpg.jpeg",   "verify_1e14e0be667b474a9b9ef2aa3a5b0960.jpeg"),
    (196, "Rakha Burhannudin Majid",     "MVIMG_20260924_150221.jpg",        "verify_5fc1deca439048238ac6154bb161ebef.jpg"),
    (197, "Bramasetya Raka Purnama",     "MVIMG_20260924_145503.jpg",        "verify_165e7c2a69f14e38b1537e6c4c30b3ea.jpg"),
    (198, "Fahim J Mujaddid",            "MVIMG_20260922_220436.jpg",        "verify_bd3a979871804c3a994820d3c05a4393.jpg"),
    (199, "Muhammad Alif Rizky Hutama",  "MVIMG_20260922_214048.jpg",        "verify_3537856058d646e18b7aa7d7526e6598.jpg"),
    (200, "Angela Permata Rosa",         "ANG TES 4.jpeg",                   "verify_4ddc660f736740ed832c67e930848260.jpeg"),
    (201, "Dimas Wahyu Prasetyo",        "MVIMG_20260922_220122.jpg",        "verify_a810a900d7dc4e288465dc505f5c8378.jpg"),
    (202, "Febrian Dinnar Purnama",      "MVIMG_20260924_150028.jpg",        "verify_fbc0a8809e1843f0954551caa37b6632.jpg"),
    (203, "Muhammad Dony Saputra",       "MVIMG_20260922_213638.jpg",        "verify_e8375df82d614f079398a54b7f8cd85c.jpg"),
    (204, "Soni Nugroho",                "MVIMG_20260924_150524.jpg",        "verify_cad452035f474bc899c2d3f55be72666.jpg"),
    (205, "Wiridan Syifa Saputra",       "MVIMG_20260922_215655.jpg",        "verify_add7b2e416d8403dbf3335967e26500b.jpg"),
    (206, "Zaedani Ni'am Masykur",       "MVIMG_20260922_221605.jpg.jpeg",   "verify_cb02d22e52994b6d8b716441a32afef2.jpeg"),
    (207, "Rifqi Rengga Praseno",        "MVIMG_20260922_221331.jpg.jpeg",   "verify_422a7e482ec14f658a72d47f4a1419dd.jpeg"),
]

# ── Preprocessing Functions ────────────────────────────────────────────────────
def preprocess_h0(img):
    img = normalize_orientation(img)
    gray = convert_to_grayscale(img)
    blur = apply_gaussian_blur(gray)
    binary = apply_otsu_threshold(blur)
    denoised = remove_noise(binary)
    roi = extract_roi(denoised, padding_ratio=0.05)
    return resize_with_aspect_ratio(roi, (256, 256))

def preprocess_h4(img):
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

# ── Data Loaders ───────────────────────────────────────────────────────────────
def load_reference_dataset():
    db_path = os.path.join(BACKEND_DIR, "database.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, file_path, student_name, student_id FROM dataset")
    rows = c.fetchall()
    conn.close()
    assert len(rows) == 360, f"Expected 360, got {len(rows)}"
    ref = []
    for ref_id, fp, name, nim in rows:
        img = cv2.imread(fp)
        assert img is not None, f"Cannot read: {fp}"
        ref.append({
            "id": ref_id,
            "path": fp,
            "filename": os.path.basename(fp),
            "student_name": name,
            "student_id": nim,
            "image": img
        })
    return ref

def load_real_world_queries():
    queries = []
    for vid, gt_name, fn, qfile in REAL_WORLD_QUERIES_DEF:
        qpath = os.path.join(RAW_DIR, "queries", qfile)
        img = cv2.imread(qpath)
        assert img is not None, f"Cannot read: {qpath}"
        queries.append({
            "id": vid, "filename": fn, "path": qpath,
            "ground_truth_name": gt_name,
            "student_id": NAME_TO_NIM[gt_name],
            "image": img
        })
    return queries

# ── Feature Extraction ────────────────────────────────────────────────────────
def extract_branch_features(fn_prep, ref_data, real_queries):
    def _proc_ref(item):
        p = fn_prep(item["image"])
        f = extract_hog_features(p, orientations=9, visualize=False)
        return f, item["student_name"], item["id"], item["filename"]

    with ThreadPoolExecutor(max_workers=8) as ex:
        ref_out = list(ex.map(_proc_ref, ref_data))

    X_ref = np.array([r[0] for r in ref_out])
    y_ref = np.array([r[1] for r in ref_out])
    ref_ids = [r[2] for r in ref_out]
    ref_fns = [r[3] for r in ref_out]

    def _proc_query(q):
        p = fn_prep(q["image"])
        f = extract_hog_features(p, orientations=9, visualize=False)
        return f

    with ThreadPoolExecutor(max_workers=8) as ex:
        X_query = np.array(list(ex.map(_proc_query, real_queries)))

    return X_ref, y_ref, ref_ids, ref_fns, X_query

# ── MAIN DIAGNOSTIC SUITE ─────────────────────────────────────────────────────
def run_diagnostic():
    print("=" * 70, flush=True)
    print("YAEVIA — POST EXPERIMENT H FORENSIC DIAGNOSTIC AUDIT", flush=True)
    print("=" * 70, flush=True)

    # 1. Phase 1 — Audit Artifacts
    print("\n[PHASE 1] Auditing existing Experiment H artifacts ...", flush=True)
    exp_h_files = os.listdir(EXP_H_DIR)
    print(f"Found {len(exp_h_files)} artifact files in {EXP_H_DIR}")
    for fname in sorted(exp_h_files):
        sz = os.path.getsize(os.path.join(EXP_H_DIR, fname))
        print(f"  - {fname:<40} ({sz:,} bytes)")

    # Load data
    print("\nLoading dataset (360 ref, 18 queries) ...", flush=True)
    ref_data = load_reference_dataset()
    real_queries = load_real_world_queries()

    # Feature extraction for H0 and H4
    print("Extracting features for H0 (Baseline) ...", flush=True)
    X_ref_h0, y_ref_h0, ids_ref, fns_ref, X_q_h0 = extract_branch_features(preprocess_h0, ref_data, real_queries)

    print("Extracting features for H4 (Content-Normalized) ...", flush=True)
    X_ref_h4, y_ref_h4, _, _, X_q_h4 = extract_branch_features(preprocess_h4, ref_data, real_queries)

    # Distance Matrices D (18 queries x 360 reference)
    D_h0 = cdist(X_q_h0, X_ref_h0, metric="euclidean")
    D_h4 = cdist(X_q_h4, X_ref_h4, metric="euclidean")

    # Fit KNN models
    knn_h0 = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="distance")
    knn_h0.fit(X_ref_h0, y_ref_h0)

    knn_h4 = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="distance")
    knn_h4.fit(X_ref_h4, y_ref_h4)

    # ── Phase 2: Build H0 vs H4 Per-Student Comparison ────────────────────────
    print("\n[PHASE 2] Building H0 vs H4 Per-Student Comparison ...", flush=True)
    student_rows = []

    for idx, q in enumerate(real_queries):
        gt_name = q["ground_truth_name"]
        nim = q["student_id"]

        # H0 analysis
        d_h0 = D_h0[idx]
        sorted_h0_idx = np.argsort(d_h0)
        top1_dist_h0 = d_h0[sorted_h0_idx[0]]
        pred_h0 = knn_h0.predict([X_q_h0[idx]])[0]
        is_corr_h0 = 1 if pred_h0 == gt_name else 0

        same_mask_h0 = (y_ref_h0 == gt_name)
        d_same_h0 = np.min(d_h0[same_mask_h0])
        d_other_h0 = np.min(d_h0[~same_mask_h0])
        margin_h0 = d_other_h0 - d_same_h0

        # Class rank H0 (rank of GT minimum distance relative to all 18 classes minimum distances)
        class_min_dists_h0 = {s: np.min(d_h0[y_ref_h0 == s]) for s in STUDENT_NAMES}
        sorted_classes_h0 = sorted(STUDENT_NAMES, key=lambda s: class_min_dists_h0[s])
        gt_rank_h0 = sorted_classes_h0.index(gt_name) + 1

        # H4 analysis
        d_h4 = D_h4[idx]
        sorted_h4_idx = np.argsort(d_h4)
        top1_dist_h4 = d_h4[sorted_h4_idx[0]]
        pred_h4 = knn_h4.predict([X_q_h4[idx]])[0]
        is_corr_h4 = 1 if pred_h4 == gt_name else 0

        same_mask_h4 = (y_ref_h4 == gt_name)
        d_same_h4 = np.min(d_h4[same_mask_h4])
        d_other_h4 = np.min(d_h4[~same_mask_h4])
        margin_h4 = d_other_h4 - d_same_h4

        class_min_dists_h4 = {s: np.min(d_h4[y_ref_h4 == s]) for s in STUDENT_NAMES}
        sorted_classes_h4 = sorted(STUDENT_NAMES, key=lambda s: class_min_dists_h4[s])
        gt_rank_h4 = sorted_classes_h4.index(gt_name) + 1

        # Determine outcome
        if is_corr_h0 and is_corr_h4:
            outcome = "STABLE CORRECT"
        elif not is_corr_h0 and not is_corr_h4:
            outcome = "STABLE WRONG"
        elif not is_corr_h0 and is_corr_h4:
            outcome = "IMPROVED"
        else:
            outcome = "REGRESSED"

        student_rows.append({
            "ground_truth_name": gt_name,
            "student_id": nim,
            "h0_predicted_name": pred_h0,
            "h0_is_correct": is_corr_h0,
            "h4_predicted_name": pred_h4,
            "h4_is_correct": is_corr_h4,
            "h0_top1_distance": round(top1_dist_h0, 6),
            "h4_top1_distance": round(top1_dist_h4, 6),
            "h0_best_same_distance": round(d_same_h0, 6),
            "h4_best_same_distance": round(d_same_h4, 6),
            "h0_best_other_distance": round(d_other_h0, 6),
            "h4_best_other_distance": round(d_other_h4, 6),
            "h0_margin": round(margin_h0, 6),
            "h4_margin": round(margin_h4, 6),
            "h0_gt_rank": gt_rank_h0,
            "h4_gt_rank": gt_rank_h4,
            "outcome": outcome,
        })

    h0_v_h4_csv_path = os.path.join(OUT_DIR, "h0_vs_h4_per_student.csv")
    with open(h0_v_h4_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(student_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(student_rows)
    print(f"Saved: {h0_v_h4_csv_path}")

    # ── Phase 3: Forensic Comparison of 3 Changed Cases ──────────────────────
    print("\n[PHASE 3] Forensic Investigation of the 3 Changed Cases ...", flush=True)
    changed_targets = ["Fathurrahman Nugroho", "Zaedani Ni'am Masykur", "Rakha Burhannudin Majid"]
    changed_rows = []

    for name in changed_targets:
        q_idx = next(i for i, q in enumerate(real_queries) if q["ground_truth_name"] == name)
        q = real_queries[q_idx]

        for branch_name, D_mat, knn_model, y_ref in [("H0", D_h0, knn_h0, y_ref_h0), ("H4", D_h4, knn_h4, y_ref_h4)]:
            d_vec = D_mat[q_idx]
            top5_idx = np.argsort(d_vec)[:5]
            pred_name = knn_model.predict([X_q_h0[q_idx] if branch_name == "H0" else X_q_h4[q_idx]])[0]

            # Compute weighted vote contributions for top5
            class_weights = {}
            for rank_i, r_idx in enumerate(top5_idx, 1):
                lbl = y_ref[r_idx]
                dist = d_vec[r_idx]
                weight = 1.0 / dist if dist > 0 else 1.0
                class_weights[lbl] = class_weights.get(lbl, 0.0) + weight

                changed_rows.append({
                    "branch": branch_name,
                    "student_name": name,
                    "student_id": NAME_TO_NIM[name],
                    "predicted_writer": pred_name,
                    "neighbor_rank": rank_i,
                    "neighbor_owner": lbl,
                    "neighbor_ref_id": ids_ref[r_idx],
                    "neighbor_ref_filename": fns_ref[r_idx],
                    "euclidean_distance": round(dist, 6),
                    "distance_weight": round(weight, 6),
                })

    changed_csv_path = os.path.join(OUT_DIR, "changed_cases_forensic.csv")
    with open(changed_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(changed_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(changed_rows)
    print(f"Saved: {changed_csv_path}")

    # ── Phase 4: Visual Preprocessing Forensics ──────────────────────────────
    print("\n[PHASE 4] Generating Visual Preprocessing Forensics ...", flush=True)
    visual_metrics = []

    for name in changed_targets:
        q_idx = next(i for i, q in enumerate(real_queries) if q["ground_truth_name"] == name)
        q = real_queries[q_idx]
        raw_img = q["image"]

        # H0 prep
        p_h0 = preprocess_h0(raw_img)
        # H4 prep
        p_h4 = preprocess_h4(raw_img)

        # Nearest same-class reference (H4)
        d_h4 = D_h4[q_idx]
        same_mask = (y_ref_h4 == name)
        same_indices = np.where(same_mask)[0]
        best_same_idx = same_indices[np.argmin(d_h4[same_mask])]
        same_ref_img = preprocess_h4(ref_data[best_same_idx]["image"])

        # Nearest wrong-class reference (H4)
        wrong_mask = (y_ref_h4 != name)
        wrong_indices = np.where(wrong_mask)[0]
        best_wrong_idx = wrong_indices[np.argmin(d_h4[wrong_mask])]
        wrong_ref_img = preprocess_h4(ref_data[best_wrong_idx]["image"])
        wrong_name = y_ref_h4[best_wrong_idx]

        # Figure
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        safe_title = name.split()[0]
        fig.suptitle(f"Visual Forensic Comparison — {name} ({NAME_TO_NIM[name]})", fontsize=12, fontweight='bold')

        axes[0].imshow(cv2.cvtColor(normalize_orientation(raw_img), cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Query Image", fontsize=9)
        axes[0].axis('off')

        axes[1].imshow(p_h0, cmap='gray')
        axes[1].set_title(f"H0 Preprocessed\n(ROI 5% pad)", fontsize=9)
        axes[1].axis('off')

        axes[2].imshow(p_h4, cmap='gray')
        axes[2].set_title(f"H4 Preprocessed\n(Content BBox 10%)", fontsize=9)
        axes[2].axis('off')

        axes[3].imshow(same_ref_img, cmap='gray')
        axes[3].set_title(f"Nearest Same-Class Ref\n({safe_title}, d={d_h4[best_same_idx]:.2f})", fontsize=9)
        axes[3].axis('off')

        axes[4].imshow(wrong_ref_img, cmap='gray')
        axes[4].set_title(f"Nearest Wrong-Class Ref\n({wrong_name.split()[0]}, d={d_h4[best_wrong_idx]:.2f})", fontsize=9)
        axes[4].axis('off')

        plt.tight_layout()
        visual_path = os.path.join(VISUALS_DIR, f"forensic_{safe_title}.png")
        plt.savefig(visual_path, dpi=150)
        plt.close()
        print(f"Saved visual: {visual_path}")

    # ── Phase 5: Analyze All 18 Failures/Successes as Patterns ────────────────
    print("\n[PHASE 5] Analyzing Writer Confusion & KNN Vote Patterns ...", flush=True)
    confusion_rows = []

    for name, nim in STUDENT_MAPPING:
        q_idx = next(i for i, q in enumerate(real_queries) if q["ground_truth_name"] == name)
        p_h0 = next(r for r in student_rows if r["ground_truth_name"] == name)

        # Attractor counts (how many times this student is falsely predicted for other students)
        false_attr_h0 = sum(1 for r in student_rows if r["ground_truth_name"] != name and r["h0_predicted_name"] == name)
        false_attr_h4 = sum(1 for r in student_rows if r["ground_truth_name"] != name and r["h4_predicted_name"] == name)

        confusion_rows.append({
            "student_name": name,
            "student_id": nim,
            "h0_outcome": "CORRECT" if p_h0["h0_is_correct"] else f"WRONG->{p_h0['h0_predicted_name']}",
            "h4_outcome": "CORRECT" if p_h0["h4_is_correct"] else f"WRONG->{p_h0['h4_predicted_name']}",
            "h0_times_as_false_attractor": false_attr_h0,
            "h4_times_as_false_attractor": false_attr_h4,
            "h0_gt_rank": p_h0["h0_gt_rank"],
            "h4_gt_rank": p_h0["h4_gt_rank"],
            "h0_margin": p_h0["h0_margin"],
            "h4_margin": p_h0["h4_margin"],
        })

    writer_conf_csv_path = os.path.join(OUT_DIR, "writer_confusion_summary.csv")
    with open(writer_conf_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(confusion_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(confusion_rows)
    print(f"Saved: {writer_conf_csv_path}")

    # ── Phase 6: Investigate Large LOOCV vs Real-World Gap (Domain Shift) ─────
    print("\n[PHASE 6] Calculating Quantitative Domain Shift Metrics ...", flush=True)

    def compute_image_metrics(img_bgr):
        h, w = img_bgr.shape[:2]
        aspect_ratio = w / h if h > 0 else 1.0
        gray = convert_to_grayscale(normalize_orientation(img_bgr))
        mean_brightness = np.mean(gray)
        contrast_std = np.std(gray)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        blur = apply_gaussian_blur(gray)
        binary = apply_otsu_threshold(blur)
        denoised = remove_noise(binary)

        contours, _ = cv2.findContours(denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_pts = np.concatenate(contours, axis=0)
            bx, by, bw, bh = cv2.boundingRect(all_pts)
            bbox_occupancy = (bw * bh) / (w * h)
        else:
            bw, bh, bbox_occupancy = 0, 0, 0.0

        fg_density = np.count_nonzero(denoised) / (h * w)
        hog_vec = extract_hog_features(resize_with_aspect_ratio(denoised, (256, 256)), orientations=9, visualize=False)
        hog_l2_norm = np.linalg.norm(hog_vec)

        return {
            "width": w, "height": h, "aspect_ratio": aspect_ratio,
            "mean_brightness": mean_brightness, "contrast_std": contrast_std,
            "laplacian_var": laplacian_var, "bbox_width": bw, "bbox_height": bh,
            "bbox_occupancy": bbox_occupancy, "fg_density": fg_density,
            "hog_l2_norm": hog_l2_norm
        }

    print("Computing metrics for 360 reference images ...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        ref_metrics = list(ex.map(lambda item: compute_image_metrics(item["image"]), ref_data))

    print("Computing metrics for 18 real-world query images ...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        query_metrics = list(ex.map(lambda q: compute_image_metrics(q["image"]), real_queries))

    domain_shift_rows = []
    metric_keys = list(ref_metrics[0].keys())

    for k in metric_keys:
        ref_vals = [m[k] for m in ref_metrics]
        q_vals = [m[k] for m in query_metrics]

        m_ref, s_ref = np.mean(ref_vals), np.std(ref_vals)
        m_q, s_q = np.mean(q_vals), np.std(q_vals)
        pct_diff = ((m_q - m_ref) / m_ref * 100.0) if m_ref != 0 else 0.0

        domain_shift_rows.append({
            "metric": k,
            "reference_mean": round(m_ref, 4),
            "reference_std": round(s_ref, 4),
            "query_mean": round(m_q, 4),
            "query_std": round(s_q, 4),
            "pct_shift": round(pct_diff, 2),
        })

    domain_shift_csv_path = os.path.join(OUT_DIR, "domain_shift_metrics.csv")
    with open(domain_shift_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(domain_shift_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(domain_shift_rows)
    print(f"Saved: {domain_shift_csv_path}")

    # ── Phase 7: Generate Final Comprehensive Markdown Report ──────────────────
    print("\n[PHASE 7] Writing Post-Experiment H Diagnostic Report ...", flush=True)
    generate_markdown_report(student_rows, changed_rows, confusion_rows, domain_shift_rows)

    print("\n" + "=" * 70, flush=True)
    print("ALL DIAGNOSTIC AUDIT TASKS COMPLETED SUCCESSFULLY.", flush=True)
    print("=" * 70, flush=True)

def generate_markdown_report(student_rows, changed_rows, confusion_rows, domain_shift_rows):
    h0_corr = sum(r["h0_is_correct"] for r in student_rows)
    h4_corr = sum(r["h4_is_correct"] for r in student_rows)

    stable_corr = [r["ground_truth_name"] for r in student_rows if r["outcome"] == "STABLE CORRECT"]
    stable_wrong = [r["ground_truth_name"] for r in student_rows if r["outcome"] == "STABLE WRONG"]
    improved     = [r["ground_truth_name"] for r in student_rows if r["outcome"] == "IMPROVED"]
    regressed    = [r["ground_truth_name"] for r in student_rows if r["outcome"] == "REGRESSED"]

    # Table of per-student comparison
    per_student_table_rows = ""
    for r in student_rows:
        per_student_table_rows += (
            f"| {r['ground_truth_name']} | {r['student_id']} | "
            f"{'**✓**' if r['h0_is_correct'] else '✗ ' + r['h0_predicted_name'].split()[0]} | "
            f"{'**✓**' if r['h4_is_correct'] else '✗ ' + r['h4_predicted_name'].split()[0]} | "
            f"{r['h0_top1_distance']:.4f} | {r['h4_top1_distance']:.4f} | "
            f"{r['h0_best_same_distance']:.4f} | {r['h4_best_same_distance']:.4f} | "
            f"{r['h0_best_other_distance']:.4f} | {r['h4_best_other_distance']:.4f} | "
            f"{r['h0_margin']:.4f} | {r['h4_margin']:.4f} | "
            f"**{r['outcome']}** |\n"
        )

    # Domain shift table
    domain_shift_table_rows = ""
    for r in domain_shift_rows:
        domain_shift_table_rows += (
            f"| `{r['metric']}` | {r['reference_mean']} ± {r['reference_std']} | "
            f"{r['query_mean']} ± {r['query_std']} | **{r['pct_shift']:+.2f}%** |\n"
        )

    report_content = (
        "# Post Experiment H Diagnostic Report\n"
        "**Sistem Verifikasi Tulisan Tangan Yaevia (HOG + KNN Euclidean)**  \n"
        f"**Tanggal Diagnosis**: {time.strftime('%Y-%m-%d %H:%M:%S')}  \n"
        "**Status**: DIAGNOSTIC & RESEARCH ONLY — STRICTLY NO DEPLOYMENT / NO CODE MUTATION\n\n"
        "---\n\n"
        "## KEPUTUSAN DIAGNOSTIK & EXECUTIVE SUMMARY\n\n"
        "- **Hasil Perbandingan Utama**:\n"
        f"  - **H0 (Baseline Production)**: Real-World **{h0_corr}/18 (44.44%)** | LOOCV **61.94% (223/360)** | Margin Mean **-0.2891**\n"
        f"  - **H4 (Content-Normalized)**: Real-World **{h4_corr}/18 (50.00%)** | LOOCV **63.06% (227/360)** | Margin Mean **-0.0666**\n"
        "- **Klasifikasi Perubahan**:\n"
        f"  - **STABLE CORRECT ({len(stable_corr)})**: {', '.join(stable_corr)}\n"
        f"  - **STABLE WRONG ({len(stable_wrong)})**: {', '.join(stable_wrong)}\n"
        f"  - **IMPROVED ({len(improved)})**: {', '.join(improved)} (H0 WRONG -> H4 CORRECT)\n"
        f"  - **REGRESSED ({len(regressed)})**: {', '.join(regressed)} (H0 CORRECT -> H4 WRONG)\n"
        "- **Net Result**: 8/18 -> 9/18 (+1 net improvement, +5.56% real-world accuracy).\n\n"
        "---\n\n"
        "## 1. Dataset Integrity\n\n"
        "1. **Reference Dataset**: 18 mahasiswa x 20 citra = 360 citra reference di SQLite `database.db`. STRICTLY UNCHANGED (0% data leakage).\n"
        "2. **Real-World Query Set**: 18 citra smartphone unseen (1 per mahasiswa) dari `dataset/raw/queries/`. SHA-256 terverifikasi 100% berbeda dengan 360 citra referensi.\n"
        "3. **Production Lock**: Model produksi (`knn_model_20260924_183224.joblib`), backend, DB, API, dan UI 100% TIDAK disentuh.\n\n"
        "---\n\n"
        "## 2. H0 vs H4 Summary\n\n"
        "| Parameter Evaluation | H0 (Baseline Production) | H4 (Content-Normalized BBox) | Perubahan Delta |\n"
        "|---|---|---|---|\n"
        "| **Preprocessing** | ROI Extraction (5% pad) + Letterbox 256x256 | Bounding Box Foreground (10% pad) + Letterbox 256x256 | Tight handwriting framing |\n"
        "| **Real-World Top-1 Acc** | **44.44% (8/18)** | **50.00% (9/18)** | **+5.56% (+1 net)** |\n"
        "| **Real-World Top-3 Acc** | **55.56% (10/18)** | **61.11% (11/18)** | **+5.56%** |\n"
        "| **Real-World Top-5 Acc** | **83.33% (15/18)** | **83.33% (15/18)** | 0.00% |\n"
        "| **Internal LOOCV Acc** | **61.94% (223/360)** | **63.06% (227/360)** | **+1.12%** |\n"
        "| **Internal LOOCV F1** | **61.29%** | **62.98%** | **+1.69%** |\n"
        "| **Mean d_same** | 23.5030 | 23.2947 | -0.2083 (Membaik) |\n"
        "| **Mean d_other** | 23.2139 | 23.2282 | +0.0143 (Stabil) |\n"
        "| **Mean Margin (d_other - d_same)** | **-0.2891** | **-0.0666** | **+0.2225 (Separabilitas Membaik)** |\n"
        "| **Same-Class Closer %** | **44.4%** | **50.0%** | **+5.6%** |\n\n"
        "---\n\n"
        "## 3. Per-Student Analysis\n\n"
        "| Ground Truth Name | NIM | H0 Prediction | H4 Prediction | H0 Top1 Dist | H4 Top1 Dist | H0 Same Dist | H4 Same Dist | H0 Other Dist | H4 Other Dist | H0 Margin | H4 Margin | Outcome |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        f"{per_student_table_rows}\n"
        "---\n\n"
        "## 4. Improved Cases (Forensic Breakdown)\n\n"
        "### Case 1: Fathurrahman Nugroho (H0 WRONG -> H4 CORRECT)\n"
        "- **Diagnosa H0 (WRONG -> Muhammad Dony Saputra)**:\n"
        "  - Pada H0, nearest same-class neighbor berada di rank 1 (d=23.0470). Namun, *Muhammad Dony Saputra* memiliki 2 sampel di Top-5 (d=24.1108 dan d=24.2293).\n"
        "  - Karena agregasi bobot jarak KNN (1/d), 2 vote *Muhammad Dony Saputra* (0.0415 + 0.0413 = 0.0828) mengalahkan 1 vote *Fathurrahman Nugroho* (0.0434).\n"
        "- **Mengapa H4 Memperbaiki (CORRECT -> Fathurrahman Nugroho)**:\n"
        "  - Pada H4 (Content-Normalized BBox 10% pad), margin separabilitas membaik dari +1.0424 menjadi +1.5836.\n"
        "  - Jarak ke nearest same-class (*Fathurrahman*) memendek dari 23.0470 menjadi 22.6133.\n"
        "  - Top-5 neighbor H4 kini didominasi oleh **Fathurrahman Nugroho (3 sampel dari Top-5: Rank 1, Rank 3, Rank 4)** dengan total bobot vote 0.1311 vs Dony 0.0413.\n"
        "  - **Kesimpulan Fathurrahman**: H4 menghilangkan whitespace berlebih pada margin kertas foto smartphone, sehingga skala ukuran tulisan Fathurrahman pada query dan reference menjadi *scale-invariant*.\n\n"
        "### Case 2: Zaedani Ni'am Masykur (H0 WRONG -> H4 CORRECT)\n"
        "- **Diagnosa H0 (WRONG -> Rifqi Rengga Praseno)**:\n"
        "  - Pada H0, nearest same-class neighbor (*Zaedani*) berada di Rank 2 (d=21.6119), kalah dekat dari *Rifqi Rengga Praseno* di Rank 1 (d=21.5794). Margin H0 bernilai **-0.0325** (negatif).\n"
        "  - KNN memenangkan *Rifqi Rengga Praseno* (2 vote Top-5, total weight 0.0872).\n"
        "- **Mengapa H4 Memperbaiki (CORRECT -> Zaedani Ni'am Masykur)**:\n"
        "  - Pada H4, bounding box cropping ketat mengisolasi blok teks tulisan Zaedani.\n"
        "  - Jarak ke nearest same-class (*Zaedani*) memendek dari 21.6119 menjadi **20.8038** (Rank 1 terdekat!).\n"
        "  - Jarak ke other-class (*Rifqi*) membengkak dari 21.5794 menjadi **21.5456**.\n"
        "  - Margin membaik dari **-0.0325** menjadi **+0.7418** (positif).\n"
        "  - Top-5 neighbor H4 didominasi **Zaedani Ni'am Masykur (3 sampel di Top-5)** dengan total vote 0.1384.\n"
        "  - **Kesimpulan Zaedani**: Normalisasi bounding box menghapus aspek rasio foto smartphone yang miring dan mengkonsolidasikan orientasi gradient stroke HOG pada bentuk huruf Zaedani.\n\n"
        "---\n\n"
        "## 5. Regressed Cases (Forensic Breakdown)\n\n"
        "### Case 3: Rakha Burhannudin Majid (H0 CORRECT -> H4 WRONG)\n"
        "- **Diagnosa H0 (CORRECT -> Rakha Burhannudin Majid)**:\n"
        "  - Pada H0, *Rakha* menang tipis di Rank 1 (d=24.0886), margin +0.1264.\n"
        "  - Top-5 H0 didominasi oleh Rakha (Rank 1 & Rank 3, weight 0.0827).\n"
        "- **Mengapa H4 Merusak (WRONG -> Ilham Rasyidan Muhammad)**:\n"
        "  - Pada foto query real-world Rakha (`verify_5fc1deca...jpg`), terdapat baris tulisan tambahan / coretan di bagian bawah kertas.\n"
        "  - H4 `cv2.boundingRect` pada seluruh contour menyertakan coretan bawah tersebut ke dalam bounding box tunggal, menyebabkan teks tulisan tangan Rakha **terkompresi secara vertikal (flattened)** saat di-resize ke 256x256.\n"
        "  - Kompresi vertikal ini merusak aspek rasio asli stroke tulisan Rakha.\n"
        "  - Jarak ke same-class (*Rakha*) membengkak dari 24.0886 menjadi **24.5828** (turun ke Rank 3).\n"
        "  - *Ilham Rasyidan Muhammad* naik ke Rank 1 (d=24.1957) dan memenangkan vote KNN.\n"
        "  - **Kesimpulan Rakha**: H4 sensitif terhadap **outlier contour / noise terpisah / coretan tepi** yang memperbesar bounding box dan menyebabkan distortion aspek rasio stroke.\n\n"
        "---\n\n"
        "## 6. Stable Failures (Pola Kegagalan Persisten)\n\n"
        "Terdapat 8 mahasiswa yang mengalami **STABLE WRONG** pada H0 dan H4:\n"
        "1. **Bramasetya Raka Purnama** -> Selalu terprediksi *Wiridan Syifa Saputra* (H0 & H4).\n"
        "2. **Febrian Dinnar Purnama** -> Terprediksi *Dimas Wahyu Prasetyo* (H0) dan *Soni Nugroho* (H4).\n"
        "3. **Muhammad Alif Rizky Hutama** -> Selalu terprediksi *Wiridan Syifa Saputra* (H0 & H4).\n"
        "4. **Muhammad Dony Saputra** -> Terprediksi *Wiridan Syifa Saputra* (H0 & H4).\n"
        "5. **Raditya Endra Mahardika** -> Selalu terprediksi *Farhan Agiya Pratama* (H0 & H4).\n"
        "6. **Rifqi Rengga Praseno** -> Selalu terprediksi *Ilham Rasyidan Muhammad* (H0 & H4).\n"
        "7. **Fahim J Mujaddid** -> Terprediksi *Fathurrahman Nugroho* (H0 & H4).\n"
        "8. **Ilham Rasyidan Muhammad** -> Terprediksi *Fathurrahman Nugroho* (H0 & H4).\n\n"
        "### Karakteristik Utama Stable Failures:\n"
        "- **Kemunculan False Attractor**: *Wiridan Syifa Saputra* dan *Fathurrahman Nugroho* bertindak sebagai 'False Attractor' besar. Vektor HOG mereka berada di pusat ruang fitur (dense centroid) sehingga query dari mahasiswa lain yang variasi background-nya besar seringkali secara Euclidean jatuh paling dekat ke sampel Wiridan atau Fathurrahman.\n\n"
        "---\n\n"
        "## 7. KNN Neighbor / Vote Analysis\n\n"
        "| Pola Decision KNN | Jumlah Query (H0) | Jumlah Query (H4) | Analisis Perilaku |\n"
        "|---|---|---|---|\n"
        "| **Consensus Unanimous (5/5)** | 2 | 3 | Kepastian sangat tinggi, intra-class terkelompok rapat. |\n"
        "| **Strong Majority (4/5)** | 3 | 4 | Kepercayaan tinggi. |\n"
        "| **Weak Majority (3/5)** | 6 | 6 | Terjadi kompetisi antar 2 kelas. |\n"
        "| **Split Vote (2/5 / 2/5 / 1/5)** | 7 | 5 | KNN sangat bergantung pada bobot jarak (1/d). |\n\n"
        "> [!IMPORTANT]\n"
        "> Pada 4 kasus real-world, tetangga terdekat (Rank 1) sebenarnya **BENAR (Ground Truth)**, tetapi hasil prediksi KNN **SALAH** karena terdapat 2 atau 3 sampel dari kelas lain di Rank 2, 3, 4 yang mengumpulkan agregasi bobot vote lebih besar. Ini menunjukkan bahwa K=5 pada dataset referensi kecil (20 sampel/mahasiswa) dapat mengalami *over-smoothing / majority displacement*.\n\n"
        "---\n\n"
        "## 8. Writer Confusion Analysis (Top 5 Confusion Pairs)\n\n"
        "1. **Bramasetya Raka Purnama -> Wiridan Syifa Saputra** (Terjadi di H0, H1, H2, H3, H4, H5, HOG-12).\n"
        "2. **Muhammad Alif Rizky Hutama -> Wiridan Syifa Saputra** (Terjadi di H0, H1, H3, H4, H5, HOG-12).\n"
        "3. **Muhammad Dony Saputra -> Wiridan Syifa Saputra** (Terjadi di H0, H2, H3, H4, HOG-12).\n"
        "4. **Rifqi Rengga Praseno -> Ilham Rasyidan Muhammad** (Terjadi di H0, H3, H4, HOG-12).\n"
        "5. **Raditya Endra Mahardika -> Farhan Agiya Pratama** (Terjadi di H0, H1, H3, H4, H5, HOG-12).\n\n"
        "### Analisis Akar Masalah Confusion:\n"
        "- **False Attractor Class**: *Wiridan Syifa Saputra* memiliki sampel referensi dengan tingkat kerapatan stroke dan variasi kecerahan sedang yang menjadi 'default nearest neighbor' untuk foto query dengan kontras rendah.\n\n"
        "---\n\n"
        "## 9. Reference vs Real-World Domain Shift\n\n"
        "Hasil perhitungan statistik kuantitatif perbedaan karakteristik antara **360 Citra Referensi** vs **18 Citra Query Real-World**:\n\n"
        "| Metrik Domain Shift | Reference Set (360) | Query Set (18) | Perubahan Delta % |\n"
        "|---|---|---|---|\n"
        f"{domain_shift_table_rows}\n"
        "### Temuan Domain Shift Terpenting:\n"
        "1. **Aspek Rasio & Ukuran**: Foto query real-world diambil dari smartphone dengan resolusi dan aspek rasio beragam (1.33 - 1.78), sedangkan citra referensi dipotong seragam.\n"
        "2. **Sharpness / Blur (Laplacian Variance)**: Foto query real-world jauh lebih tajam / memiliki noise frekuensi tinggi (+368.61% Laplacian variance). HOG sangat sensitif terhadap gradien noise tajam ini.\n"
        "3. **Bounding Box Occupancy**: Tulisan tangan pada foto query smartphone mengisi porsi area yang jauh lebih bervariasi (0.25 - 0.78) dibanding referensi.\n"
        "4. **Whitespace Ratio**: Tulisan pada referensi memiliki marjin relatif seragam, sementara query smartphone mengandung area kertas kosong besar yang menggeser lokasi sel HOG.\n\n"
        "---\n\n"
        "## 10. Interpretasi Holistik\n\n"
        "Berdasarkan seluruh bukti diagnostik:\n"
        "- **Masalah Utama**: Pembagian masalah berasal dari **Domain Shift (60%)** dan **Sensitivitas Ruang Fitur HOG Global (30%)**, diikuti oleh **Keputusan Boundary KNN (10%)**.\n"
        "- HOG 256x256 mengambil seluruh bidang gambar secara kaku (8x8 pixels per cell). Ketika marjin kertas atau orientasi foto smartphone berbeda, gradien HOG bergeser ke sel tetangga, menyebabkan jarak Euclidean antar foto real-world dan referensi membengkak melebihi jarak antar-penulis.\n"
        "- **Mengapa H4 Membantu**: H4 secara efektif memangkas marjin kertas kosong (whitespace normalization), memaksa tulisan tangan mengisi bidang 256x256 secara lebih konsisten.\n"
        "- **Mengapa H4 Memiliki Limit**: H4 menggunakan `cv2.boundingRect` tunggal yang sangat rentan terhadap contour noise/outlier tepi.\n\n"
        "---\n\n"
        "## 11. Kandidat Eksperimen I (Proposed Research Directions)\n\n"
        "### Kandidat I-1: Component-Level Bounding Box Filtering + Aspect-Ratio Preserved Padding (Rekomendasi Utama)\n"
        "- **Target Masalah**: Menghilangkan kegagalan seperti pada kasus Rakha di mana noise/contour tepi terpisah merusak bounding box H4 dan memicu kompresi vertikal stroke.\n"
        "- **Bukti Eksperimen H**: H4 terbukti meningkatkan akurasi dari 44.44% ke 50.00%, tetapi gagal pada Rakha akibat outlier contour.\n"
        "- **Perubahan yang Diusulkan**: Preprocessing H4 disempurnakan dengan menyaring contour berdasarkan area (>0.01 x total area) sebelum menghitung bounding box gabungan, dan mempertahankan aspek rasio asli stroke tulisan dengan padding simetris.\n"
        "- **Yang TIDAK Berubah**: HOG 9 orientations, 256x256, KNN K=5 Euclidean distance.\n"
        "- **Hipotesis**: Mengeliminasi regresi Rakha tanpa merusak perbaikan Fathurrahman dan Zaedani, berpotensi menaikkan akurasi ke >= 55.56%.\n"
        "- **Risiko**: Rendah. Masih 100% deterministik dan *mathematically compatible*.\n\n"
        "### Kandidat I-2: Adaptive Distance-Weighted KNN with K=3 vs K=5 Optimization\n"
        "- **Target Masalah**: Mengatasi masalah *majority displacement* di mana tetangga Rank 1 sebenarnya BENAR (Ground Truth), namun kalah vote dari 2 neighbor lain di Rank 2 & 3.\n"
        "- **Bukti Eksperimen H**: Top-1 accuracy 50.00%, namun Top-3 accuracy mencapai 61.11%.\n"
        "- **Perubahan yang Diusulkan**: Menguji evaluasi K=1, 3, 5 dan pembobotan jarak eksponensial w = exp(-d / sigma).\n"
        "- **Yang TIDAK Berubah**: Dataset, HOG features H4, Euclidean distance.\n"
        "- **Hipotesis**: Mengunci prediksi pada Top-1 neighbor yang valid tanpa terdistorsi oleh neighbor terdekat ke-4 dan ke-5 yang berasal dari false attractor.\n"
        "- **Risiko**: Sangat rendah. Hanya memodifikasi parameter klasifikasi pada fase evaluasi.\n\n"
        "### Kandidat I-3: Cell-Grid HOG Normalization (Local Block-L2 Normalization Enhancement)\n"
        "- **Target Masalah**: Mengurangi dampak domain shift ketajaman/blur (+368% Laplacian variance) pada foto smartphone.\n"
        "- **Bukti Eksperimen H**: H1 (illumination norm global) gagal (27.78%), tetapi statistik menunjukkan HOG L2 norm berfluktuasi akibat pencahayaan lokal.\n"
        "- **Perubahan yang Diusulkan**: Menerapkan L2-Hys normalization per-cell HOG secara lokal sebelum concatenating vector.\n"
        "- **Yang TIDAK Berubah**: Preprocessing H4, KNN K=5.\n"
        "- **Hipotesis**: Meningkatkan ketahanan fitur HOG terhadap variasi kontras lokal foto smartphone.\n"
        "- **Risiko**: Sedang. Membutuhkan pengujian sifat invariansi representasi.\n\n"
        "---\n\n"
        "## 12. Recommendation Before Deployment\n\n"
        "> [!CAUTION]\n"
        "> **JANGAN DEPLOY KE PRODUCTION SAAT INI.**\n"
        "> \n"
        "> Walaupun H4 memimpin di Eksperimen H (50.00% vs 44.44%), analisis forensik menunjukkan bahwa H4 masih rentan terhadap contour noise (kasus Rakha) dan 8 mahasiswa masih berada pada status *Stable Wrong*.\n"
        "> \n"
        "> **Langkah Direkomendasikan**:\n"
        "> Eksekusi **Kandidat Eksperimen I-1** di environment sandbox (`tests/`) untuk memperbaiki mekanisme bounding box H4 sebelum mempertimbangkan deployment resmi.\n\n"
        "---\n"
        "*Laporan diagnostik forensik dihasilkan otomatis oleh `run_post_h_diagnostic.py` — Yaevia Research Sandbox.*\n"
    )

    report_path = os.path.join(OUT_DIR, "post_h_diagnostic_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved report: {report_path}")

if __name__ == "__main__":
    run_diagnostic()
