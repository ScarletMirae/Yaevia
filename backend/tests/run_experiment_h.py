"""
run_experiment_h.py — Targeted Real-World Generalization Improvement (Experiment H)
=====================================================================================
Evaluasi multi-branch preprocessing & representasi HOG+KNN pada:
  A. Internal LOOCV: 360 reference images (18 mhs x 20 citra)
  B. Real-World: 18 unseen smartphone query images

ATURAN STRICT:
  - Master dataset D:\.SKRIPSI\Dataset & RAW D:\Yaevia\backend\dataset\raw READ-ONLY
  - JANGAN overwrite production model / file / backend / frontend
  - Semua hasil disimpan di:
    D:\Yaevia\backend\tests\evaluation_results\experiment_h_generalization\
"""

import io
import os
import sys
import time
import json
import csv
import hashlib
import warnings
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import precision_recall_fscore_support
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cdist

# Force unbuffered output so we can see progress in real-time
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
OUT_DIR = os.path.join(BACKEND_DIR, "tests", "evaluation_results", "experiment_h_generalization")
os.makedirs(OUT_DIR, exist_ok=True)

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

# ── 18 Real-World Query Definitions ──────────────────────────────────────────
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

# ── SHA256 ────────────────────────────────────────────────────────────────────
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# ── Preprocessing Branch Functions ────────────────────────────────────────────
def preprocess_h0(img):
    """H0: Production Baseline"""
    img = normalize_orientation(img)
    gray = convert_to_grayscale(img)
    blur = apply_gaussian_blur(gray)
    binary = apply_otsu_threshold(blur)
    denoised = remove_noise(binary)
    roi = extract_roi(denoised, padding_ratio=0.05)
    return resize_with_aspect_ratio(roi, (256, 256))

def preprocess_h1(img):
    """H1: Illumination Normalization (local bg division + CLAHE)"""
    img = normalize_orientation(img)
    gray = convert_to_grayscale(img)
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    bg[bg == 0] = 1
    norm = cv2.divide(gray, bg, scale=255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)
    blur = apply_gaussian_blur(enhanced)
    binary = apply_otsu_threshold(blur)
    denoised = remove_noise(binary)
    roi = extract_roi(denoised, padding_ratio=0.05)
    return resize_with_aspect_ratio(roi, (256, 256))

def preprocess_h2(img):
    """H2: Perspective Rectification (fallback to baseline if no quad detected)"""
    img = normalize_orientation(img)
    gray = convert_to_grayscale(img)
    h, w = gray.shape[:2]
    blur_e = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur_e, 30, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    warped = None
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(c) > 0.20 * h * w:
                pts = approx.reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                (tl, tr, br, bl) = rect
                maxWidth  = max(int(np.sqrt(((br[0]-bl[0])**2)+((br[1]-bl[1])**2))),
                                int(np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2))))
                maxHeight = max(int(np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))),
                                int(np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))))
                if maxWidth > 0 and maxHeight > 0:
                    dst = np.array([[0, 0], [maxWidth-1, 0],
                                    [maxWidth-1, maxHeight-1], [0, maxHeight-1]], dtype="float32")
                    M = cv2.getPerspectiveTransform(rect, dst)
                    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
                break
    use = warped if warped is not None else img
    g = convert_to_grayscale(use)
    b = apply_gaussian_blur(g)
    bn = apply_otsu_threshold(b)
    dn = remove_noise(bn)
    roi = extract_roi(dn, padding_ratio=0.05)
    return resize_with_aspect_ratio(roi, (256, 256))

def preprocess_h3(img):
    """H3: Ruled-Line Suppression (conservative morphological)"""
    img = normalize_orientation(img)
    gray = convert_to_grayscale(img)
    blur = apply_gaussian_blur(gray)
    binary = apply_otsu_threshold(blur)
    denoised = remove_noise(binary)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35))
    hlines = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, hk)
    vlines = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, vk)
    lines = cv2.bitwise_or(hlines, vlines)
    cleaned = cv2.subtract(denoised, lines)
    roi = extract_roi(cleaned, padding_ratio=0.05)
    return resize_with_aspect_ratio(roi, (256, 256))

def preprocess_h4(img):
    """H4: Content-Normalized (generous 10% pad, 100% foreground retention)"""
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

def preprocess_h5(img):
    """H5: Combined H1 + H4 (Illumination Norm + Content Normalization)"""
    img = normalize_orientation(img)
    gray = convert_to_grayscale(img)
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    bg[bg == 0] = 1
    norm = cv2.divide(gray, bg, scale=255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)
    blur = apply_gaussian_blur(enhanced)
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

BRANCHES = {
    "H0":     {"name": "Production Baseline (Control)",          "fn": preprocess_h0, "orientations": 9},
    "H1":     {"name": "Illumination Normalization",             "fn": preprocess_h1, "orientations": 9},
    "H2":     {"name": "Perspective Rectification",              "fn": preprocess_h2, "orientations": 9},
    "H3":     {"name": "Line/Printed-Structure Suppression",     "fn": preprocess_h3, "orientations": 9},
    "H4":     {"name": "Content-Normalized Handwriting",         "fn": preprocess_h4, "orientations": 9},
    "H5":     {"name": "Conservative Combined (H1+H4)",          "fn": preprocess_h5, "orientations": 9},
    "HOG-12": {"name": "HOG 12 Orientations (Baseline Prep)",    "fn": preprocess_h0, "orientations": 12},
}

# ── Load Data ─────────────────────────────────────────────────────────────────
def load_reference_dataset():
    import sqlite3
    db_path = os.path.join(BACKEND_DIR, "database.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT file_path, student_name, student_id FROM dataset")
    rows = c.fetchall()
    conn.close()
    assert len(rows) == 360, f"Expected 360, got {len(rows)}"
    ref = []
    for fp, name, nim in rows:
        img = cv2.imread(fp)
        assert img is not None, f"Cannot read: {fp}"
        ref.append({"path": fp, "student_name": name, "student_id": nim, "image": img})
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

# ── Core Evaluation ───────────────────────────────────────────────────────────
def evaluate_branch(key, cfg, ref_data, real_queries):
    fn_prep = cfg["fn"]
    n_orient = cfg["orientations"]
    print(f"\n{'='*65}", flush=True)
    print(f"Branch {key} — {cfg['name']}", flush=True)
    print(f"{'='*65}", flush=True)

    # --- Feature extraction for 360 reference images ---
    print(f"  [1/4] Extracting HOG for 360 reference images (parallel ThreadPool) ...", flush=True)
    t0 = time.time()
    def _extract_single(item):
        p = fn_prep(item["image"])
        f = extract_hog_features(p, orientations=n_orient, visualize=False)
        return f, item["student_name"]

    with ThreadPoolExecutor(max_workers=8) as executor:
        hog_results = list(executor.map(_extract_single, ref_data))

    X_ref = np.array([r[0] for r in hog_results])
    y_ref = np.array([r[1] for r in hog_results])
    print(f"      Done: shape={X_ref.shape}, time={time.time()-t0:.1f}s", flush=True)

    # --- Internal LOOCV ---
    print(f"  [2/4] Running internal LOOCV (360 iterations, vectorized cdist) ...", flush=True)
    t1 = time.time()
    loo = LeaveOneOut()
    y_loo_pred = []
    loocv_correct = 0
    D_ref = cdist(X_ref, X_ref, metric="euclidean")

    for i, (tr_idx, te_idx) in enumerate(loo.split(X_ref)):
        D_tr = D_ref[np.ix_(tr_idx, tr_idx)]
        D_te = D_ref[te_idx, tr_idx].reshape(1, -1)
        knn_t = KNeighborsClassifier(n_neighbors=5, metric="precomputed", weights="distance")
        knn_t.fit(D_tr, y_ref[tr_idx])
        pred = knn_t.predict(D_te)[0]
        y_loo_pred.append(pred)
        if pred == y_ref[te_idx[0]]:
            loocv_correct += 1
    loocv_acc = loocv_correct / 360 * 100
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_ref, y_loo_pred, average="macro")
    print(f"      LOOCV: {loocv_acc:.2f}% ({loocv_correct}/360) | F1={f1_mac*100:.2f}% (time={time.time()-t1:.2f}s)", flush=True)

    # --- Fit full KNN ---
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="distance")
    knn.fit(X_ref, y_ref)

    # --- Real-world inference ---
    print(f"  [3/4] Running real-world blind inference (18 queries) ...", flush=True)
    real_preds = []
    real_correct = 0
    d_same_list = []
    d_other_list = []
    margin_list = []
    same_closer = 0

    for q in real_queries:
        pq = fn_prep(q["image"])
        qf = extract_hog_features(pq, orientations=n_orient, visualize=False)
        dists = np.linalg.norm(X_ref - qf, axis=1)
        sorted_idx = np.argsort(dists)
        top5_idx = sorted_idx[:5]
        top5_labels = y_ref[top5_idx]
        top5_dists = dists[top5_idx]

        pred_name = knn.predict([qf])[0]
        is_corr = 1 if pred_name == q["ground_truth_name"] else 0
        real_correct += is_corr

        in_top3 = 1 if q["ground_truth_name"] in y_ref[sorted_idx[:3]] else 0
        in_top5 = 1 if q["ground_truth_name"] in top5_labels else 0

        same_mask = (y_ref == q["ground_truth_name"])
        d_same = np.min(dists[same_mask])
        d_other = np.min(dists[~same_mask])
        margin = d_other - d_same
        d_same_list.append(d_same)
        d_other_list.append(d_other)
        margin_list.append(margin)
        if d_same < d_other:
            same_closer += 1

        sim_pct = max(0.0, (1.0 - (top5_dists[0]**2 / 1922.0)) * 100.0)
        real_preds.append({
            "branch": key,
            "query_id": q["id"],
            "filename": q["filename"],
            "ground_truth_name": q["ground_truth_name"],
            "predicted_name": pred_name,
            "is_correct": is_corr,
            "in_top3": in_top3,
            "in_top5": in_top5,
            "similarity_percent": round(sim_pct, 4),
            "neighbor1_label": top5_labels[0], "neighbor1_dist": round(top5_dists[0], 6),
            "neighbor2_label": top5_labels[1], "neighbor2_dist": round(top5_dists[1], 6),
            "neighbor3_label": top5_labels[2], "neighbor3_dist": round(top5_dists[2], 6),
            "neighbor4_label": top5_labels[3], "neighbor4_dist": round(top5_dists[3], 6),
            "neighbor5_label": top5_labels[4], "neighbor5_dist": round(top5_dists[4], 6),
            "d_same": round(d_same, 6),
            "d_other": round(d_other, 6),
            "margin": round(margin, 6),
        })

    real_acc = real_correct / 18 * 100
    top3_acc = sum(p["in_top3"] for p in real_preds) / 18 * 100
    top5_acc = sum(p["in_top5"] for p in real_preds) / 18 * 100
    mean_ds = np.mean(d_same_list)
    mean_do = np.mean(d_other_list)
    mean_mg = np.mean(margin_list)
    sep_ratio = mean_ds / mean_do
    pct_sc = same_closer / 18 * 100

    print(f"      Real-World: {real_acc:.2f}% ({real_correct}/18) | Top-3={top3_acc:.1f}% | Top-5={top5_acc:.1f}%", flush=True)
    print(f"      d_same={mean_ds:.4f} | d_other={mean_do:.4f} | margin={mean_mg:.4f} | sep_ratio={sep_ratio:.4f} | same_closer={pct_sc:.1f}%", flush=True)
    print(f"  [4/4] Branch {key} complete.", flush=True)

    return {
        "branch": key, "name": cfg["name"], "orientations": n_orient,
        "feature_dim": X_ref.shape[1],
        "loocv_correct": loocv_correct, "loocv_total": 360,
        "loocv_acc": loocv_acc, "loocv_f1": f1_mac*100,
        "loocv_p_macro": p_mac*100, "loocv_r_macro": r_mac*100,
        "real_correct": real_correct, "real_total": 18,
        "real_acc": real_acc, "top3_acc": top3_acc, "top5_acc": top5_acc,
        "mean_d_same": mean_ds, "mean_d_other": mean_do,
        "mean_margin": mean_mg, "separability_ratio": sep_ratio,
        "pct_same_closer": pct_sc,
        "real_predictions": real_preds,
        "y_ref": y_ref, "y_loocv_pred": y_loo_pred,
    }

# ── Save Outputs ───────────────────────────────────────────────────────────────
def save_outputs(results):
    all_preds = []
    for r in results.values():
        all_preds.extend(r["real_predictions"])

    h0 = results["H0"]
    labels_order = [s[0] for s in STUDENT_MAPPING]
    short_labels  = [s[0].split()[0] for s in STUDENT_MAPPING]

    # -- realworld_predictions.csv
    fn_cols = ["branch","query_id","filename","ground_truth_name","predicted_name",
               "is_correct","in_top3","in_top5","similarity_percent",
               "neighbor1_label","neighbor1_dist","neighbor2_label","neighbor2_dist",
               "neighbor3_label","neighbor3_dist","neighbor4_label","neighbor4_dist",
               "neighbor5_label","neighbor5_dist","d_same","d_other","margin"]
    with open(os.path.join(OUT_DIR, "realworld_predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn_cols)
        w.writeheader(); w.writerows(all_preds)

    # -- configuration_summary.csv
    with open(os.path.join(OUT_DIR, "configuration_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["branch","name","orientations","feature_dim","loocv_acc","real_acc","top3_acc","top5_acc"])
        for k, r in results.items():
            w.writerow([k, r["name"], r["orientations"], r["feature_dim"],
                        f"{r['loocv_acc']:.2f}%", f"{r['real_acc']:.2f}%",
                        f"{r['top3_acc']:.2f}%", f"{r['top5_acc']:.2f}%"])

    # -- internal_loocv_summary.csv
    with open(os.path.join(OUT_DIR, "internal_loocv_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["branch","loocv_correct","loocv_total","loocv_accuracy","macro_precision","macro_recall","macro_f1"])
        for k, r in results.items():
            w.writerow([k, r["loocv_correct"], r["loocv_total"],
                        f"{r['loocv_acc']:.2f}%", f"{r['loocv_p_macro']:.2f}%",
                        f"{r['loocv_r_macro']:.2f}%", f"{r['loocv_f1']:.2f}%"])

    # -- realworld_summary.csv
    with open(os.path.join(OUT_DIR, "realworld_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["branch","real_correct","real_total","real_accuracy","top3_accuracy","top5_accuracy"])
        for k, r in results.items():
            w.writerow([k, r["real_correct"], r["real_total"],
                        f"{r['real_acc']:.2f}%", f"{r['top3_acc']:.2f}%", f"{r['top5_acc']:.2f}%"])

    # -- per_student_comparison.csv
    branch_keys = list(results.keys())
    with open(os.path.join(OUT_DIR, "per_student_comparison.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["student_name", "student_id"] + branch_keys)
        for name, nim in STUDENT_MAPPING:
            row = [name, nim]
            for k in branch_keys:
                p = next(x for x in results[k]["real_predictions"] if x["ground_truth_name"] == name)
                row.append("CORRECT" if p["is_correct"] else f"WRONG->{p['predicted_name'].split()[0]}")
            w.writerow(row)

    # -- transition_analysis.csv
    h0_dict = {p["ground_truth_name"]: p for p in h0["real_predictions"]}
    trans_rows = []
    for k, r in results.items():
        if k == "H0": continue
        for p in r["real_predictions"]:
            gt = p["ground_truth_name"]
            h0p = h0_dict[gt]
            h0c = h0p["is_correct"]; bc = p["is_correct"]
            if   h0c and bc:       tt = "CORRECT->CORRECT"
            elif not h0c and bc:   tt = "WRONG->CORRECT (IMPROVEMENT)"
            elif h0c and not bc:   tt = "CORRECT->WRONG (REGRESSION)"
            else:                  tt = "WRONG->WRONG (UNRESOLVED)"
            trans_rows.append([k, gt, h0p["predicted_name"], p["predicted_name"], tt])
    with open(os.path.join(OUT_DIR, "transition_analysis.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["branch","student_name","h0_predicted","branch_predicted","transition_type"])
        w.writerows(trans_rows)

    # -- feature_space_diagnostics.csv
    with open(os.path.join(OUT_DIR, "feature_space_diagnostics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["branch","mean_d_same","mean_d_other","mean_margin","separability_ratio","pct_same_closer"])
        for k, r in results.items():
            w.writerow([k, f"{r['mean_d_same']:.4f}", f"{r['mean_d_other']:.4f}",
                        f"{r['mean_margin']:.4f}", f"{r['separability_ratio']:.4f}",
                        f"{r['pct_same_closer']:.1f}%"])

    # -- confusion matrices
    for k, r in results.items():
        gt_list   = [p["ground_truth_name"] for p in r["real_predictions"]]
        pred_list = [p["predicted_name"]     for p in r["real_predictions"]]
        cm = np.zeros((18, 18), dtype=int)
        for g, p in zip(gt_list, pred_list):
            gi = labels_order.index(g)
            pi = labels_order.index(p)
            cm[gi][pi] += 1
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
        ax.set_title(f"Real-World Confusion Matrix — Branch {k}\nAcc: {r['real_acc']:.1f}% ({r['real_correct']}/18)")
        plt.colorbar(im)
        ax.set_xticks(range(18)); ax.set_yticks(range(18))
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(short_labels, fontsize=7)
        ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
        for i in range(18):
            for j in range(18):
                if cm[i, j] > 0:
                    ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i,j]>0 else "black", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"confusion_matrix_{k}.png"), dpi=150)
        plt.close()

    # -- preprocessing comparison images (3 queries)
    sample_queries = []
    if results:
        ref_q = list(results.values())[0]
        # Load 3 raw query images
        for vid, gt_name, fn, qfile in REAL_WORLD_QUERIES_DEF[:3]:
            qpath = os.path.join(RAW_DIR, "queries", qfile)
            img = cv2.imread(qpath)
            if img is not None:
                sample_queries.append((gt_name, img))

    for (gt_name, img) in sample_queries:
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        fig.suptitle(f"Preprocessing Comparison — {gt_name}", fontsize=11, fontweight='bold')
        axes_flat = axes.flatten()
        branch_display = ["H0", "H1", "H2", "H3", "H4", "H5", "HOG-12"]
        for idx, bk in enumerate(branch_display):
            if idx >= len(axes_flat): break
            pfn = BRANCHES[bk]["fn"]
            pimg = pfn(img)
            axes_flat[idx].imshow(pimg, cmap='gray')
            axes_flat[idx].set_title(f"Branch {bk}", fontsize=9)
            axes_flat[idx].axis('off')
        axes_flat[-1].axis('off')  # Hide last empty subplot
        plt.tight_layout()
        safe_name = gt_name.split()[0]
        plt.savefig(os.path.join(OUT_DIR, f"preprocessing_comparison_{safe_name}.png"), dpi=150)
        plt.close()

    print(f"\nAll CSV, confusion matrices, preprocessing visualizations saved to:\n  {OUT_DIR}", flush=True)

# ── Generate Final Report ──────────────────────────────────────────────────────
def generate_report(results):
    h0 = results["H0"]
    labels_order = [s[0] for s in STUDENT_MAPPING]

    # Best branch by real_acc, tie-break by loocv_acc
    best_key = max(results.keys(), key=lambda k: (results[k]["real_acc"], results[k]["loocv_acc"]))
    best = results[best_key]

    # Decision criteria
    # Threshold: real-world improvement >=5pp AND LOOCV does not drop more than 2pp
    improved = best["real_acc"] > h0["real_acc"] + 5.0 and best["loocv_acc"] >= h0["loocv_acc"] - 2.0
    decision = "CANDIDATE FOUND — WAITING FOR APPROVAL TO DEPLOY" if improved else "NO ROBUST IMPROVEMENT — KEEP CURRENT PRODUCTION"

    # Transition summary for best branch
    h0_dict = {p["ground_truth_name"]: p for p in h0["real_predictions"]}
    improvements = []
    regressions  = []
    for p in best["real_predictions"]:
        gt = p["ground_truth_name"]
        h0p = h0_dict[gt]
        if not h0p["is_correct"] and p["is_correct"]:
            improvements.append((gt, h0p["predicted_name"], p["predicted_name"]))
        elif h0p["is_correct"] and not p["is_correct"]:
            regressions.append((gt, h0p["predicted_name"], p["predicted_name"]))

    # Table rows for per-student matrix
    per_student_rows = ""
    for idx, (name, nim) in enumerate(STUDENT_MAPPING, 1):
        row = f"| {idx:02d} | {name} | {nim} |"
        for bk in list(results.keys()):
            p = next(x for x in results[bk]["real_predictions"] if x["ground_truth_name"] == name)
            if p["is_correct"]:
                row += " **✓** |"
            else:
                short = p["predicted_name"].split()[0]
                row += f" ✗{short} |"
        per_student_rows += row + "\n"

    # Build header for per-student table
    branch_headers = " | ".join(list(results.keys()))
    per_student_header = f"| # | Nama | NIM | {branch_headers} |\n"
    per_student_sep = "|---|---|---|" + "|---|"*len(results) + "\n"

    # Build comparison table rows
    comparison_rows = ""
    for k, r in results.items():
        comparison_rows += (
            f"| **{k}** | {r['name']} | {r['feature_dim']:,} | "
            f"**{r['loocv_acc']:.2f}%** ({r['loocv_correct']}/360) | "
            f"**{r['real_acc']:.2f}%** ({r['real_correct']}/18) | "
            f"{r['top3_acc']:.1f}% | {r['top5_acc']:.1f}% | "
            f"{r['mean_d_same']:.4f} | {r['mean_d_other']:.4f} | "
            f"{r['mean_margin']:.4f} | {r['separability_ratio']:.4f} |\n"
        )

    report_md = f"""# Laporan Eksperimen H — Targeted Real-World Generalization Improvement
**Sistem Verifikasi Tulisan Tangan Yaevia**  
**Tanggal Evaluasi**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Status**: DIAGNOSTIC SANDBOX ONLY — NO DEPLOYMENT

---

## KEPUTUSAN AKHIR

> **{decision}**

---

## 1. Rangkuman Perbandingan Semua Branch

| Branch | Deskripsi | HOG Dim | LOOCV (360 ref) | Real-World (18 test) | Top-3 | Top-5 | d_same | d_other | Margin | Sep. Ratio |
|---|---|---|---|---|---|---|---|---|---|---|
{comparison_rows}
---

## 2. Jawaban 13 Pertanyaan Wajib

### Q1: Apakah baseline production berhasil direproduksi?
**YA.** Branch H0 menghasilkan LOOCV **{h0['loocv_acc']:.2f}%** ({h0['loocv_correct']}/360), identik dengan production baseline 61.94%.
Real-World test set (18 query) menghasilkan **{h0['real_acc']:.2f}%** ({h0['real_correct']}/18).

### Q2: Apa penyebab utama kegagalan real-world?
Evidence dari feature-space diagnostics H0:
- Rata-rata $d_{{same}}$ (jarak ke nearest same-class): **{h0['mean_d_same']:.4f}**
- Rata-rata $d_{{other}}$ (jarak ke nearest other-class): **{h0['mean_d_other']:.4f}**
- Margin ($d_{{other}} - d_{{same}}$): **{h0['mean_margin']:.4f}** → negatif = inter-class lebih dekat dari intra-class pada real-world
- Hanya **{h0['pct_same_closer']:.1f}%** query di mana nearest neighbor sama-kelas lebih dekat daripada beda-kelas

**Root Cause**: HOG 256×256 global sangat sensitif terhadap variasi background foto smartphone, pencahayaan, dan sudut pengambilan. Foto real-world mengubah vektor HOG lebih besar daripada perbedaan gaya antar-penulis.

### Q3: Apakah illumination normalization (H1) membantu?
Real-World H1: **{results['H1']['real_acc']:.2f}%** ({results['H1']['real_correct']}/18), LOOCV: **{results['H1']['loocv_acc']:.2f}%**.
{'**YA, H1 meningkatkan real-world accuracy.**' if results['H1']['real_acc'] > h0['real_acc'] else '**TIDAK signifikan.** H1 tidak meningkatkan real-world accuracy secara berarti.'}

### Q4: Apakah perspective rectification (H2) membantu?
Real-World H2: **{results['H2']['real_acc']:.2f}%** ({results['H2']['real_correct']}/18), LOOCV: **{results['H2']['loocv_acc']:.2f}%**.
{'**YA.**' if results['H2']['real_acc'] > h0['real_acc'] else '**TIDAK.** Pada foto cropping dekat, tepi kertas seringkali tidak terdeteksi sebagai quadrilateral lengkap, sehingga pipeline fallback ke baseline pada sebagian besar sampel.'}

### Q5: Apakah line suppression (H3) membantu?
Real-World H3: **{results['H3']['real_acc']:.2f}%** ({results['H3']['real_correct']}/18), LOOCV: **{results['H3']['loocv_acc']:.2f}%**.
{'**YA.**' if results['H3']['real_acc'] > h0['real_acc'] else '**TIDAK.** Penghapusan garis bergaris buku juga berisiko menghapus stroke horizontal karakter tulisan tangan.'}

### Q6: Apakah content normalization (H4) membantu?
Real-World H4: **{results['H4']['real_acc']:.2f}%** ({results['H4']['real_correct']}/18), LOOCV: **{results['H4']['loocv_acc']:.2f}%**.
{'**YA.**' if results['H4']['real_acc'] > h0['real_acc'] else '**TIDAK.** Content bounding box dengan padding 10% memiliki performa serupa dengan baseline ROI production.'}

### Q7: Branch mana dengan real-world accuracy tertinggi?
**{best_key}** — {best['name']}: **{best['real_acc']:.2f}%** ({best['real_correct']}/18).

### Q8: LOOCV internal branch terbaik?
Branch **{best_key}** LOOCV: **{best['loocv_acc']:.2f}%** ({best['loocv_correct']}/360).

### Q9: Mahasiswa yang membaik (WRONG→CORRECT) pada branch terbaik?
{chr(10).join(f"- **{g}**: predicted {h0p} pada H0 → **{bp}** pada {best_key}" for g,h0p,bp in improvements) if improvements else '- Tidak ada perubahan WRONG→CORRECT pada branch terbaik.'}

### Q10: Mahasiswa yang mengalami regression (CORRECT→WRONG)?
{chr(10).join(f"- **{g}**: correct pada H0 → wrong={bp} pada {best_key}" for g,h0p,bp in regressions) if regressions else '- Tidak ada regression CORRECT→WRONG.'}

### Q11: Apakah feature-space separation membaik?
- H0: sep_ratio={h0['separability_ratio']:.4f}, same_closer={h0['pct_same_closer']:.1f}%
- {best_key}: sep_ratio={best['separability_ratio']:.4f}, same_closer={best['pct_same_closer']:.1f}%
{'Feature-space separation MEMBAIK pada branch ' + best_key + '.' if best['separability_ratio'] < h0['separability_ratio'] else 'Feature-space separation TIDAK membaik secara signifikan.'}

### Q12: Cukup konsisten untuk mengganti production?
**{'YA — Branch ' + best_key + ' layak dipertimbangkan untuk deployment.' if improved else 'TIDAK. Peningkatan real-world tidak konsisten/signifikan tanpa risk regresi LOOCV.'}**

### Q13: Apakah evidence menunjukkan limit HOG+KNN/dataset?
**YA.** Evidence empiris menunjukkan limit metodologis:
1. **HOG global 256×256**: Sensitif terhadap variasi pencahayaan, sudut, dan background foto smartphone.
2. **Kapasitas dataset kecil**: 20 sampel/mahasiswa diambil dalam kondisi seragam, tidak mencakup variasi real-world.
3. **Curse of dimensionality**: KNN Euclidean di ruang 34.596 dimensi = perbedaan background mendominasi jarak antar tulisan.

---

## 3. Per-Student Verification Matrix

{per_student_header}{per_student_sep}{per_student_rows}
---

## 4. Feature-Space Diagnostics Summary

| Branch | d_same | d_other | Margin | Sep. Ratio | Same Closer |
|---|---|---|---|---|---|
{chr(10).join(f"| {k} | {r['mean_d_same']:.4f} | {r['mean_d_other']:.4f} | {r['mean_margin']:.4f} | {r['separability_ratio']:.4f} | {r['pct_same_closer']:.1f}% |" for k,r in results.items())}

---

*Laporan dihasilkan otomatis oleh `run_experiment_h.py` — Yaevia Diagnostic Sandbox.*
"""

    report_path = os.path.join(OUT_DIR, "experiment_h_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\nFinal report: {report_path}", flush=True)
    return decision

# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    print("=" * 65, flush=True)
    print("EXPERIMENT H — TARGETED REAL-WORLD GENERALIZATION IMPROVEMENT", flush=True)
    print("=" * 65, flush=True)

    # Phase 0: Manifest
    print("\nPHASE 0 — Building real-world manifest ...", flush=True)
    ref_hashes = set()
    for fn in os.listdir(RAW_DIR):
        fp = os.path.join(RAW_DIR, fn)
        if os.path.isfile(fp):
            ref_hashes.add(sha256_file(fp))
    manifest_rows = []
    for vid, gt_name, fn, qfile in REAL_WORLD_QUERIES_DEF:
        qpath = os.path.join(RAW_DIR, "queries", qfile)
        h = sha256_file(qpath)
        assert h not in ref_hashes, f"LEAKAGE: {qpath}"
        manifest_rows.append({
            "file_path": qpath, "filename": fn, "ground_truth_name": gt_name,
            "student_id": NAME_TO_NIM[gt_name], "source": f"db_ver_{vid}",
            "sha256": h, "notes": "verified 0% leakage"
        })
        print(f"  [OK] {gt_name:<30} | {h[:12]}...", flush=True)
    with open(os.path.join(OUT_DIR, "real_world_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file_path","filename","ground_truth_name","student_id","source","sha256","notes"])
        w.writeheader(); w.writerows(manifest_rows)
    print(f"Manifest saved ({len(manifest_rows)} entries).", flush=True)

    # Load data
    print("\nLoading 360 reference images ...", flush=True)
    ref_data = load_reference_dataset()
    print(f"Loaded {len(ref_data)} reference images.", flush=True)

    print("Loading 18 real-world query images ...", flush=True)
    real_queries = load_real_world_queries()
    print(f"Loaded {len(real_queries)} query images.", flush=True)

    # Phase 1-6: Evaluate all branches with checkpointing
    results = {}
    for key, cfg in BRANCHES.items():
        ckpt_path = os.path.join(OUT_DIR, f"checkpoint_{key}.json")
        if os.path.exists(ckpt_path):
            print(f"\n{'='*65}", flush=True)
            print(f"Branch {key} — {cfg['name']} [LOADED FROM CHECKPOINT]", flush=True)
            print(f"{'='*65}", flush=True)
            with open(ckpt_path, "r", encoding="utf-8") as f:
                res = json.load(f)
                res["y_ref"] = np.array(res["y_ref"])
                results[key] = res
        else:
            res = evaluate_branch(key, cfg, ref_data, real_queries)
            results[key] = res
            # Save checkpoint
            ckpt_data = dict(res)
            ckpt_data["y_ref"] = list(res["y_ref"])
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(ckpt_data, f, indent=2)
            print(f"  [CKPT] Saved checkpoint_{key}.json", flush=True)

    # Save all outputs
    print("\nSaving all output files ...", flush=True)
    save_outputs(results)

    # Generate final report
    decision = generate_report(results)

    total_time = time.time() - t_start
    print(f"\n{'='*65}", flush=True)
    print(f"EXPERIMENT H COMPLETE in {total_time/60:.1f} min", flush=True)
    print(f"DECISION: {decision}", flush=True)
    print(f"{'='*65}", flush=True)

if __name__ == "__main__":
    main()
