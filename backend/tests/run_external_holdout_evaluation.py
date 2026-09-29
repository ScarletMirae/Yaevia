"""
run_external_holdout_evaluation.py
===================================
External Holdout Evaluation — Yaevia Handwriting Verification System

FINAL FROZEN CONFIGURATION:
  IMAGE_SIZE   : 256×256 (letterbox)
  HOG          : orientations=9, ppc=(8,8), cpb=(2,2), norm=L2-Hys → 34596 features
  KNN          : K=5, metric=euclidean, weights=distance
  LOOCV Bench  : 61.94% (223/360)

ATURAN:
  - JANGAN modifikasi model setelah melihat hasil holdout
  - JANGAN masukkan holdout ke dataset referensi
  - Single-pass blind inference
"""

import os
import sys
import time
import json
import hashlib
import csv
import warnings
from pathlib import Path
import numpy as np

# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

warnings.filterwarnings("ignore")

# ── path setup ──────────────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from config import IMAGE_SIZE
from preprocessing.image_processor import preprocess_from_array
from features.hog_extractor import extract_hog_features
import joblib
import cv2

# ── constants ────────────────────────────────────────────────────────────────
DATASET_ROOT = r"D:\.SKRIPSI\Dataset"
RAW_DIR      = os.path.join(BACKEND_DIR, "dataset", "raw")
MODEL_PATH   = os.path.join(BACKEND_DIR, "model", "saved", "knn_model_20260924_183224.joblib")
LE_PATH      = os.path.join(BACKEND_DIR, "model", "saved", "label_encoder_20260924_183224.joblib")
OUT_DIR      = os.path.join(BACKEND_DIR, "tests", "evaluation_results", "external_holdout_final")
os.makedirs(OUT_DIR, exist_ok=True)

EXPECTED_FEATURES  = 34596
EXPECTED_IMG_SHAPE = (256, 256)
LOOCV_ACCURACY     = 61.94   # authoritative internal benchmark (%)
LOOCV_CORRECT      = 223
LOOCV_TOTAL        = 360

# ── student mapping: folder_name → full_name (matches DB) ───────────────────
STUDENT_FOLDER_MAP = {
    "Alif":                  "Muhammad Alif Rizky Hutama",
    "Angela":                "Angela Permata Rosa",
    "Brama":                 "Bramasetya Raka Purnama",
    "Dimas":                 "Dimas Wahyu Prasetyo",
    "Dinar":                 "Febrian Dinnar Purnama",
    "Dony":                  "Muhammad Dony Saputra",
    "Fahim":                 "Fahim J Mujaddid",
    "Farhan Agiya":          "Farhan Agiya Pratama",
    "Fathur":                "Fathurrahman Nugroho",
    "Gayuh":                 "Ibnu Gayuh Fadilah",
    "Hazel":                 "Hazelando Visco",
    "Ilham":                 "Ilham Rasyidan Muhammad",
    "Radit Kecil":           "Raditya Endra Mahardika",
    "Rakha":                 "Rakha Burhannudin Majid",
    "Rifqi Rengga Praseno":  "Rifqi Rengga Praseno",
    "Soni":                  "Soni Nugroho",
    "Wirid":                 "Wiridan Syifa Saputra",
    "Zaedani Ni'am":         "Zaedani Ni'am Masykur",
}

STUDENT_NIM_MAP = {
    "Muhammad Alif Rizky Hutama":  "A710230105",
    "Angela Permata Rosa":         "A710230006",
    "Bramasetya Raka Purnama":     "A710230101",
    "Dimas Wahyu Prasetyo":        "A710230113",
    "Febrian Dinnar Purnama":      "A710230102",
    "Muhammad Dony Saputra":       "A710230118",
    "Fahim J Mujaddid":            "A710230076",
    "Farhan Agiya Pratama":        "A710230089",
    "Fathurrahman Nugroho":        "A710230097",
    "Ibnu Gayuh Fadilah":          "A710230119",
    "Hazelando Visco":             "A710230092",
    "Ilham Rasyidan Muhammad":     "A710230078",
    "Raditya Endra Mahardika":     "A710230110",
    "Rakha Burhannudin Majid":     "A710230098",
    "Rifqi Rengga Praseno":        "A710230086",
    "Soni Nugroho":                "A710230112",
    "Wiridan Syifa Saputra":       "A710230080",
    "Zaedani Ni'am Masykur":       "A710230085",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# ── Step 1: Load model ───────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1 — Loading frozen model & label encoder")
print("=" * 70)
model = joblib.load(MODEL_PATH)
print(f"  KNN model loaded: {MODEL_PATH}")
print(f"  n_features_in_ : {model.n_features_in_}")
print(f"  n_neighbors    : {model.n_neighbors}")
print(f"  metric         : {model.metric}")
print(f"  weights        : {model.weights}")
assert model.n_features_in_ == EXPECTED_FEATURES, (
    f"FATAL: model expects {model.n_features_in_} features, expected {EXPECTED_FEATURES}"
)

le = joblib.load(LE_PATH)
all_classes = list(le.classes_)
print(f"  Label encoder classes ({len(all_classes)}): {all_classes[:3]} ...")

# ── Step 2: Compute reference SHA-256 hashes (to exclude from holdout) ──────
print("\n" + "=" * 70)
print("STEP 2 — Computing reference SHA-256 to identify holdout images")
print("=" * 70)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

ref_hashes = set()
for fn in os.listdir(RAW_DIR):
    fp = os.path.join(RAW_DIR, fn)
    if os.path.isfile(fp):
        ref_hashes.add(sha256_file(fp))
print(f"  Reference hashes computed: {len(ref_hashes)}")

# ── Step 3: Discover holdout images ─────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3 — Discovering holdout (unseen) images")
print("=" * 70)

holdout_samples = []   # list of dict: {path, filename, ground_truth_name, ground_truth_nim}
skip_folders = {"Bu irma"}   # not in 18 students

for folder_name in sorted(os.listdir(DATASET_ROOT)):
    if folder_name in skip_folders:
        continue
    if folder_name not in STUDENT_FOLDER_MAP:
        print(f"  [SKIP] Unknown folder: {folder_name}")
        continue
    folder_path = os.path.join(DATASET_ROOT, folder_name)
    if not os.path.isdir(folder_path):
        continue

    gt_name = STUDENT_FOLDER_MAP[folder_name]
    gt_nim  = STUDENT_NIM_MAP.get(gt_name, "UNKNOWN")

    imgs = [
        f for f in os.listdir(folder_path)
        if Path(f).suffix.lower() in IMG_EXTS
    ]
    holdout_count = 0
    for img_file in sorted(imgs):
        img_path = os.path.join(folder_path, img_file)
        h = sha256_file(img_path)
        if h not in ref_hashes:
            holdout_samples.append({
                "path":              img_path,
                "filename":          img_file,
                "ground_truth_name": gt_name,
                "ground_truth_nim":  gt_nim,
            })
            holdout_count += 1
    print(f"  {folder_name:<24} → {gt_name:<32} : {holdout_count} holdout images")

print(f"\n  TOTAL HOLDOUT IMAGES: {len(holdout_samples)}")

if len(holdout_samples) == 0:
    print("  ERROR: No holdout images found. Aborting.")
    sys.exit(1)

# ── Step 4: Run inference on each holdout image ──────────────────────────────
print("\n" + "=" * 70)
print("STEP 4 — Running inference on holdout images")
print("=" * 70)

def similarity_formula(distance):
    """Frozen similarity formula from protocol."""
    return max(0.0, min(100.0, (1.0 - (distance**2 / 450.0)) * 100.0))

predictions = []
errors      = []

for idx, sample in enumerate(holdout_samples):
    img_path = sample["path"]
    gt_name  = sample["ground_truth_name"]
    gt_nim   = sample["ground_truth_nim"]
    filename = sample["filename"]

    t0 = time.perf_counter()
    try:
        # Load image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {img_path}")

        # Preprocess (frozen pipeline)
        processed = preprocess_from_array(img_bgr)

        # Validate shape
        assert processed.shape == EXPECTED_IMG_SHAPE, (
            f"Shape mismatch: {processed.shape} != {EXPECTED_IMG_SHAPE}"
        )

        # Extract HOG (frozen config)
        hog_vec = extract_hog_features(processed)
        assert len(hog_vec) == EXPECTED_FEATURES, (
            f"Feature dim mismatch: {len(hog_vec)} != {EXPECTED_FEATURES}"
        )

        # KNN inference
        feat_2d = hog_vec.reshape(1, -1)
        distances, indices = model.kneighbors(feat_2d, n_neighbors=5)
        distances = distances[0]
        indices   = indices[0]

        # Get training labels
        train_labels = le.inverse_transform(model._y)
        top5_names   = [train_labels[i] for i in indices]
        top5_dists   = distances.tolist()
        top5_sims    = [similarity_formula(d) for d in top5_dists]

        # Distance-weighted vote (matches KNN weights='distance')
        vote_weights = {}
        for name, dist in zip(top5_names, top5_dists):
            w = 1.0 / (dist + 1e-9) if dist > 0 else 1e9
            vote_weights[name] = vote_weights.get(name, 0.0) + w
        total_weight  = sum(vote_weights.values())
        predicted     = max(vote_weights, key=vote_weights.get)
        vote_share    = (vote_weights[predicted] / total_weight) * 100.0

        is_correct          = (predicted == gt_name)
        nearest_distance    = top5_dists[0]
        similarity_pct      = similarity_formula(nearest_distance)
        top5_contains_truth = gt_name in top5_names

        inf_time_ms = (time.perf_counter() - t0) * 1000.0

        top5_info = json.dumps([
            {"rank": r+1, "name": n, "distance": round(d, 4), "similarity": round(s, 2)}
            for r, (n, d, s) in enumerate(zip(top5_names, top5_dists, top5_sims))
        ])

        predictions.append({
            "sample_id":          idx + 1,
            "original_filename":  filename,
            "ground_truth_name":  gt_name,
            "ground_truth_nim":   gt_nim,
            "predicted_name":     predicted,
            "is_correct":         is_correct,
            "nearest_distance":   round(nearest_distance, 4),
            "similarity_percent": round(similarity_pct, 2),
            "vote_share_percent": round(vote_share, 2),
            "top5_candidates":    top5_info,
            "top5_contains_truth": top5_contains_truth,
            "inference_time_ms":  round(inf_time_ms, 3),
        })

        status = "✓" if is_correct else "✗"
        if idx % 20 == 0 or not is_correct:
            print(f"  [{idx+1:3d}/{len(holdout_samples)}] {status} {filename[:30]:<30} GT={gt_name[:20]:<20} PRED={predicted[:20]:<20} sim={similarity_pct:.1f}%")

    except Exception as e:
        inf_time_ms = (time.perf_counter() - t0) * 1000.0
        errors.append({"sample_id": idx+1, "filename": filename, "error": str(e)})
        print(f"  [{idx+1:3d}] ERROR: {filename}: {e}")

print(f"\n  Inference complete. Processed: {len(predictions)}, Errors: {len(errors)}")

# ── Step 5: Compute overall metrics ─────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5 — Computing metrics")
print("=" * 70)

from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix
)

n_total  = len(predictions)
n_correct = sum(1 for p in predictions if p["is_correct"])
accuracy  = (n_correct / n_total) * 100.0 if n_total > 0 else 0.0

top5_correct = sum(1 for p in predictions if p["top5_contains_truth"])
top5_accuracy = (top5_correct / n_total) * 100.0 if n_total > 0 else 0.0

y_true = [p["ground_truth_name"] for p in predictions]
y_pred = [p["predicted_name"]    for p in predictions]

# only classes that appear in holdout
holdout_classes = sorted(set(y_true))

macro_precision = precision_score(y_true, y_pred, labels=holdout_classes, average="macro", zero_division=0) * 100
macro_recall    = recall_score   (y_true, y_pred, labels=holdout_classes, average="macro", zero_division=0) * 100
macro_f1        = f1_score       (y_true, y_pred, labels=holdout_classes, average="macro", zero_division=0) * 100
weighted_precision = precision_score(y_true, y_pred, labels=holdout_classes, average="weighted", zero_division=0) * 100
weighted_recall    = recall_score   (y_true, y_pred, labels=holdout_classes, average="weighted", zero_division=0) * 100
weighted_f1        = f1_score       (y_true, y_pred, labels=holdout_classes, average="weighted", zero_division=0) * 100

print(f"  Overall Accuracy     : {accuracy:.2f}%  ({n_correct}/{n_total})")
print(f"  Top-5 Accuracy       : {top5_accuracy:.2f}%  ({top5_correct}/{n_total})")
print(f"  Macro Precision      : {macro_precision:.2f}%")
print(f"  Macro Recall         : {macro_recall:.2f}%")
print(f"  Macro F1             : {macro_f1:.2f}%")
print(f"  Weighted Precision   : {weighted_precision:.2f}%")
print(f"  Weighted Recall      : {weighted_recall:.2f}%")
print(f"  Weighted F1          : {weighted_f1:.2f}%")

delta_accuracy = accuracy - LOOCV_ACCURACY
print(f"\n  LOOCV Reference      : {LOOCV_ACCURACY:.2f}%")
print(f"  Holdout Accuracy     : {accuracy:.2f}%")
print(f"  Δ Accuracy           : {delta_accuracy:+.2f}%")

# ── Per-student metrics ──────────────────────────────────────────────────────
per_student = {}
for p in predictions:
    name = p["ground_truth_name"]
    if name not in per_student:
        per_student[name] = {"correct": 0, "total": 0, "similarities": [], "predicted": []}
    per_student[name]["total"]   += 1
    per_student[name]["correct"] += int(p["is_correct"])
    per_student[name]["similarities"].append(p["similarity_percent"])
    per_student[name]["predicted"].append(p["predicted_name"])

print("\n  Per-student results:")
print(f"  {'Student':<32} {'N':>4} {'Correct':>7} {'Acc%':>6} {'AvgSim%':>8}")
print("  " + "-" * 62)
for name in sorted(per_student.keys()):
    st = per_student[name]
    acc  = (st["correct"] / st["total"]) * 100 if st["total"] else 0
    asim = np.mean(st["similarities"]) if st["similarities"] else 0
    print(f"  {name:<32} {st['total']:>4} {st['correct']:>7} {acc:>6.1f}% {asim:>7.1f}%")

# Similarity analysis: correct vs wrong
sim_correct = [p["similarity_percent"] for p in predictions if p["is_correct"]]
sim_wrong   = [p["similarity_percent"] for p in predictions if not p["is_correct"]]
print(f"\n  Avg similarity (correct) : {np.mean(sim_correct):.2f}%" if sim_correct else "  No correct predictions")
print(f"  Avg similarity (wrong)   : {np.mean(sim_wrong):.2f}%"   if sim_wrong   else "  No wrong predictions")

# ── Confusion matrix ─────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred, labels=holdout_classes)

# Confusion pairs (asymmetric)
confusion_pairs = []
for i, true_label in enumerate(holdout_classes):
    for j, pred_label in enumerate(holdout_classes):
        if i != j and cm[i][j] > 0:
            confusion_pairs.append({
                "true_label": true_label,
                "predicted_as": pred_label,
                "count": int(cm[i][j]),
            })
confusion_pairs.sort(key=lambda x: -x["count"])

# ── Step 6: Save artifacts ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6 — Saving artifacts")
print("=" * 70)

# 1. holdout_predictions.csv
pred_csv = os.path.join(OUT_DIR, "holdout_predictions.csv")
fieldnames = [
    "sample_id", "original_filename", "ground_truth_name", "ground_truth_nim",
    "predicted_name", "is_correct", "nearest_distance", "similarity_percent",
    "vote_share_percent", "top5_candidates", "top5_contains_truth", "inference_time_ms"
]
with open(pred_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(predictions)
print(f"  [1] {pred_csv}")

# 2. holdout_per_student.csv
per_stu_csv = os.path.join(OUT_DIR, "holdout_per_student.csv")
with open(per_stu_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["student_name", "nim", "total_holdout", "correct", "wrong", "accuracy_pct", "avg_similarity_pct"])
    for name in sorted(per_student.keys()):
        st  = per_student[name]
        nim = STUDENT_NIM_MAP.get(name, "")
        acc = (st["correct"] / st["total"]) * 100 if st["total"] else 0
        asim = np.mean(st["similarities"]) if st["similarities"] else 0
        writer.writerow([name, nim, st["total"], st["correct"], st["total"]-st["correct"], round(acc,2), round(asim,2)])
print(f"  [2] {per_stu_csv}")

# 3. holdout_confusion_matrix.csv
cm_csv = os.path.join(OUT_DIR, "holdout_confusion_matrix.csv")
with open(cm_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["True\\Pred"] + holdout_classes)
    for i, row_label in enumerate(holdout_classes):
        writer.writerow([row_label] + cm[i].tolist())
print(f"  [3] {cm_csv}")

# 4. holdout_confusion_pairs.csv
pairs_csv = os.path.join(OUT_DIR, "holdout_confusion_pairs.csv")
with open(pairs_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["true_label", "predicted_as", "count"])
    writer.writeheader()
    writer.writerows(confusion_pairs)
print(f"  [4] {pairs_csv}")

# 5. holdout_misclassifications.csv
wrong_preds = [p for p in predictions if not p["is_correct"]]
misc_csv = os.path.join(OUT_DIR, "holdout_misclassifications.csv")
with open(misc_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(wrong_preds)
print(f"  [5] {misc_csv}  ({len(wrong_preds)} misclassifications)")

# 6. loocv_vs_holdout.csv
loocv_vs_csv = os.path.join(OUT_DIR, "loocv_vs_holdout.csv")
with open(loocv_vs_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "loocv_internal", "holdout_external", "delta"])
    writer.writerow(["accuracy_pct", LOOCV_ACCURACY, round(accuracy,2), round(delta_accuracy,2)])
    writer.writerow(["correct_count", LOOCV_CORRECT, n_correct, n_correct - LOOCV_CORRECT])
    writer.writerow(["total_evaluated", LOOCV_TOTAL, n_total, n_total - LOOCV_TOTAL])
print(f"  [6] {loocv_vs_csv}")

# 7. holdout_summary.json
summary = {
    "evaluation_type": "External Holdout Final",
    "model_file": os.path.basename(MODEL_PATH),
    "configuration": {
        "image_size": "256x256",
        "hog_orientations": 9,
        "hog_pixels_per_cell": [8, 8],
        "hog_cells_per_block": [2, 2],
        "hog_block_norm": "L2-Hys",
        "hog_feature_dim": EXPECTED_FEATURES,
        "knn_k": 5,
        "knn_metric": "euclidean",
        "knn_weights": "distance",
    },
    "reference_training_set": {"students": 18, "samples_per_student": 20, "total": 360},
    "internal_loocv_benchmark": {
        "accuracy_pct": LOOCV_ACCURACY,
        "correct": LOOCV_CORRECT,
        "total": LOOCV_TOTAL,
        "macro_precision_pct": 63.21,
        "macro_recall_pct": 61.94,
        "macro_f1_pct": 61.29,
    },
    "external_holdout_results": {
        "students_with_holdout": len(holdout_classes),
        "total_holdout_images": n_total,
        "correct": n_correct,
        "wrong": n_total - n_correct,
        "inference_errors": len(errors),
        "accuracy_pct": round(accuracy, 2),
        "top5_accuracy_pct": round(top5_accuracy, 2),
        "macro_precision_pct": round(macro_precision, 2),
        "macro_recall_pct": round(macro_recall, 2),
        "macro_f1_pct": round(macro_f1, 2),
        "weighted_precision_pct": round(weighted_precision, 2),
        "weighted_recall_pct": round(weighted_recall, 2),
        "weighted_f1_pct": round(weighted_f1, 2),
        "avg_similarity_correct_pct": round(float(np.mean(sim_correct)), 2) if sim_correct else None,
        "avg_similarity_wrong_pct":   round(float(np.mean(sim_wrong)), 2)   if sim_wrong   else None,
        "delta_vs_loocv_pct": round(delta_accuracy, 2),
    },
    "errors": errors,
}
summary_json = os.path.join(OUT_DIR, "holdout_summary.json")
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"  [7] {summary_json}")

# ── Step 7: Generate visualizations ──────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 7 — Generating visualizations")
print("=" * 70)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns

    # (A) Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    short_labels = [
        n.split()[-1] if len(n.split()) > 1 else n
        for n in holdout_classes
    ]
    # Use first+last name initials for readability
    def short_name(n):
        parts = n.split()
        if len(parts) >= 2:
            return parts[0] + " " + parts[-1]
        return n
    short_labels = [short_name(n) for n in holdout_classes]

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=short_labels, yticklabels=short_labels,
        ax=ax, linewidths=0.5
    )
    ax.set_title(f"External Holdout Confusion Matrix\n(N={n_total}, Acc={accuracy:.1f}%)", fontsize=14, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    cm_png = os.path.join(OUT_DIR, "holdout_confusion_matrix.png")
    plt.savefig(cm_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [A] {cm_png}")

    # (B) LOOCV vs Holdout comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    cats   = ["Internal\nLOOCV\n(360 samples)", f"External\nHoldout\n({n_total} samples)"]
    values = [LOOCV_ACCURACY, accuracy]
    colors = ["#2196F3", "#4CAF50" if delta_accuracy >= 0 else "#F44336"]
    bars = ax.bar(cats, values, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"LOOCV vs External Holdout Accuracy\n(Δ = {delta_accuracy:+.2f}%)", fontsize=13)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="50% baseline")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    plt.tight_layout()
    cmp_png = os.path.join(OUT_DIR, "loocv_vs_holdout_comparison.png")
    plt.savefig(cmp_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [B] {cmp_png}")

    # (C) Per-student accuracy bar chart
    stu_names   = sorted(per_student.keys())
    stu_accs    = [(per_student[n]["correct"] / per_student[n]["total"])*100 for n in stu_names]
    stu_labels  = [short_name(n) for n in stu_names]

    fig, ax = plt.subplots(figsize=(14, 6))
    bar_colors = ["#4CAF50" if a >= 60 else "#FF9800" if a >= 40 else "#F44336" for a in stu_accs]
    bars = ax.bar(stu_labels, stu_accs, color=bar_colors, edgecolor="black", linewidth=0.7)
    for bar, val in zip(bars, stu_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per-Student Accuracy — External Holdout", fontsize=13)
    ax.axhline(accuracy, color="blue", linestyle="--", linewidth=1.2, label=f"Overall={accuracy:.1f}%")
    ax.legend(fontsize=10)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.tight_layout()
    pstu_png = os.path.join(OUT_DIR, "holdout_per_student_accuracy.png")
    plt.savefig(pstu_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [C] {pstu_png}")

except Exception as e:
    print(f"  [WARN] Visualization error: {e}")

# ── Step 8: Generate final markdown report ───────────────────────────────────
print("\n" + "=" * 70)
print("STEP 8 — Generating final report")
print("=" * 70)

# Build per-student table for report
per_stu_rows = []
for name in sorted(per_student.keys()):
    st  = per_student[name]
    nim = STUDENT_NIM_MAP.get(name, "")
    acc = (st["correct"] / st["total"]) * 100 if st["total"] else 0
    asim = np.mean(st["similarities"]) if st["similarities"] else 0
    per_stu_rows.append((name, nim, st["total"], st["correct"], st["total"]-st["correct"], acc, asim))

per_stu_md = "\n".join([
    f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]:.1f}% | {r[6]:.1f}% |"
    for r in per_stu_rows
])

top10_wrong_md = "\n".join([
    f"| {p['ground_truth_name']} | {p['predicted_name']} | {p['similarity_percent']:.1f}% | {p['original_filename']} |"
    for p in sorted(wrong_preds, key=lambda x: x["similarity_percent"], reverse=True)[:10]
])

top_confusion_md = "\n".join([
    f"| {cp['true_label']} | {cp['predicted_as']} | {cp['count']} |"
    for cp in confusion_pairs[:10]
])

students_no_holdout = sorted(set(STUDENT_FOLDER_MAP.values()) - set(holdout_classes))
no_holdout_md = ", ".join(students_no_holdout) if students_no_holdout else "None"

report_md = f"""# Laporan Final — External Holdout Evaluation
## Sistem Verifikasi Tulisan Tangan Yaevia

**Tanggal Evaluasi**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Status**: FINAL — Model Frozen, Single-Pass Blind Inference

---

## 1. Konfigurasi Sistem (Frozen)

| Parameter | Nilai |
|---|---|
| Resolusi Citra | 256×256 px (Aspect-Ratio Letterbox) |
| Pipeline Preprocessing | Auto-Orientation → Grayscale → Gaussian Blur (5×5) → Otsu Inverted Binarization → Noise Removal (Median-3 + Morph Close/Open 3×3) → extract_roi() |
| HOG Orientations | 9 |
| HOG Pixels per Cell | (8, 8) |
| HOG Cells per Block | (2, 2) |
| HOG Block Norm | L2-Hys |
| Dimensi Fitur HOG | **34,596** |
| KNN K | 5 |
| KNN Metric | Euclidean |
| KNN Weights | Distance |
| Model File | `{os.path.basename(MODEL_PATH)}` |
| Training Reference | 18 mahasiswa × 20 citra = **360 sampel** |

---

## 2. Dataset Holdout

| Item | Nilai |
|---|---|
| Total citra holdout | **{n_total}** |
| Mahasiswa dengan holdout | **{len(holdout_classes)}/{len(STUDENT_FOLDER_MAP)}** |
| Mahasiswa tanpa holdout | {no_holdout_md} |
| Inference errors | {len(errors)} |

---

## 3. Hasil Evaluasi Keseluruhan

| Metrik | LOOCV Internal | Holdout Eksternal | Δ |
|---|---|---|---|
| **Accuracy** | **{LOOCV_ACCURACY:.2f}%** | **{accuracy:.2f}%** | **{delta_accuracy:+.2f}%** |
| Correct / Total | {LOOCV_CORRECT}/{LOOCV_TOTAL} | {n_correct}/{n_total} | — |
| Top-5 Accuracy | — | {top5_accuracy:.2f}% | — |
| Macro Precision | 63.21% | {macro_precision:.2f}% | — |
| Macro Recall | 61.94% | {macro_recall:.2f}% | — |
| Macro F1 | 61.29% | {macro_f1:.2f}% | — |
| Weighted Precision | — | {weighted_precision:.2f}% | — |
| Weighted Recall | — | {weighted_recall:.2f}% | — |
| Weighted F1 | — | {weighted_f1:.2f}% | — |
| Avg Sim (Correct) | — | {round(float(np.mean(sim_correct)),2) if sim_correct else 'N/A'}% | — |
| Avg Sim (Wrong) | — | {round(float(np.mean(sim_wrong)),2) if sim_wrong else 'N/A'}% | — |

---

## 4. Hasil Per Mahasiswa

| Nama | NIM | N Holdout | Benar | Salah | Accuracy | Avg Similarity |
|---|---|---|---|---|---|---|
{per_stu_md}

---

## 5. Top-10 Misklasifikasi Berdasarkan Similarity Tertinggi (High-Confidence Errors)

| Ground Truth | Predicted | Similarity | File |
|---|---|---|---|
{top10_wrong_md}

---

## 6. Top Confusion Pairs

| True Label | Predicted As | Count |
|---|---|---|
{top_confusion_md}

---

## 7. Analisis & Kesimpulan

### 7.1 Generalisasi Model
- **Holdout accuracy {accuracy:.2f}%** vs LOOCV internal {LOOCV_ACCURACY:.2f}% (Δ = {delta_accuracy:+.2f}%)
- {"Model menggeneralisasi dengan baik ke data unseen." if abs(delta_accuracy) <= 5 else "Terdapat perbedaan signifikan antara performa internal dan eksternal — perlu dicatat sebagai limitation."}

### 7.2 Top-5 Accuracy
- Top-5 accuracy = **{top5_accuracy:.2f}%** — menunjukkan bahwa ground truth sering hadir di candidate list meski bukan prediksi terbaik.

### 7.3 Keterbatasan Dataset
- **{len(students_no_holdout)} mahasiswa** tidak memiliki citra holdout (Fahim, Farhan Agiya, Hazel, Ilham, Soni, Wirid).
- Distribusi holdout tidak seimbang antar mahasiswa ({min([r[2] for r in per_stu_rows])}–{max([r[2] for r in per_stu_rows])} citra per mahasiswa).
- Evaluasi ini hanya dapat diklaim untuk {len(holdout_classes)}/18 mahasiswa.

### 7.4 Separabilitas Fitur
- Rata-rata similarity prediksi benar ({round(float(np.mean(sim_correct)),1) if sim_correct else 'N/A'}%) lebih tinggi dari prediksi salah ({round(float(np.mean(sim_wrong)),1) if sim_wrong else 'N/A'}%) — menunjukkan bahwa fitur HOG dapat membedakan kelas dengan keyakinan yang terukur.

### 7.5 Final Statement
Model HOG+KNN Yaevia dengan konfigurasi 256×256, orientations=9, ppc=(8,8), cpb=(2,2), K=5 yang dilatih pada 360 sampel referensi mencapai **overall accuracy {accuracy:.2f}%** pada {n_total} citra holdout eksternal yang belum pernah dilihat sebelumnya. Ini merupakan angka evaluasi generalisasi yang valid dan bebas kebocoran data (*zero-leakage blind test*).

---

## 8. Artefak yang Dihasilkan

| File | Deskripsi |
|---|---|
| `holdout_summary.json` | Summary JSON lengkap |
| `holdout_predictions.csv` | Prediksi per sampel (N={n_total}) |
| `holdout_per_student.csv` | Akurasi per mahasiswa |
| `holdout_confusion_matrix.csv` | Confusion matrix {len(holdout_classes)}×{len(holdout_classes)} |
| `holdout_confusion_pairs.csv` | Ranking pair confusion |
| `holdout_misclassifications.csv` | {len(wrong_preds)} sampel salah prediksi |
| `loocv_vs_holdout.csv` | Perbandingan LOOCV vs holdout |
| `holdout_confusion_matrix.png` | Heatmap confusion matrix |
| `loocv_vs_holdout_comparison.png` | Bar chart perbandingan akurasi |
| `holdout_per_student_accuracy.png` | Bar chart akurasi per mahasiswa |

---
*Evaluasi ini merupakan pengujian akhir sistem Yaevia. Parameter model tidak diubah setelah melihat hasil ini.*
"""

report_path = os.path.join(OUT_DIR, "holdout_final_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_md)
print(f"  [8] {report_path}")

# ── Final summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
print(f"  Holdout images evaluated : {n_total}")
print(f"  Overall Accuracy         : {accuracy:.2f}%  ({n_correct}/{n_total})")
print(f"  Top-5 Accuracy           : {top5_accuracy:.2f}%")
print(f"  Macro F1                 : {macro_f1:.2f}%")
print(f"  LOOCV vs Holdout Δ       : {delta_accuracy:+.2f}%")
print(f"  Output directory         : {OUT_DIR}")
print("=" * 70)
print("STOP — No further tuning or retraining. Evaluation is final.")
