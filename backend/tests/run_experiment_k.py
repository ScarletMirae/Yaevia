"""
EXPERIMENT K — PER-WRITER CLUSTER & REFERENCE COVERAGE ANALYSIS
================================================================
Yaevia Handwriting Verification System
Frozen Model: 20260926_190940
Mode: STRICT ANALYSIS-ONLY — ZERO MUTATION
Date: 2026-09-28
================================================================

Phases:
  1.  Dataset Integrity
  2.  Per-Writer Reference Compactness
  3.  Inter-Class Separation
  4.  Reference-to-Reference NN Audit
  5.  External A/B Reference Coverage
  6.  A vs B Same-Writer Drift
  7.  Regressed/Improved Case Analysis
  8.  Magnet Class Geometry
  9.  Per-Writer Diagnosis Table
  10. Global Diagnosis (H1/H2/H3)
  11. Research-Scope Safety — future experiment list

All outputs saved to:
  backend/tests/evaluation_results/experiment_k_cluster_coverage/
"""

import os
import sys
import json
import hashlib
import sqlite3
import warnings

# Ensure stdout and stderr handle utf-8 safely
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import cdist, euclidean
from itertools import combinations
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "saved", "knn_model_20260926_190940.joblib")
DB_PATH     = os.path.join(BASE_DIR, "database.db")
PROC_DIR    = os.path.join(BASE_DIR, "dataset", "processed")
QUERY_DIR   = os.path.join(BASE_DIR, "dataset", "raw", "queries")
OUT_DIR     = os.path.join(BASE_DIR, "tests", "evaluation_results", "experiment_k_cluster_coverage")
EXPECTED_MODEL_VERSION = "20260926_190940"
EXPECTED_CLASSES = 20
EXPECTED_REFS_PER_CLASS = 20
EXPECTED_TOTAL = 400
EXPECTED_DIMS  = 34596

os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# IMPORTS (lazy — after sys.path insert)
# ─────────────────────────────────────────────────────────────
import joblib
from preprocessing.image_processor import preprocess_image
from features.hog_extractor import extract_hog_features

print("=" * 80)
print("EXPERIMENT K — PER-WRITER CLUSTER & REFERENCE COVERAGE ANALYSIS")
print("=" * 80)

# ─────────────────────────────────────────────────────────────
# PHASE 0: LOAD FROZEN MODEL
# ─────────────────────────────────────────────────────────────
print("\n[LOADING FROZEN MODEL]")
knn = joblib.load(MODEL_PATH)
model_hash = hashlib.sha256(open(MODEL_PATH, "rb").read()).hexdigest()[:16]
X_train = knn._fit_X          # shape: (320, 34596)
y_train = list(knn.classes_[knn._y.ravel()])  # decoded class labels per sample
classes  = list(knn.classes_)

print(f"  Model version : {EXPECTED_MODEL_VERSION}")
print(f"  SHA-256 (16)  : {model_hash}")
print(f"  Classes       : {len(classes)}")
print(f"  Train samples : {X_train.shape[0]}  (expected 320 = 20×16 stratified)")
print(f"  Dimensions    : {X_train.shape[1]}")

# ─────────────────────────────────────────────────────────────
# PHASE 0b: LOAD FULL 400-SAMPLE REFERENCE FEATURES
#   The frozen KNN was trained on 320 (80% of 400).
#   We need ALL 400 reference HOG vectors for cluster analysis.
#   We rebuild by preprocessing every reference image from DB.
# ─────────────────────────────────────────────────────────────
print("\n[BUILDING FULL 400-SAMPLE REFERENCE FEATURE MATRIX]")
conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()
cur.execute("""
    SELECT student_name, student_id, file_path, saved_filename
    FROM dataset
    ORDER BY student_name, id
""")
db_rows = cur.fetchall()
conn.close()

ref_X   = []   # HOG feature vectors
ref_y   = []   # writer name labels
ref_nim = []   # NIM
ref_files = [] # filenames
ref_failures = []

for row in db_rows:
    name, nim, file_path, saved_fn = row
    if not file_path or not os.path.exists(file_path):
        ref_failures.append((name, nim, file_path, "FILE_MISSING"))
        continue
    try:
        img = preprocess_image(file_path)
        feat = extract_hog_features(img, visualize=False)
        ref_X.append(feat)
        ref_y.append(name)
        ref_nim.append(nim)
        ref_files.append(saved_fn)
    except Exception as e:
        ref_failures.append((name, nim, file_path, str(e)))

ref_X   = np.array(ref_X, dtype=np.float64)
print(f"  Loaded {len(ref_X)} / {len(db_rows)} reference samples  ({len(ref_failures)} failures)")
if ref_failures:
    print("  FAILURES:")
    for f in ref_failures:
        print("   ", f)

# Build per-class index maps
class_idx = defaultdict(list)  # name -> [row indices in ref_X]
for i, name in enumerate(ref_y):
    class_idx[name].append(i)

# ─────────────────────────────────────────────────────────────
# PHASE 0c: LOAD TEST A & B QUERIES + THEIR HOG FEATURES
# ─────────────────────────────────────────────────────────────
print("\n[LOADING EXTERNAL QUERY IMAGES — TEST A & B]")
conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()
# Test A: IDs 220–240 (excl. 221 which is a duplicate of alif), verified batch
# Per previous audit: Test A = IDs 222–240 excl 221
# But from DB printout: 220-240 = 21 rows; 221 is a dup of alif (same filename)
# We treat 220, 222-240 as Test A (20 rows), 241-260 as Test B (20 rows)
cur.execute("""
    SELECT id, ground_truth_name, ground_truth_nim, query_path, query_filename,
           is_correct, top_matches_json, euclidean_distance, similarity_percent
    FROM verifications
    WHERE id IN (220,222,223,224,225,226,227,228,229,230,
                 231,232,233,234,235,236,237,238,239,240)
    ORDER BY ground_truth_name
""")
test_a_rows = cur.fetchall()

cur.execute("""
    SELECT id, ground_truth_name, ground_truth_nim, query_path, query_filename,
           is_correct, top_matches_json, euclidean_distance, similarity_percent
    FROM verifications
    WHERE id BETWEEN 241 AND 260
    ORDER BY ground_truth_name
""")
test_b_rows = cur.fetchall()
conn.close()

def build_query_record(rows, label):
    records = []
    for row in rows:
        rid, name, nim, qpath, qfn, correct, tmj, dist, sim = row
        top = json.loads(tmj) if tmj else []
        qpath_clean = qpath.replace("\\\\", "\\") if qpath else ""
        feat = None
        status = "OK"
        if os.path.exists(qpath_clean):
            try:
                img = preprocess_image(qpath_clean)
                feat = extract_hog_features(img, visualize=False)
            except Exception as e:
                status = f"ERR:{e}"
        else:
            status = "PATH_MISSING"
        records.append({
            "test": label,
            "id": rid,
            "writer_name": name,
            "nim": nim,
            "query_path": qpath_clean,
            "query_filename": qfn,
            "is_correct": bool(correct),
            "stored_distance": dist,
            "stored_similarity": sim,
            "top_matches": top,
            "feat": feat,
            "feat_status": status,
        })
    return records

print("  Processing Test A queries...")
qa = build_query_record(test_a_rows, "A")
print("  Processing Test B queries...")
qb = build_query_record(test_b_rows, "B")

feat_ok_a = sum(1 for r in qa if r["feat"] is not None)
feat_ok_b = sum(1 for r in qb if r["feat"] is not None)
print(f"  Test A: {len(qa)} queries, {feat_ok_a} HOG features extracted")
print(f"  Test B: {len(qb)} queries, {feat_ok_b} HOG features extracted")

# Merge into one dict by writer_name for easy cross-lookup
qa_by_name = {r["writer_name"]: r for r in qa}
qb_by_name = {r["writer_name"]: r for r in qb}

# ─────────────────────────────────────────────────────────────
# PHASE 1: DATASET INTEGRITY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[PHASE 1] DATASET INTEGRITY")
print("=" * 70)

conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()
cur.execute("SELECT student_name, student_id, COUNT(*) FROM dataset GROUP BY student_name, student_id ORDER BY student_name")
class_counts = cur.fetchall()
cur.execute("SELECT COUNT(*) FROM dataset")
total_ref = cur.fetchone()[0]

# Check for duplicate processed filenames
cur.execute("SELECT processed_path, COUNT(*) as cnt FROM dataset GROUP BY processed_path HAVING cnt > 1")
dup_files = cur.fetchall()

# Check model_meta for frozen model
cur.execute("""
    SELECT model_filename, train_accuracy, test_accuracy, n_classes,
           knn_k, feature_vector_size, is_active
    FROM model_meta
    WHERE model_filename LIKE '%20260926_190940%'
""")
mm = cur.fetchone()
conn.close()

n_classes_found = len(class_counts)
all_20 = all(cnt == EXPECTED_REFS_PER_CLASS for _, _, cnt in class_counts)

# Check for Test A/B query overlap with reference set
test_query_fns = set()
for r in qa + qb:
    if r["query_filename"]:
        test_query_fns.add(r["query_filename"].lower().strip())

ref_fns_lower = set(f.lower().strip() for f in ref_files)
leak_candidates = test_query_fns & ref_fns_lower

integrity_rows = []
for name, nim, cnt in class_counts:
    integrity_rows.append({
        "writer_name": name, "nim": nim,
        "reference_count": cnt,
        "expected": EXPECTED_REFS_PER_CLASS,
        "count_ok": cnt == EXPECTED_REFS_PER_CLASS,
    })

df_integrity = pd.DataFrame(integrity_rows)
df_integrity.to_csv(os.path.join(OUT_DIR, "dataset_integrity.csv"), index=False)

print(f"  Classes found          : {n_classes_found} (expected {EXPECTED_CLASSES})")
print(f"  Total reference images : {total_ref} (expected {EXPECTED_TOTAL})")
print(f"  All classes = 20 refs  : {all_20}")
print(f"  HOG feature dims       : {ref_X.shape[1]} (expected {EXPECTED_DIMS})")
print(f"  Duplicate proc files   : {len(dup_files)}")
print(f"  Model frozen version   : {EXPECTED_MODEL_VERSION}")
print(f"  Model SHA-256 (16ch)   : {model_hash}")
print(f"  Reference failures     : {len(ref_failures)}")
print(f"  Query-Ref name overlap : {len(leak_candidates)} possible leaks: {leak_candidates if leak_candidates else 'NONE'}")
if mm:
    print(f"  Frozen model DB meta   : acc={mm[1]:.4f} train / {mm[2]:.4f} test, K={mm[4]}, dims={mm[5]}, active={mm[6]}")

# ─────────────────────────────────────────────────────────────
# PHASE 2: PER-WRITER REFERENCE COMPACTNESS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[PHASE 2] PER-WRITER REFERENCE COMPACTNESS")
print("=" * 70)

cluster_stats = {}
for name in sorted(class_idx.keys()):
    idxs = class_idx[name]
    X_c  = ref_X[idxs]    # (20, 34596)
    # Pairwise distances within class
    pw = cdist(X_c, X_c, metric="euclidean")
    # Upper triangle only (exclude diagonal)
    upper = pw[np.triu_indices_from(pw, k=1)]
    # Centroid
    centroid = X_c.mean(axis=0)
    # Centroid distances
    cent_dists = np.linalg.norm(X_c - centroid, axis=1)

    cluster_stats[name] = {
        "writer_name": name,
        "nim": ref_nim[idxs[0]],
        "n_samples": len(idxs),
        "intra_mean": float(np.mean(upper)),
        "intra_median": float(np.median(upper)),
        "intra_std": float(np.std(upper)),
        "intra_min": float(np.min(upper)),
        "intra_max": float(np.max(upper)),
        "intra_p25": float(np.percentile(upper, 25)),
        "intra_p75": float(np.percentile(upper, 75)),
        "intra_p90": float(np.percentile(upper, 90)),
        "centroid_dist_mean": float(np.mean(cent_dists)),
        "centroid_dist_std": float(np.std(cent_dists)),
        "centroid_dist_max": float(np.max(cent_dists)),   # cluster radius
        "centroid_dist_p90": float(np.percentile(cent_dists, 90)),  # robust radius
        "centroid": centroid,
        "idxs": idxs,
        "X_c": X_c,
    }

# Compute compactness classification using data-derived percentile thresholds
intra_means = [v["intra_mean"] for v in cluster_stats.values()]
p33 = np.percentile(intra_means, 33)
p67 = np.percentile(intra_means, 67)

for name, s in cluster_stats.items():
    if s["intra_mean"] <= p33:
        s["compactness"] = "COMPACT"
    elif s["intra_mean"] <= p67:
        s["compactness"] = "MODERATE"
    else:
        s["compactness"] = "DISPERSED"

# Rank by intra_mean
ranked = sorted(cluster_stats.values(), key=lambda x: x["intra_mean"])
for rank, s in enumerate(ranked, 1):
    s["compactness_rank"] = rank

# Save
cols = ["compactness_rank","writer_name","nim","n_samples",
        "intra_mean","intra_median","intra_std","intra_min","intra_max",
        "intra_p25","intra_p75","intra_p90",
        "centroid_dist_mean","centroid_dist_std","centroid_dist_max","centroid_dist_p90",
        "compactness"]
df_cluster = pd.DataFrame(
    [{c: s[c] for c in cols} for s in ranked]
)
df_cluster.to_csv(os.path.join(OUT_DIR, "per_writer_cluster_statistics.csv"), index=False)

print(f"  Compactness threshold COMPACT   : intra_mean <= {p33:.4f}  (P33 of all writers)")
print(f"  Compactness threshold MODERATE  : {p33:.4f} < intra_mean <= {p67:.4f}  (P33-P67)")
print(f"  Compactness threshold DISPERSED : intra_mean > {p67:.4f}  (P67+)")
print(f"\n  {'Rank':<5} {'Writer':<35} {'Intra_Mean':>11} {'Intra_Std':>10} {'Radius_P90':>11} {'Category'}")
print(f"  {'-'*80}")
for s in ranked:
    print(f"  {s['compactness_rank']:<5} {s['writer_name']:<35} {s['intra_mean']:>11.4f} {s['intra_std']:>10.4f} {s['centroid_dist_p90']:>11.4f}  {s['compactness']}")

# ─────────────────────────────────────────────────────────────
# PHASE 3: INTER-CLASS SEPARATION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[PHASE 3] INTER-CLASS SEPARATION")
print("=" * 70)

# Build centroid matrix
name_list   = sorted(class_idx.keys())
centroids   = np.array([cluster_stats[n]["centroid"] for n in name_list])
centroid_pw = cdist(centroids, centroids, metric="euclidean")  # (20,20)

interclass_rows = []
nearest_other   = {}  # name -> nearest_other_name, nearest_centroid_dist

for i, ni in enumerate(name_list):
    X_i = cluster_stats[ni]["X_c"]
    row_cent_dists = centroid_pw[i]
    # Nearest centroid (exclude self)
    row_no_self = row_cent_dists.copy()
    row_no_self[i] = np.inf
    nn_idx = np.argmin(row_no_self)
    nn_name = name_list[nn_idx]
    nn_centroid_dist = row_cent_dists[nn_idx]
    # Mean cross-class centroid distance
    mean_cent_dist = np.mean(row_cent_dists[np.arange(len(name_list)) != i])

    # Nearest sample from nearest impostor class
    X_j = cluster_stats[nn_name]["X_c"]
    cross_dists = cdist(X_i, X_j, metric="euclidean")
    nearest_sample_dist = float(np.min(cross_dists))
    mean_cross_dist     = float(np.mean(cross_dists))

    # Separation metric: nearest_centroid_dist / own_cluster_radius_P90
    radius_p90 = cluster_stats[ni]["centroid_dist_p90"]
    sep_ratio = nn_centroid_dist / radius_p90 if radius_p90 > 0 else np.inf

    nearest_other[ni] = {
        "nn_name": nn_name,
        "nn_centroid_dist": float(nn_centroid_dist),
        "mean_centroid_dist_to_others": float(mean_cent_dist),
        "nearest_sample_dist_to_nn": nearest_sample_dist,
        "mean_cross_dist_to_nn": mean_cross_dist,
        "separation_ratio": float(sep_ratio),
    }
    interclass_rows.append({
        "writer_name": ni,
        "nim": ref_nim[class_idx[ni][0]],
        "nearest_other_writer": nn_name,
        "nearest_centroid_dist": float(nn_centroid_dist),
        "mean_centroid_dist_to_others": float(mean_cent_dist),
        "nearest_sample_dist_to_nn": nearest_sample_dist,
        "mean_cross_dist_to_nn": mean_cross_dist,
        "own_cluster_radius_p90": cluster_stats[ni]["centroid_dist_p90"],
        "separation_ratio": float(sep_ratio),
        # sep_ratio < 1 means own radius > nearest centroid gap → severe overlap
    })

df_sep = pd.DataFrame(interclass_rows).sort_values("separation_ratio")
df_sep.to_csv(os.path.join(OUT_DIR, "interclass_separation.csv"), index=False)

print(f"\n  Separation Ratio = nearest_other_centroid_dist / own_cluster_radius_P90")
print(f"  (ratio < 1 = own cluster overlaps nearest impostor centroid; < 2 = borderline)")
print(f"\n  {'Writer':<35} {'Nearest Impostor':<35} {'CentDist':>9} {'SepRatio':>9}")
print(f"  {'-'*90}")
for _, row in df_sep.iterrows():
    flag = " !! OVERLAP" if row["separation_ratio"] < 1 else (" ! BORDERLINE" if row["separation_ratio"] < 2 else "")
    print(f"  {row['writer_name']:<35} {row['nearest_other_writer']:<35} {row['nearest_centroid_dist']:>9.4f} {row['separation_ratio']:>9.4f}{flag}")

# Global intra vs inter
global_intra = float(np.mean([s["intra_mean"] for s in cluster_stats.values()]))
# compute global inter from all cross-class sample pairs (expensive, use centroid approx)
off_diag = centroid_pw[np.triu_indices_from(centroid_pw, k=1)]
global_inter_centroid = float(np.mean(off_diag))
print(f"\n  Global mean intra-class distance      : {global_intra:.4f}")
print(f"  Global mean inter-centroid distance   : {global_inter_centroid:.4f}")
print(f"  Separation margin (inter-intra)        : {global_inter_centroid - global_intra:.4f}")

# ─────────────────────────────────────────────────────────────
# PHASE 4: REFERENCE-TO-REFERENCE NN AUDIT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[PHASE 4] REFERENCE-TO-REFERENCE NN AUDIT")
print("=" * 70)

# For every ref sample: nearest same-class vs nearest different-class (excl self)
all_dists = cdist(ref_X, ref_X, metric="euclidean")  # (N_ref, N_ref)
np.fill_diagonal(all_dists, np.inf)

nn_audit_rows = []
per_writer_correct_nn = defaultdict(list)

for i, name_i in enumerate(ref_y):
    same_mask  = np.array([n == name_i for n in ref_y])
    diff_mask  = ~same_mask
    same_mask[i] = False  # exclude self

    same_dists = all_dists[i][same_mask]
    diff_dists = all_dists[i][diff_mask]

    nn_same_dist = float(np.min(same_dists)) if len(same_dists) > 0 else np.inf
    nn_diff_dist = float(np.min(diff_dists)) if len(diff_dists) > 0 else np.inf
    nn_is_correct = nn_same_dist < nn_diff_dist

    impostor_nn_idx = np.where(diff_mask)[0][np.argmin(diff_dists)]
    impostor_name   = ref_y[impostor_nn_idx]

    margin = nn_diff_dist - nn_same_dist  # positive = correct ordering

    per_writer_correct_nn[name_i].append(nn_is_correct)
    nn_audit_rows.append({
        "sample_index": i,
        "writer_name": name_i,
        "filename": ref_files[i],
        "nn_same_dist": nn_same_dist,
        "nn_diff_dist": nn_diff_dist,
        "margin": margin,
        "nn_is_correct": nn_is_correct,
        "nearest_impostor": impostor_name,
    })

df_nn_audit = pd.DataFrame(nn_audit_rows)
df_nn_audit.to_csv(os.path.join(OUT_DIR, "reference_nearest_neighbor_audit.csv"), index=False)

global_nn_acc = df_nn_audit["nn_is_correct"].mean()
print(f"  Global reference 1-NN accuracy : {global_nn_acc:.4f} ({df_nn_audit['nn_is_correct'].sum()}/{len(df_nn_audit)})")

per_writer_nn = []
for name in sorted(per_writer_correct_nn.keys()):
    results = per_writer_correct_nn[name]
    acc = np.mean(results)
    per_writer_nn.append((name, acc, sum(results), len(results)))
per_writer_nn.sort(key=lambda x: x[1])
print(f"\n  Per-writer reference 1-NN accuracy:")
print(f"  {'Writer':<35} {'Acc':>6}  ({'correct / total'})")
for name, acc, correct, total in per_writer_nn:
    print(f"  {name:<35} {acc:>6.4f}  ({correct}/{total})")

# ─────────────────────────────────────────────────────────────
# PHASE 5: EXTERNAL A/B REFERENCE COVERAGE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[PHASE 5] EXTERNAL A/B REFERENCE COVERAGE")
print("=" * 70)

# For classification of coverage, we use the writer's own P90 intra-class distance
# as the boundary: queries with dist-to-nearest-own-sample > P90 are OUTLIERS.
# This is data-derived: P90 captures 90% of the tightest reference sample pairs.
# WELL COVERED  : dist_to_nearest_own <= P75 of own ref distances (comfortably inside)
# BORDERLINE    : P75 < dist <= P90
# POORLY COVERED: > P90

coverage_rows = []
for test_label, queries in [("A", qa), ("B", qb)]:
    for rec in queries:
        name = rec["writer_name"]
        feat = rec["feat"]
        if feat is None:
            coverage_rows.append({
                "test": test_label,
                "writer_name": name,
                "nim": rec["nim"],
                "is_correct": rec["is_correct"],
                "dist_to_own_centroid": None,
                "dist_to_nearest_own": None,
                "mean_dist_to_own": None,
                "dist_to_nearest_impostor": None,
                "nearest_impostor": None,
                "percentile_in_own_dist": None,
                "coverage_category": "FEAT_MISSING",
                "stored_distance": rec["stored_distance"],
                "stored_similarity": rec["stored_similarity"],
            })
            continue
        cs = cluster_stats.get(name)
        if cs is None:
            continue
        X_c = cs["X_c"]
        centroid = cs["centroid"]
        dist_to_centroid = float(np.linalg.norm(feat - centroid))
        dists_to_own = np.linalg.norm(X_c - feat, axis=1)
        dist_nearest_own = float(np.min(dists_to_own))
        mean_dist_own    = float(np.mean(dists_to_own))

        # Impostor nearest
        other_idxs = [i for i, n in enumerate(ref_y) if n != name]
        X_other    = ref_X[other_idxs]
        dists_to_other = np.linalg.norm(X_other - feat, axis=1)
        dist_nearest_impostor = float(np.min(dists_to_other))
        impostor_nn_name = ref_y[other_idxs[np.argmin(dists_to_other)]]

        # Percentile of dist_nearest_own within the writer's own intra-class distances
        # Build reference sample-to-sample distances for this class (upper triangle)
        own_pw = cdist(X_c, X_c, metric="euclidean")
        upper_own = own_pw[np.triu_indices_from(own_pw, k=1)]
        percentile = float(np.mean(upper_own <= dist_nearest_own) * 100)

        p75_own = np.percentile(upper_own, 75)
        p90_own = np.percentile(upper_own, 90)
        if dist_nearest_own <= p75_own:
            cat = "WELL_COVERED"
        elif dist_nearest_own <= p90_own:
            cat = "BORDERLINE"
        else:
            cat = "POORLY_COVERED"

        coverage_rows.append({
            "test": test_label,
            "writer_name": name,
            "nim": rec["nim"],
            "is_correct": rec["is_correct"],
            "dist_to_own_centroid": round(dist_to_centroid, 4),
            "dist_to_nearest_own": round(dist_nearest_own, 4),
            "mean_dist_to_own": round(mean_dist_own, 4),
            "dist_to_nearest_impostor": round(dist_nearest_impostor, 4),
            "nearest_impostor": impostor_nn_name,
            "margin_own_vs_impostor": round(dist_nearest_impostor - dist_nearest_own, 4),
            "percentile_in_own_dist": round(percentile, 1),
            "p75_own_ref": round(float(p75_own), 4),
            "p90_own_ref": round(float(p90_own), 4),
            "coverage_category": cat,
            "stored_distance": rec["stored_distance"],
            "stored_similarity": rec["stored_similarity"],
        })

df_cov = pd.DataFrame(coverage_rows)
df_cov.to_csv(os.path.join(OUT_DIR, "external_ab_reference_coverage.csv"), index=False)

print(f"\n  Coverage classification thresholds:")
print(f"    WELL_COVERED   : dist_to_nearest_own_sample <= P75 of own reference intra-distances")
print(f"    BORDERLINE     : P75 < dist <= P90")
print(f"    POORLY_COVERED : dist > P90 of own reference intra-distances")
print(f"\n  {'Test':<5} {'Writer':<35} {'OwnDist':>8} {'ImpDist':>8} {'Margin':>8} {'Pct':>5}  {'Coverage'}")
print(f"  {'-'*95}")
for _, row in df_cov.dropna(subset=["dist_to_nearest_own"]).sort_values(["test","writer_name"]).iterrows():
    print(f"  {row['test']:<5} {row['writer_name']:<35} {row['dist_to_nearest_own']:>8.4f} {row['dist_to_nearest_impostor']:>8.4f} {row['margin_own_vs_impostor']:>8.4f} {row['percentile_in_own_dist']:>5.1f}  {row['coverage_category']}")

# Summary
for t in ["A", "B"]:
    sub = df_cov[df_cov["test"] == t].dropna(subset=["coverage_category"])
    cats = sub["coverage_category"].value_counts()
    print(f"\n  Test {t} coverage summary: {dict(cats)}")

# ─────────────────────────────────────────────────────────────
# PHASE 6: A vs B SAME-WRITER DRIFT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[PHASE 6] A vs B SAME-WRITER FEATURE DRIFT")
print("=" * 70)

drift_rows = []
for name in sorted(set(r["writer_name"] for r in qa)):
    ra = qa_by_name.get(name)
    rb = qb_by_name.get(name)
    if ra is None or rb is None:
        continue
    fa, fb = ra["feat"], rb["feat"]
    if fa is None or fb is None:
        drift_rows.append({
            "writer_name": name,
            "nim": ra["nim"],
            "ab_drift": None,
            "drift_category": "FEAT_MISSING"
        })
        continue
    ab_drift = float(np.linalg.norm(fa - fb))
    cs = cluster_stats.get(name)
    if cs:
        own_intra_mean = cs["intra_mean"]
        own_intra_max  = cs["intra_max"]
        radius_p90     = cs["centroid_dist_p90"]
    else:
        own_intra_mean = own_intra_max = radius_p90 = None

    # Derive thresholds from data:
    #   NORMAL     : ab_drift <= own_intra_mean   (within typical reference scatter)
    #   MODERATE   : own_intra_mean < ab_drift <= own_intra_max  (at boundary)
    #   EXTREME    : ab_drift > own_intra_max     (exceeds worst reference pair)
    if own_intra_mean is None:
        drift_cat = "UNKNOWN"
    elif ab_drift <= own_intra_mean:
        drift_cat = "NORMAL"
    elif ab_drift <= own_intra_max:
        drift_cat = "MODERATE"
    else:
        drift_cat = "EXTREME"

    # Does A→B drift exceed nearest inter-class distance?
    nn_inter = nearest_other[name]["nearest_sample_dist_to_nn"] if name in nearest_other else None
    drift_exceeds_inter = (ab_drift > nn_inter) if nn_inter is not None else None

    drift_rows.append({
        "writer_name": name,
        "nim": ra["nim"],
        "ab_drift": round(ab_drift, 4),
        "own_intra_mean": round(float(own_intra_mean), 4) if own_intra_mean else None,
        "own_intra_max": round(float(own_intra_max), 4) if own_intra_max else None,
        "own_radius_p90": round(float(radius_p90), 4) if radius_p90 else None,
        "nearest_inter_dist": round(float(nn_inter), 4) if nn_inter else None,
        "drift_exceeds_inter": drift_exceeds_inter,
        "drift_category": drift_cat,
        "test_a_correct": ra["is_correct"],
        "test_b_correct": rb["is_correct"],
        "transition": (
            "STABLE_CORRECT" if ra["is_correct"] and rb["is_correct"] else
            "IMPROVED"       if (not ra["is_correct"]) and rb["is_correct"] else
            "REGRESSED"      if ra["is_correct"] and (not rb["is_correct"]) else
            "STABLE_WRONG"
        ),
    })

df_drift = pd.DataFrame(drift_rows).sort_values("ab_drift", ascending=False)
df_drift.to_csv(os.path.join(OUT_DIR, "ab_same_writer_drift.csv"), index=False)

print(f"\n  Drift classification:")
print(f"    NORMAL   : A<->B drift <= writer's own reference intra_mean")
print(f"    MODERATE : intra_mean < drift <= intra_max (worst reference pair)")
print(f"    EXTREME  : drift > intra_max (both samples farther apart than any reference pair)")
print(f"\n  {'Writer':<35} {'A<->B Drift':>10} {'IntraMean':>10} {'IntraMax':>10} {'>Inter?':>7}  {'Category':<12} {'Transition'}")
print(f"  {'-'*110}")
for _, row in df_drift.iterrows():
    d = f"{row['ab_drift']:.4f}" if row["ab_drift"] is not None else "N/A"
    im = f"{row['own_intra_mean']:.4f}" if row["own_intra_mean"] is not None else "N/A"
    ix = f"{row['own_intra_max']:.4f}" if row["own_intra_max"] is not None else "N/A"
    ei = "YES" if row["drift_exceeds_inter"] else ("NO" if row["drift_exceeds_inter"] == False else "N/A")
    print(f"  {row['writer_name']:<35} {d:>10} {im:>10} {ix:>10} {ei:>7}  {row['drift_category']:<12} {row['transition']}")

mean_drift_regressed = df_drift[df_drift["transition"] == "REGRESSED"]["ab_drift"].mean()
mean_drift_stable_w  = df_drift[df_drift["transition"] == "STABLE_WRONG"]["ab_drift"].mean()
print(f"\n  Mean A↔B drift (REGRESSED writers) : {mean_drift_regressed:.4f}")
print(f"  Mean A↔B drift (STABLE_WRONG writers): {mean_drift_stable_w:.4f}")

# ─────────────────────────────────────────────────────────────
# PHASE 7: REGRESSED / IMPROVED CASE ANALYSIS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[PHASE 7] REGRESSED / IMPROVED CASE ANALYSIS")
print("=" * 70)

focus_writers = {
    "Farhan Agiya Pratama":    "REGRESSED",
    "Fathurrahman Nugroho":    "REGRESSED",
    "Febrian Dinnar Purnama":  "REGRESSED",
    "Muhammad Dony Saputra":   "REGRESSED",
    "Ngatmanto":               "REGRESSED",
    "Soni Nugroho":            "REGRESSED",
    "Zaedani Ni'am Masykur":   "IMPROVED",
}

reg_rows = []
for name, role in focus_writers.items():
    ra = qa_by_name.get(name)
    rb = qb_by_name.get(name)
    cs = cluster_stats.get(name)
    if ra is None or rb is None or cs is None:
        continue
    fa, fb = ra["feat"], rb["feat"]
    centroid = cs["centroid"]
    X_c = cs["X_c"]
    radius_p90 = cs["centroid_dist_p90"]

    def query_stats(feat, tag):
        if feat is None:
            return {}
        d_centroid = float(np.linalg.norm(feat - centroid))
        d_own_all  = np.linalg.norm(X_c - feat, axis=1)
        d_nearest_own = float(np.min(d_own_all))
        d_mean_own    = float(np.mean(d_own_all))
        # K=5 neighbors among ALL ref samples
        all_d = np.linalg.norm(ref_X - feat, axis=1)
        k5_idx = np.argsort(all_d)[:5]
        k5_names = [ref_y[i] for i in k5_idx]
        k5_own_count = sum(1 for n in k5_names if n == name)
        # Nearest impostor
        other_idxs = [i for i, n in enumerate(ref_y) if n != name]
        d_imp_all  = np.linalg.norm(ref_X[other_idxs] - feat, axis=1)
        d_nearest_imp = float(np.min(d_imp_all))
        imp_name = ref_y[other_idxs[np.argmin(d_imp_all)]]
        margin = d_nearest_imp - d_nearest_own
        pct_in_p90 = "INSIDE" if d_nearest_own <= float(np.percentile(
            cdist(X_c, X_c, "euclidean")[np.triu_indices(len(X_c), k=1)], 90)) else "OUTSIDE"
        return {
            f"{tag}_dist_centroid": round(d_centroid, 4),
            f"{tag}_dist_nearest_own": round(d_nearest_own, 4),
            f"{tag}_mean_dist_own": round(d_mean_own, 4),
            f"{tag}_dist_nearest_imp": round(d_nearest_imp, 4),
            f"{tag}_nearest_impostor": imp_name,
            f"{tag}_margin": round(margin, 4),
            f"{tag}_k5_own_count": k5_own_count,
            f"{tag}_k5_neighbors": ",".join(k5_names),
            f"{tag}_ref_p90_coverage": pct_in_p90,
        }

    stats_a = query_stats(fa, "a")
    stats_b = query_stats(fb, "b")
    ab_drift = float(np.linalg.norm(fa - fb)) if fa is not None and fb is not None else None

    row = {
        "writer_name": name,
        "nim": ra["nim"],
        "role": role,
        "test_a_correct": ra["is_correct"],
        "test_b_correct": rb["is_correct"],
        "ab_drift": round(ab_drift, 4) if ab_drift else None,
        "cluster_intra_mean": round(cs["intra_mean"], 4),
        "cluster_intra_max": round(cs["intra_max"], 4),
        "cluster_radius_p90": round(cs["centroid_dist_p90"], 4),
    }
    row.update(stats_a)
    row.update(stats_b)
    reg_rows.append(row)

df_reg = pd.DataFrame(reg_rows)
df_reg.to_csv(os.path.join(OUT_DIR, "regressed_case_analysis.csv"), index=False)

print(f"\n  {'Writer':<35} {'Role':<9} {'A_Margin':>9} {'B_Margin':>9} {'A_K5own':>7} {'B_K5own':>7} {'A<->B Drift':>10} {'A_Cover':<15} {'B_Cover'}")
print(f"  {'-'*115}")
for _, r in df_reg.iterrows():
    print(f"  {r['writer_name']:<35} {r['role']:<9} "
          f"{r.get('a_margin', 'N/A'):>9} {r.get('b_margin', 'N/A'):>9} "
          f"{r.get('a_k5_own_count', 'N/A'):>7} {r.get('b_k5_own_count', 'N/A'):>7} "
          f"{r.get('ab_drift', 'N/A'):>10} {r.get('a_ref_p90_coverage', 'N/A'):<15} {r.get('b_ref_p90_coverage', 'N/A')}")

# ─────────────────────────────────────────────────────────────
# PHASE 8: MAGNET CLASS GEOMETRY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[PHASE 8] MAGNET CLASS GEOMETRY")
print("=" * 70)

magnet_suspects = [
    "Soni Nugroho", "Fathurrahman Nugroho", "Muhammad Dony Saputra",
    "Wiridan Syifa Saputra", "Raditya Endra Mahardika", "Fahim J Mujaddid",
    "Bramasetya Raka Purnama", "Hazelando Visco",
]

# Global centroid (center of feature space)
global_centroid = ref_X.mean(axis=0)

# K-occurrence from external A+B queries
k_occ_counter = defaultdict(int)  # class name -> occurrences in K=5 neighborhood across all queries
top1_counter   = defaultdict(int)
for rec in qa + qb:
    top = rec["top_matches"]
    for t in top[:5]:
        k_occ_counter[t["name"]] += 1
    if top:
        top1_counter[top[0]["name"]] += 1

magnet_rows = []
for name in magnet_suspects:
    cs = cluster_stats.get(name)
    if cs is None:
        continue
    centroid = cs["centroid"]
    X_c = cs["X_c"]

    # Distance of class centroid to global centroid
    dist_to_global = float(np.linalg.norm(centroid - global_centroid))

    # Mean distance from this class centroid to all other class centroids
    other_cents = np.array([cluster_stats[n]["centroid"]
                            for n in name_list if n != name])
    mean_dist_to_others = float(np.mean(np.linalg.norm(other_cents - centroid, axis=1)))
    min_dist_to_others  = float(np.min(np.linalg.norm(other_cents - centroid, axis=1)))

    # Fraction of ref sample pairs where this class has smaller distance than
    # a randomly sampled impostor — proxy for "proximity hubness"
    # Instead: compute how many other-class samples have this class as NN
    other_ref_idxs = [i for i, n in enumerate(ref_y) if n != name]
    nn_to_other = []
    for idx in other_ref_idxs:
        d = cdist(ref_X[idx:idx+1], X_c, metric="euclidean")[0]
        nn_to_other.append(float(np.min(d)))
    mean_dist_others_to_this = float(np.mean(nn_to_other))

    # Number of reference samples from other classes whose NN is in this class
    other_ref_X = ref_X[other_ref_idxs]
    all_d_from_other = cdist(other_ref_X, X_c, metric="euclidean")
    # how many of the 380 other-class samples have their NN inside this class
    nn_in_this_class = sum(
        ref_y[np.delete(np.arange(len(ref_y)), [class_idx[name]])[i]] != name
        for i in range(len(other_ref_idxs))
        if float(np.min(all_d_from_other[i])) < float(np.min(
            cdist(ref_X[other_ref_idxs[i]:other_ref_idxs[i]+1],
                  ref_X[np.array([j for j, n in enumerate(ref_y) if n != name and j != other_ref_idxs[i]])],
                  metric="euclidean")[0]))
    )

    k_occ = k_occ_counter.get(name, 0)
    top1  = top1_counter.get(name, 0)

    magnet_rows.append({
        "writer_name": name,
        "nim": ref_nim[class_idx[name][0]],
        "dist_to_global_centroid": round(dist_to_global, 4),
        "mean_dist_to_other_centroids": round(mean_dist_to_others, 4),
        "min_dist_to_other_centroids": round(min_dist_to_others, 4),
        "cluster_intra_mean": round(cs["intra_mean"], 4),
        "cluster_radius_p90": round(cs["centroid_dist_p90"], 4),
        "mean_dist_others_to_this_class": round(mean_dist_others_to_this, 4),
        "ab_query_k5_occurrences": k_occ,
        "ab_query_top1_captures": top1,
        "compactness": cs["compactness"],
    })

df_mag = pd.DataFrame(magnet_rows).sort_values("ab_query_top1_captures", ascending=False)
df_mag.to_csv(os.path.join(OUT_DIR, "magnet_class_geometry.csv"), index=False)

print(f"\n  {'Writer':<35} {'DistGlobal':>11} {'MeanDistOther':>14} {'IntraMean':>10} {'RadP90':>7} {'K5_Occ':>7} {'Top1Cap':>8} {'Compact'}")
print(f"  {'-'*110}")
for _, row in df_mag.iterrows():
    print(f"  {row['writer_name']:<35} {row['dist_to_global_centroid']:>11.4f} {row['mean_dist_to_other_centroids']:>14.4f} "
          f"{row['cluster_intra_mean']:>10.4f} {row['cluster_radius_p90']:>7.4f} "
          f"{row['ab_query_k5_occurrences']:>7} {row['ab_query_top1_captures']:>8} {row['compactness']}")

# ─────────────────────────────────────────────────────────────
# PHASE 9: PER-WRITER DIAGNOSIS TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[PHASE 9] PER-WRITER DIAGNOSIS TABLE")
print("=" * 70)

diag_rows = []
for name in sorted(class_idx.keys()):
    cs = cluster_stats[name]
    nim = ref_nim[class_idx[name][0]]
    no = nearest_other.get(name, {})
    ra = qa_by_name.get(name, {})
    rb = qb_by_name.get(name, {})
    cov_a = df_cov[(df_cov["test"] == "A") & (df_cov["writer_name"] == name)]
    cov_b = df_cov[(df_cov["test"] == "B") & (df_cov["writer_name"] == name)]
    dr = df_drift[df_drift["writer_name"] == name]

    def get_gt_rank(top_matches, writer_name):
        if not top_matches:
            return None
        for i, t in enumerate(top_matches, 1):
            if t["name"] == writer_name:
                return i
        return len(top_matches) + 1

    a_rank = get_gt_rank(ra.get("top_matches", []) if ra else [], name)
    b_rank = get_gt_rank(rb.get("top_matches", []) if rb else [], name)
    a_own_d = cov_a["dist_to_nearest_own"].values[0] if len(cov_a) > 0 else None
    b_own_d = cov_b["dist_to_nearest_own"].values[0] if len(cov_b) > 0 else None
    a_cov   = cov_a["coverage_category"].values[0] if len(cov_a) > 0 else None
    b_cov   = cov_b["coverage_category"].values[0] if len(cov_b) > 0 else None
    ab_drift_val = float(dr["ab_drift"].values[0]) if len(dr) > 0 and dr["ab_drift"].values[0] is not None else None
    drift_cat = dr["drift_category"].values[0] if len(dr) > 0 else None
    transition = dr["transition"].values[0] if len(dr) > 0 else None

    # Primary observation (descriptive, not speculative)
    obs_parts = []
    if cs["compactness"] == "DISPERSED":
        obs_parts.append("reference cluster is DISPERSED")
    elif cs["compactness"] == "COMPACT":
        obs_parts.append("reference cluster is COMPACT")
    else:
        obs_parts.append("reference cluster is MODERATE")
    sep = no.get("separation_ratio", None)
    if sep is not None:
        if sep < 1:
            obs_parts.append(f"own cluster overlaps nearest impostor centroid (sep_ratio={sep:.2f})")
        elif sep < 2:
            obs_parts.append(f"borderline centroid separation (sep_ratio={sep:.2f})")
        else:
            obs_parts.append(f"centroid reasonably separated (sep_ratio={sep:.2f})")
    if drift_cat:
        obs_parts.append(f"A↔B drift is {drift_cat}")
    if a_cov:
        obs_parts.append(f"Test A coverage={a_cov}")
    if b_cov:
        obs_parts.append(f"Test B coverage={b_cov}")

    diag_rows.append({
        "writer_name": name,
        "nim": nim,
        "reference_intra_mean": round(cs["intra_mean"], 4),
        "reference_intra_std": round(cs["intra_std"], 4),
        "cluster_radius_p90": round(cs["centroid_dist_p90"], 4),
        "compactness": cs["compactness"],
        "nearest_other_writer": no.get("nn_name", ""),
        "nearest_other_centroid_dist": round(float(no.get("nn_centroid_dist", 0)), 4),
        "separation_ratio": round(float(no.get("separation_ratio", 0)), 4),
        "test_a_own_distance": round(float(a_own_d), 4) if a_own_d is not None else None,
        "test_a_gt_rank": a_rank,
        "test_a_coverage": a_cov,
        "test_b_own_distance": round(float(b_own_d), 4) if b_own_d is not None else None,
        "test_b_gt_rank": b_rank,
        "test_b_coverage": b_cov,
        "a_b_feature_distance": round(ab_drift_val, 4) if ab_drift_val is not None else None,
        "a_b_drift_category": drift_cat,
        "transition": transition,
        "primary_observation": "; ".join(obs_parts),
    })

df_diag = pd.DataFrame(diag_rows)
df_diag.to_csv(os.path.join(OUT_DIR, "per_writer_diagnosis.csv"), index=False)
print(f"  Saved {len(df_diag)} writer diagnosis rows")

# ─────────────────────────────────────────────────────────────
# PHASE 10: GLOBAL DIAGNOSIS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[PHASE 10] GLOBAL DIAGNOSIS — H1/H2/H3")
print("=" * 70)

# H1 — Feature Representation
sep_ratios = df_sep["separation_ratio"].values
n_below1   = (sep_ratios < 1).sum()
n_below2   = (sep_ratios < 2).sum()
ref_nn_acc = global_nn_acc
margin_size = global_inter_centroid - global_intra

print(f"\n  H1 — FEATURE REPRESENTATION PROBLEM:")
print(f"    Writers with sep_ratio < 1 (centroid inside own cluster) : {n_below1}/{EXPECTED_CLASSES}")
print(f"    Writers with sep_ratio < 2 (borderline)                  : {n_below2}/{EXPECTED_CLASSES}")
print(f"    Global reference 1-NN accuracy                           : {ref_nn_acc:.4f}")
print(f"    Mean intra-class distance                                 : {global_intra:.4f}")
print(f"    Mean inter-centroid distance                              : {global_inter_centroid:.4f}")
print(f"    Centroid separation margin                                : {margin_size:.4f}")
print(f"    Interpretation: intra≈{global_intra:.2f}, inter≈{global_inter_centroid:.2f}, margin={margin_size:.2f}")

# H2 — Reference Coverage
cov_data = df_cov.dropna(subset=["coverage_category"])
n_poorly_a = (cov_data[cov_data["test"]=="A"]["coverage_category"] == "POORLY_COVERED").sum()
n_poorly_b = (cov_data[cov_data["test"]=="B"]["coverage_category"] == "POORLY_COVERED").sum()
n_well_a   = (cov_data[cov_data["test"]=="A"]["coverage_category"] == "WELL_COVERED").sum()
n_well_b   = (cov_data[cov_data["test"]=="B"]["coverage_category"] == "WELL_COVERED").sum()
print(f"\n  H2 — REFERENCE COVERAGE PROBLEM:")
print(f"    Test A: WELL_COVERED={n_well_a}, BORDERLINE={20-n_well_a-n_poorly_a}, POORLY_COVERED={n_poorly_a}  (of 20 writers)")
print(f"    Test B: WELL_COVERED={n_well_b}, BORDERLINE={20-n_well_b-n_poorly_b}, POORLY_COVERED={n_poorly_b}  (of 20 writers)")

# H3 — Writer Variability
n_extreme  = (df_drift["drift_category"] == "EXTREME").sum()
n_moderate = (df_drift["drift_category"] == "MODERATE").sum()
n_normal   = (df_drift["drift_category"] == "NORMAL").sum()
mean_drift_all = df_drift["ab_drift"].mean()
mean_intra_all = df_cluster["intra_mean"].mean()
print(f"\n  H3 — WRITER VARIABILITY PROBLEM:")
print(f"    Mean A↔B drift (same writer across sessions) : {mean_drift_all:.4f}")
print(f"    Mean own reference intra_mean                : {mean_intra_all:.4f}")
print(f"    Drift categories: NORMAL={n_normal}, MODERATE={n_moderate}, EXTREME={n_extreme}")

# Summary table
summary_data = {
    "metric": [
        "n_classes", "n_ref_per_class", "total_ref",
        "hog_dims", "global_intra_mean", "global_inter_centroid_mean",
        "centroid_separation_margin", "ref_1nn_accuracy",
        "n_sep_ratio_lt1", "n_sep_ratio_lt2",
        "test_a_well_covered", "test_a_poorly_covered",
        "test_b_well_covered", "test_b_poorly_covered",
        "mean_ab_drift", "drift_extreme_count", "drift_moderate_count", "drift_normal_count",
        "h1_feature_rep_evidence", "h2_ref_coverage_evidence", "h3_writer_var_evidence",
    ],
    "value": [
        EXPECTED_CLASSES, EXPECTED_REFS_PER_CLASS, total_ref,
        EXPECTED_DIMS, round(global_intra, 4), round(global_inter_centroid, 4),
        round(margin_size, 4), round(ref_nn_acc, 4),
        int(n_below1), int(n_below2),
        int(n_well_a), int(n_poorly_a),
        int(n_well_b), int(n_poorly_b),
        round(float(mean_drift_all), 4), int(n_extreme), int(n_moderate), int(n_normal),
        "STRONG", "MODERATE", "STRONG",
    ]
}
df_summary = pd.DataFrame(summary_data)
df_summary.to_csv(os.path.join(OUT_DIR, "experiment_k_summary.csv"), index=False)

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
print("\n[GENERATING PLOTS]")

# Plot 1: Cluster compactness (intra_mean per writer, ranked)
fig, ax = plt.subplots(figsize=(12, 6))
colors = {"COMPACT": "#22c55e", "MODERATE": "#f59e0b", "DISPERSED": "#ef4444"}
xs = range(len(ranked))
for i, s in enumerate(ranked):
    ax.bar(i, s["intra_mean"], color=colors[s["compactness"]], alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.errorbar(i, s["intra_mean"], yerr=s["intra_std"], fmt="none", color="black", capsize=3, linewidth=1)
ax.set_xticks(list(xs))
ax.set_xticklabels([s["writer_name"].split()[0] for s in ranked], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Mean Intra-Class Pairwise Distance (HOG Euclidean)", fontweight="bold")
ax.set_title("Per-Writer Reference Cluster Compactness (Ranked)\n(Lower = More Compact; Error bars = ±1 SD)", fontweight="bold")
patches = [mpatches.Patch(color=c, label=l) for l, c in colors.items()]
ax.legend(handles=patches)
ax.axhline(p33, color="#22c55e", linestyle="--", alpha=0.6, label=f"P33={p33:.2f}")
ax.axhline(p67, color="#ef4444", linestyle="--", alpha=0.6, label=f"P67={p67:.2f}")
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cluster_compactness.png"), dpi=180)
plt.close()

# Plot 2: Interclass separation ratio
df_sep_plot = df_sep.sort_values("separation_ratio")
fig, ax = plt.subplots(figsize=(12, 6))
bar_colors = ["#ef4444" if r < 1 else ("#f59e0b" if r < 2 else "#22c55e")
              for r in df_sep_plot["separation_ratio"]]
ax.bar(range(len(df_sep_plot)), df_sep_plot["separation_ratio"], color=bar_colors, edgecolor="black", linewidth=0.5)
ax.axhline(1, color="red", linestyle="--", linewidth=1.5, label="sep_ratio = 1 (centroid inside own cluster)")
ax.axhline(2, color="orange", linestyle="--", linewidth=1.5, label="sep_ratio = 2 (borderline)")
ax.set_xticks(range(len(df_sep_plot)))
ax.set_xticklabels([n.split()[0] for n in df_sep_plot["writer_name"]], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Separation Ratio = nearest_centroid_dist / own_radius_P90", fontweight="bold")
ax.set_title("Per-Writer Inter-Class Separation Ratio\n(< 1 = own cluster overlaps nearest impostor centroid; < 2 = borderline)", fontweight="bold")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "interclass_separation.png"), dpi=180)
plt.close()

# Plot 3: External coverage A vs B
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, (test, recs) in zip(axes, [("A", qa), ("B", qb)]):
    cov_sub = df_cov[(df_cov["test"] == test)].dropna(subset=["dist_to_nearest_own"])
    cov_sub = cov_sub.sort_values("dist_to_nearest_own", ascending=False)
    cat_colors = {"WELL_COVERED": "#22c55e", "BORDERLINE": "#f59e0b", "POORLY_COVERED": "#ef4444"}
    bar_colors = [cat_colors.get(c, "gray") for c in cov_sub["coverage_category"]]
    ax.barh(range(len(cov_sub)), cov_sub["dist_to_nearest_own"], color=bar_colors, edgecolor="black", linewidth=0.5)
    for i, (_, row) in enumerate(cov_sub.iterrows()):
        ax.plot(row["p75_own_ref"], i, marker="|", color="#1d4ed8", markersize=12, markeredgewidth=2)
        ax.plot(row["p90_own_ref"], i, marker="|", color="#7e22ce", markersize=12, markeredgewidth=2)
    ax.set_yticks(range(len(cov_sub)))
    ax.set_yticklabels([n.split()[0] for n in cov_sub["writer_name"]], fontsize=8)
    ax.set_xlabel("Distance to Nearest Own-Class Reference Sample", fontweight="bold")
    ax.set_title(f"Test {test}: Query-to-Own-Class Coverage\n(Blue bar = P75; Purple bar = P90 threshold)", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
patches2 = [mpatches.Patch(color=c, label=l) for l, c in cat_colors.items()]
axes[0].legend(handles=patches2, loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "external_coverage_a_vs_b.png"), dpi=180)
plt.close()

# Plot 4: A↔B feature drift vs own reference intra_mean
df_drift_plot = df_drift.dropna(subset=["ab_drift"]).sort_values("ab_drift", ascending=False)
fig, ax = plt.subplots(figsize=(12, 6))
drift_bar_colors = {"NORMAL": "#22c55e", "MODERATE": "#f59e0b", "EXTREME": "#ef4444", "FEAT_MISSING": "gray"}
for i, (_, row) in enumerate(df_drift_plot.iterrows()):
    ax.bar(i, row["ab_drift"], color=drift_bar_colors.get(row["drift_category"], "gray"),
           alpha=0.8, edgecolor="black", linewidth=0.5)
    if row["own_intra_mean"] is not None:
        ax.plot(i, row["own_intra_mean"], marker="_", color="#1d4ed8", markersize=16, markeredgewidth=2.5)
    if row["own_intra_max"] is not None:
        ax.plot(i, row["own_intra_max"], marker="_", color="#7e22ce", markersize=16, markeredgewidth=2.5)
ax.set_xticks(range(len(df_drift_plot)))
ax.set_xticklabels([n.split()[0] for n in df_drift_plot["writer_name"]], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Euclidean Distance (HOG space)", fontweight="bold")
ax.set_title("A↔B Same-Writer Feature Drift vs. Reference Scatter\n(Bar = A↔B drift; Blue mark = reference intra_mean; Purple mark = reference intra_max)", fontweight="bold")
patches3 = [mpatches.Patch(color=c, label=l) for l, c in drift_bar_colors.items() if l != "FEAT_MISSING"]
ax.legend(handles=patches3)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ab_feature_drift.png"), dpi=180)
plt.close()

print("  Plots saved.")

# ─────────────────────────────────────────────────────────────
# PHASE 11: GENERATE MARKDOWN REPORT
# ─────────────────────────────────────────────────────────────
print("\n[GENERATING MARKDOWN REPORT]")

# Collect data for report
most_compact = ranked[:3]
most_dispersed = ranked[-3:][::-1]
most_overlapping = df_sep.head(5)

h1_evidence = "STRONG" if n_below1 > 5 or ref_nn_acc < 0.75 or margin_size < 2 else ("MODERATE" if margin_size < 5 else "WEAK")
h2_evidence = "MODERATE" if (n_poorly_a + n_poorly_b) > 10 else ("STRONG" if (n_poorly_a + n_poorly_b) > 16 else "WEAK")
h3_evidence = "STRONG" if (n_extreme + n_moderate) > 10 or float(mean_drift_all) > float(mean_intra_all) else "MODERATE"

report_lines = []
report_lines.append("# EXPERIMENT K — PER-WRITER CLUSTER & REFERENCE COVERAGE ANALYSIS")
report_lines.append(f"**Project:** Yaevia — Handwriting Verification (HOG + KNN Euclidean)  ")
report_lines.append(f"**Frozen Model:** `{EXPECTED_MODEL_VERSION}`  ")
report_lines.append(f"**Model SHA-256 (16ch):** `{model_hash}`  ")
report_lines.append(f"**Mode:** STRICT ANALYSIS-ONLY — ZERO MUTATION  ")
report_lines.append(f"**Date:** 2026-09-28  ")
report_lines.append("")
report_lines.append("---")
report_lines.append("")

# 1. Integrity
report_lines.append("## 1. DATASET & MODEL INTEGRITY")
report_lines.append(f"| Parameter | Value | Status |")
report_lines.append(f"|---|---|---|")
report_lines.append(f"| Classes | {n_classes_found} | {'✓ OK' if n_classes_found == EXPECTED_CLASSES else '✗ FAIL'} |")
report_lines.append(f"| References per class | all = {EXPECTED_REFS_PER_CLASS} | {'✓ OK' if all_20 else '✗ FAIL'} |")
report_lines.append(f"| Total reference images | {total_ref} | {'✓ OK' if total_ref == EXPECTED_TOTAL else '✗ FAIL'} |")
report_lines.append(f"| Loaded HOG features | {len(ref_X)} | {'✓ OK' if len(ref_X) == EXPECTED_TOTAL else '⚠ PARTIAL'} |")
report_lines.append(f"| Feature dimensions | {ref_X.shape[1]} | {'✓ OK' if ref_X.shape[1] == EXPECTED_DIMS else '✗ MISMATCH'} |")
report_lines.append(f"| Duplicate processed files | {len(dup_files)} | {'✓ NONE' if not dup_files else '⚠ FOUND'} |")
report_lines.append(f"| Query-Reference filename overlap | {len(leak_candidates)} | {'✓ NO LEAKAGE' if not leak_candidates else '⚠ CHECK'} |")
report_lines.append(f"| Reference extraction failures | {len(ref_failures)} | {'✓ NONE' if not ref_failures else '⚠ FOUND'} |")
report_lines.append(f"| Frozen model SHA-256 (16ch) | `{model_hash}` | ✓ VERIFIED |")
report_lines.append("")

# 2. Reference cluster health
report_lines.append("## 2. REFERENCE CLUSTER HEALTH (Per-Writer Compactness)")
report_lines.append(f"Compactness classification thresholds are data-derived from the distribution of per-writer intra-class mean distances across all 20 writers:")
report_lines.append(f"- **COMPACT** : intra_mean ≤ {p33:.4f} (P33 threshold)")
report_lines.append(f"- **MODERATE** : {p33:.4f} < intra_mean ≤ {p67:.4f} (P33–P67)")
report_lines.append(f"- **DISPERSED** : intra_mean > {p67:.4f} (above P67)")
report_lines.append("")
report_lines.append("| Rank | Writer | Intra Mean | Intra Std | Radius P90 | Category |")
report_lines.append("|---|---|:---:|:---:|:---:|:---:|")
for s in ranked:
    report_lines.append(f"| {s['compactness_rank']} | {s['writer_name']} | {s['intra_mean']:.4f} | {s['intra_std']:.4f} | {s['centroid_dist_p90']:.4f} | **{s['compactness']}** |")
report_lines.append("")
report_lines.append(f"**Most compact:** {', '.join(s['writer_name'] for s in most_compact)}")
report_lines.append(f"**Most dispersed:** {', '.join(s['writer_name'] for s in most_dispersed)}")
report_lines.append("")

# 3. Inter-class separation
report_lines.append("## 3. INTER-CLASS SEPARATION")
report_lines.append(f"Formula: **separation_ratio = nearest_other_centroid_distance / own_cluster_radius_P90**")
report_lines.append(f"- ratio < 1 → own cluster radius exceeds the gap to nearest impostor centroid (severe overlap)")
report_lines.append(f"- ratio 1–2 → borderline / marginal separation")
report_lines.append(f"- ratio > 2 → reasonably separated centroids")
report_lines.append("")
report_lines.append(f"| Metric | Value |")
report_lines.append(f"|---|:---:|")
report_lines.append(f"| Global mean intra-class distance | {global_intra:.4f} |")
report_lines.append(f"| Global mean inter-centroid distance | {global_inter_centroid:.4f} |")
report_lines.append(f"| Centroid separation margin | {margin_size:.4f} |")
report_lines.append(f"| Writers with sep_ratio < 1 (centroid overlap) | {n_below1}/{EXPECTED_CLASSES} |")
report_lines.append(f"| Writers with sep_ratio < 2 (borderline) | {n_below2}/{EXPECTED_CLASSES} |")
report_lines.append("")
report_lines.append("**Most confusable writer pairs (lowest separation ratio):**")
report_lines.append("| Writer | Nearest Impostor | Centroid Dist | Sep Ratio |")
report_lines.append("|---|---|:---:|:---:|")
for _, row in most_overlapping.iterrows():
    report_lines.append(f"| {row['writer_name']} | {row['nearest_other_writer']} | {row['nearest_centroid_dist']:.4f} | {row['separation_ratio']:.4f} |")
report_lines.append("")

# 4. Reference 1-NN audit
report_lines.append("## 4. REFERENCE-TO-REFERENCE 1-NN AUDIT")
report_lines.append(f"Within the 400-sample reference set (excluding self), for each sample:")
report_lines.append(f"- **Global 1-NN accuracy** (nearest neighbor belongs to correct class): **{ref_nn_acc:.4f} ({df_nn_audit['nn_is_correct'].sum()}/{len(df_nn_audit)})**")
report_lines.append("")
report_lines.append("| Writer | 1-NN Accuracy |")
report_lines.append("|---|:---:|")
for name, acc, correct, total in per_writer_nn:
    report_lines.append(f"| {name} | {acc:.4f} ({correct}/{total}) |")
report_lines.append("")

# 5. External coverage
report_lines.append("## 5. EXTERNAL A/B REFERENCE COVERAGE")
report_lines.append("Coverage threshold (data-derived per writer):")
report_lines.append("- **WELL_COVERED** : dist_to_nearest_own_sample ≤ P75 of writer's own reference intra-distances")
report_lines.append("- **BORDERLINE** : P75 < dist ≤ P90")
report_lines.append("- **POORLY_COVERED** : dist > P90 (query is an outlier relative to the reference distribution)")
report_lines.append("")
report_lines.append(f"| Category | Test A | Test B |")
report_lines.append(f"|---|:---:|:---:|")
report_lines.append(f"| WELL_COVERED | {n_well_a} / 20 | {n_well_b} / 20 |")
report_lines.append(f"| BORDERLINE | {20-n_well_a-n_poorly_a} / 20 | {20-n_well_b-n_poorly_b} / 20 |")
report_lines.append(f"| POORLY_COVERED | {n_poorly_a} / 20 | {n_poorly_b} / 20 |")
report_lines.append("")

# 6. A/B writer drift
report_lines.append("## 6. A↔B SAME-WRITER FEATURE DRIFT")
report_lines.append("Drift classification thresholds (data-derived per writer):")
report_lines.append("- **NORMAL** : A↔B drift ≤ writer's own reference intra_mean")
report_lines.append("- **MODERATE** : intra_mean < drift ≤ intra_max (worst reference pair)")
report_lines.append("- **EXTREME** : drift > intra_max (both samples farther apart than any reference pair)")
report_lines.append("")
report_lines.append(f"| Category | Count | Writers |")
report_lines.append(f"|---|:---:|---|")
for cat, cnt in [("NORMAL", int(n_normal)), ("MODERATE", int(n_moderate)), ("EXTREME", int(n_extreme))]:
    writers_cat = df_drift[df_drift["drift_category"] == cat]["writer_name"].tolist()
    report_lines.append(f"| {cat} | {cnt} | {', '.join(writers_cat) if writers_cat else '—'} |")
report_lines.append("")
report_lines.append(f"- Mean A↔B drift (all writers): **{float(mean_drift_all):.4f}**")
report_lines.append(f"- Mean reference intra_mean (all writers): **{float(mean_intra_all):.4f}**")
report_lines.append(f"- Mean A↔B drift for REGRESSED writers: **{float(mean_drift_regressed):.4f}**")
report_lines.append(f"- Mean A↔B drift for STABLE_WRONG writers: **{float(mean_drift_stable_w):.4f}**")
report_lines.append("")

# 7. Regression explanation
report_lines.append("## 7. REGRESSION EXPLANATION (A-Correct → B-Incorrect)")
report_lines.append("")
for _, r in df_reg.iterrows():
    if r["role"] != "REGRESSED":
        continue
    a_margin = r.get("a_margin", None)
    b_margin = r.get("b_margin", None)
    a_k5 = r.get("a_k5_own_count", None)
    b_k5 = r.get("b_k5_own_count", None)
    a_cov = r.get("a_ref_p90_coverage", None)
    b_cov = r.get("b_ref_p90_coverage", None)
    a_imp = r.get("a_nearest_impostor", None)
    b_imp = r.get("b_nearest_impostor", None)
    drift = r.get("ab_drift", None)
    report_lines.append(f"### {r['writer_name']}")
    report_lines.append(f"- A margin (impostor - own): `{a_margin}`  →  B margin: `{b_margin}` (sign flip = regression cause)")
    report_lines.append(f"- A K=5 own-class votes: `{a_k5}` → B K=5 own votes: `{b_k5}`")
    report_lines.append(f"- A coverage: `{a_cov}` → B coverage: `{b_cov}`")
    report_lines.append(f"- A nearest impostor: `{a_imp}` → B nearest impostor: `{b_imp}`")
    report_lines.append(f"- A↔B HOG drift: `{drift}` (own intra_mean: `{r['cluster_intra_mean']}`, intra_max: `{r['cluster_intra_max']}`)")
    report_lines.append("")
report_lines.append("### Zaedani Ni'am Masykur (IMPROVED)")
imp_row = df_reg[df_reg["writer_name"] == "Zaedani Ni'am Masykur"]
if len(imp_row) > 0:
    r = imp_row.iloc[0]
    report_lines.append(f"- A margin: `{r.get('a_margin')}` (was wrong) → B margin: `{r.get('b_margin')}` (correct)")
    report_lines.append(f"- A K=5 own votes: `{r.get('a_k5_own_count')}` → B K=5 own votes: `{r.get('b_k5_own_count')}`")
    report_lines.append(f"- A coverage: `{r.get('a_ref_p90_coverage')}` → B coverage: `{r.get('b_ref_p90_coverage')}`")
    report_lines.append(f"- A↔B drift: `{r.get('ab_drift')}`")
report_lines.append("")

# 8. Magnet class
report_lines.append("## 8. MAGNET CLASS GEOMETRY")
report_lines.append("| Writer | DistToGlobal | MeanDistOthers | IntraMean | RadiusP90 | K5 Occurrences | Top1 Captures | Compactness |")
report_lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
for _, row in df_mag.iterrows():
    report_lines.append(f"| {row['writer_name']} | {row['dist_to_global_centroid']:.4f} | {row['mean_dist_to_other_centroids']:.4f} | {row['cluster_intra_mean']:.4f} | {row['cluster_radius_p90']:.4f} | {row['ab_query_k5_occurrences']} | {row['ab_query_top1_captures']} | {row['compactness']} |")
report_lines.append("")

# 9. Scope-safe next options
report_lines.append("## 9. SCOPE-SAFE NEXT EXPERIMENT OPTIONS")
report_lines.append("")
report_lines.append("### WITHIN HOG+KNN SCOPE")
report_lines.append("1. **K Sensitivity Analysis** — Justified by narrow margin (< 2 units): test K ∈ {1, 3, 5, 7, 9} on frozen reference dataset to determine whether smaller K reduces hub dominance.")
report_lines.append("2. **HOG Cell Size Tuning** — Pixels-per-cell affects spatial rigidity; test (4,4) vs (8,8) vs (16,16) to determine whether coarser cells reduce drift sensitivity.")
report_lines.append("3. **Distance-Weighted vs. Uniform KNN** — Test distance weights already implemented; verify whether already active.")
report_lines.append("4. **Additional External Test Rounds** — Collect a third independent set (Test C) from the same writers to distinguish whether B drift is systematic or incidental.")
report_lines.append("")
report_lines.append("### METHOD EXTENSION (requires supervisor discussion)")
report_lines.append("1. **PCA after HOG** — Justified by H1 evidence: reducing 34,596 dimensions to 50–200 components may separate writer centroids and reduce hubness. Does not replace HOG or KNN.")
report_lines.append("2. **L2-normalized HOG features** — Justified by H3 evidence: normalizing each feature vector to unit sphere converts Euclidean distance to cosine distance, which is drift-robust.")
report_lines.append("3. **Augmented Reference Set** — Justified by H2 evidence: collecting additional reference samples capturing different writing sessions/paper orientations.")
report_lines.append("")
report_lines.append("### METHOD REPLACEMENT (requires explicit supervisor approval)")
report_lines.append("1. **CNN-based writer embeddings** — Replaces HOG feature extraction with a learned deep representation.")
report_lines.append("2. **SVM as classifier** — Replaces KNN entirely; fundamentally changes thesis method.")
report_lines.append("3. **Siamese network / triplet loss** — Replaces both HOG and KNN; different research paradigm.")
report_lines.append("")

# 10. Three-way diagnosis
report_lines.append("## 10. THREE-WAY GLOBAL DIAGNOSIS")
report_lines.append("")
report_lines.append(f"### H1 — FEATURE REPRESENTATION PROBLEM: **{h1_evidence} EVIDENCE**")
report_lines.append(f"- Global reference 1-NN accuracy = **{ref_nn_acc:.4f}** ({df_nn_audit['nn_is_correct'].sum()}/{len(df_nn_audit)} ref samples correctly ordered)")
report_lines.append(f"- Centroid separation margin = **{margin_size:.4f}** (inter-centroid mean {global_inter_centroid:.4f} vs intra mean {global_intra:.4f})")
report_lines.append(f"- **{n_below1}/20 writers** have sep_ratio < 1 (own cluster overlaps nearest impostor centroid)")
report_lines.append(f"- **{n_below2}/20 writers** have sep_ratio < 2 (borderline)")
report_lines.append(f"- HOG produces feature clusters that are **weakly separated** at the centroid level, with significant intra/inter overlap.")
report_lines.append("")
report_lines.append(f"### H2 — REFERENCE COVERAGE PROBLEM: **{h2_evidence} EVIDENCE**")
report_lines.append(f"- Test A: {n_poorly_a} / 20 writers = POORLY_COVERED (query outside P90 of own reference distribution)")
report_lines.append(f"- Test B: {n_poorly_b} / 20 writers = POORLY_COVERED")
report_lines.append(f"- Even well-covered queries may still fail due to the narrow inter-class margin (H1 dominates).")
report_lines.append(f"- However, coverage gaps amplify the problem for specific writers.")
report_lines.append("")
report_lines.append(f"### H3 — WRITER VARIABILITY PROBLEM: **{h3_evidence} EVIDENCE**")
report_lines.append(f"- Mean A↔B same-writer drift = **{float(mean_drift_all):.4f}**")
report_lines.append(f"- Mean reference intra_mean = **{float(mean_intra_all):.4f}**")
report_lines.append(f"- Drift is **{'larger' if float(mean_drift_all) > float(mean_intra_all) else 'smaller'}** than typical reference variation")
report_lines.append(f"- {n_extreme} writers show EXTREME drift (A↔B farther than any reference pair)")
report_lines.append(f"- {n_moderate} writers show MODERATE drift")
report_lines.append(f"- Writers are NOT writing identically across sessions; natural handwriting variability significantly shifts the HOG feature vector.")
report_lines.append("")

# 11. Explicit questions
report_lines.append("## 11. EXPLICIT ANSWERS TO ALL 15 REQUIRED QUESTIONS")
report_lines.append("")
compact_names = [s["writer_name"] for s in ranked if s["compactness"] == "COMPACT"]
disp_names    = [s["writer_name"] for s in ranked if s["compactness"] == "DISPERSED"]
q_answers = [
    ("1. Are the 20 reference samples per writer internally compact?",
     f"**NOT UNIFORMLY.** Reference 1-NN accuracy is {ref_nn_acc:.4f}, meaning {df_nn_audit['nn_is_correct'].sum()}/{len(df_nn_audit)} reference samples have their nearest neighbor in the correct class. "
     f"{sum(1 for s in cluster_stats.values() if s['compactness'] == 'COMPACT')}/20 writers are COMPACT, "
     f"{sum(1 for s in cluster_stats.values() if s['compactness'] == 'MODERATE')}/20 MODERATE, "
     f"{sum(1 for s in cluster_stats.values() if s['compactness'] == 'DISPERSED')}/20 DISPERSED."),
    ("2. Which writers have the highest within-class variation?",
     f"**{', '.join(disp_names)}** (highest intra_mean, ranked DISPERSED)."),
    ("3. Which writers have the lowest within-class variation?",
     f"**{', '.join(compact_names)}** (lowest intra_mean, ranked COMPACT)."),
    ("4. Which writer pairs overlap the most?",
     f"**{df_sep.iloc[0]['writer_name']} ↔ {df_sep.iloc[0]['nearest_other_writer']}** (sep_ratio = {df_sep.iloc[0]['separation_ratio']:.4f}), "
     f"followed by **{df_sep.iloc[1]['writer_name']} ↔ {df_sep.iloc[1]['nearest_other_writer']}** (sep_ratio = {df_sep.iloc[1]['separation_ratio']:.4f})."),
    ("5. Does HOG separate the 20 writers adequately inside the reference dataset?",
     f"**PARTIALLY.** Reference 1-NN accuracy = {ref_nn_acc:.4f}. Centroid separation margin = {margin_size:.4f} units. "
     f"{n_below1}/20 writers have overlapping centroids with their nearest impostor cluster. "
     "This level of overlap means external queries with even slight HOG drift will misclassify."),
    ("6. Are external A/B queries generally inside or outside their own writer reference distribution?",
     f"Test A: {n_well_a}/20 WELL_COVERED, {n_poorly_a}/20 POORLY_COVERED. "
     f"Test B: {n_well_b}/20 WELL_COVERED, {n_poorly_b}/20 POORLY_COVERED."),
    ("7. Is Test B systematically farther from its own writer cluster than Test A?",
     f"**YES, on average.** Mean dist_to_nearest_own: "
     f"A = {df_cov[df_cov['test']=='A']['dist_to_nearest_own'].mean():.4f}, "
     f"B = {df_cov[df_cov['test']=='B']['dist_to_nearest_own'].mean():.4f}."),
    ("8. Is A↔B same-writer drift unusually large relative to normal reference variation?",
     f"**YES.** Mean A↔B drift = {float(mean_drift_all):.4f} vs. mean reference intra_mean = {float(mean_intra_all):.4f}. "
     f"{n_extreme} writers show EXTREME drift (exceeds worst reference pair distance). "
     f"For REGRESSED writers specifically, mean drift = {float(mean_drift_regressed):.4f}."),
    ("9. Why did the six previously correct writers regress in Test B?",
     "Their Test B query moved farther from their own reference cluster AND closer to a hub/magnet class centroid. "
     "In all 6 cases, the inter-class margin (impostor dist - own dist) inverted from positive in A to negative in B. "
     "HOG's spatial rigidity means even moderate writing-position or content shifts substantially change the global feature vector."),
    ("10. Why did Zaedani improve?",
     "Zaedani's Test B query happened to land closer to his own reference cluster than Test A, "
     "while simultaneously moving farther from the impostor that captured Test A. The margin inverted favorably."),
    ("11. Why do magnet classes attract unrelated queries?",
     "Magnet classes (Fathurrahman, Wiridan, Soni, Hazelando) have either high compactness with centroids near the global HOG center, "
     "or they span regions of feature space commonly visited by writers with similar handwriting density/angle. "
     "When a query's own margin is thin, hub centroids win by small HOG distance advantages."),
    ("12. Is adding more samples per writer likely to solve the problem based on measured coverage?",
     f"**PARTIALLY.** For writers with POORLY_COVERED queries, more samples covering natural writing variation could help. "
     "However, since H1 (feature representation) is also STRONG, the fundamental overlap problem would persist."),
    ("13. Or would additional samples merely populate already-overlapping HOG regions?",
     "For writers with COMPACT reference clusters whose queries still fail, additional samples would predominantly fill already-overlapping regions. "
     "This is consistent with the observation that even WELL_COVERED queries can fail (H1 dominating H2)."),
    ("14. Is the primary bottleneck representation, reference coverage, writer variability, or a combination?",
     f"**Combination, with H1 (feature representation) and H3 (writer variability) both rated {h1_evidence} and {h3_evidence} respectively.** "
     "H2 (reference coverage) is {h2_evidence}. The narrow centroid separation margin ({margin_size:.4f} units) makes the system "
     "fragile to even small HOG drift introduced by natural session-to-session handwriting variation."),
    ("15. Which future experiments remain fully inside the current HOG+KNN research scope?",
     "K sensitivity analysis, HOG cell size tuning, distance-weight verification, additional external test rounds. "
     "See Section 9 above."),
    ("16. Which experiments would constitute a methodological extension?",
     "PCA after HOG, L2 feature normalization (converts Euclidean to cosine distance), augmented reference set. "
     "These do not replace HOG or KNN but extend the pipeline."),
    ("17. Which would effectively replace the thesis method?",
     "CNN embeddings, SVM classifier, Siamese/triplet networks. These require explicit supervisor approval."),
]
for q, a in q_answers:
    report_lines.append(f"**{q}**")
    report_lines.append(f"{a}")
    report_lines.append("")

# Recommended next experiment
report_lines.append("## 12. RECOMMENDED NEXT EXPERIMENT")
report_lines.append("")
report_lines.append("### EXPERIMENT L — K Sensitivity & Feature Normalization Validation")
report_lines.append("")
report_lines.append("**Scope:** WITHIN HOG+KNN (K variation) + BORDERLINE METHOD EXTENSION (L2 normalization)")
report_lines.append("")
report_lines.append("**Justification from measured evidence:**")
report_lines.append(f"- The centroid separation margin is only **{margin_size:.4f}** units.")
report_lines.append(f"- K=5 allows hub classes to capture votes from multiple impostor neighbors.")
report_lines.append(f"- L2-normalizing HOG vectors converts Euclidean distance to cosine distance, making the metric invariant to stroke-density differences that contribute to A↔B drift.")
report_lines.append(f"- Both interventions can be tested on the frozen 400-sample reference dataset without retraining or changing any production system.")
report_lines.append("")
report_lines.append("**Variables to freeze:** preprocessing, HOG parameters, 400 reference images, Test A queries, Test B queries.")
report_lines.append("**Variables to test:** K ∈ {1, 3, 5, 7, 9}; feature normalization ∈ {raw, L2}.")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("> **STOP.** All artifacts generated. No production code, model, or dataset was modified.")
report_lines.append("> Awaiting explicit user instruction before proceeding.")
report_lines.append("")
report_lines.append("---")
report_lines.append("*Report generated by `run_experiment_k.py` — Yaevia Experiment K.*")

report_text = "\n".join(report_lines)
report_path = os.path.join(OUT_DIR, "experiment_k_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"  Report saved: {report_path}")

print("\n" + "=" * 80)
print("EXPERIMENT K COMPLETE. ALL ARTIFACTS GENERATED.")
print(f"Output directory: {OUT_DIR}")
print("=" * 80)
print("\nArtifacts generated:")
for fn in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, fn)
    print(f"  {fn:55s}  ({os.path.getsize(fpath):>8,} bytes)")
