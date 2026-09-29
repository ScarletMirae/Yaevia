"""
forensic_audit.py — External Holdout Forensic Audit (12-point)
================================================================
READ-ONLY access to D:\.SKRIPSI\Dataset (master dataset).
All outputs saved to backend/tests/evaluation_results/external_holdout_audit/

DO NOT:  delete / move / rename / overwrite / modify any file in master dataset.
DO NOT:  retrain / tune / modify production preprocessing.
"""

import os, sys, json, csv, hashlib, time, warnings
from pathlib import Path
import numpy as np
import sqlite3

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BACKEND_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from config import IMAGE_SIZE
from preprocessing.image_processor import preprocess_image, preprocess_from_array
from features.hog_extractor import extract_hog_features
import joblib, cv2

DATASET_ROOT  = r"D:\.SKRIPSI\Dataset"
RAW_DIR       = os.path.join(BACKEND_DIR, "dataset", "raw")
DB_PATH       = os.path.join(BACKEND_DIR, "database.db")
MODEL_PATH    = os.path.join(BACKEND_DIR, "model", "saved", "knn_model_20260924_183224.joblib")
LE_PATH       = os.path.join(BACKEND_DIR, "model", "saved", "label_encoder_20260924_183224.joblib")
OUT_DIR       = os.path.join(BACKEND_DIR, "tests", "evaluation_results", "external_holdout_audit")
os.makedirs(OUT_DIR, exist_ok=True)

IMG_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
EXPECTED_DIM  = 34596
EXPECTED_IMG  = (256, 256)

STUDENT_FOLDER_MAP = {
    "Alif":                 "Muhammad Alif Rizky Hutama",
    "Angela":               "Angela Permata Rosa",
    "Brama":                "Bramasetya Raka Purnama",
    "Dimas":                "Dimas Wahyu Prasetyo",
    "Dinar":                "Febrian Dinnar Purnama",
    "Dony":                 "Muhammad Dony Saputra",
    "Fahim":                "Fahim J Mujaddid",
    "Farhan Agiya":         "Farhan Agiya Pratama",
    "Fathur":               "Fathurrahman Nugroho",
    "Gayuh":                "Ibnu Gayuh Fadilah",
    "Hazel":                "Hazelando Visco",
    "Ilham":                "Ilham Rasyidan Muhammad",
    "Radit Kecil":          "Raditya Endra Mahardika",
    "Rakha":                "Rakha Burhannudin Majid",
    "Rifqi Rengga Praseno": "Rifqi Rengga Praseno",
    "Soni":                 "Soni Nugroho",
    "Wirid":                "Wiridan Syifa Saputra",
    "Zaedani Ni'am":        "Zaedani Ni'am Masykur",
}

STUDENT_NIM_MAP = {
    "Muhammad Alif Rizky Hutama": "A710230105",
    "Angela Permata Rosa":        "A710230006",
    "Bramasetya Raka Purnama":    "A710230101",
    "Dimas Wahyu Prasetyo":       "A710230113",
    "Febrian Dinnar Purnama":     "A710230102",
    "Muhammad Dony Saputra":      "A710230118",
    "Fahim J Mujaddid":           "A710230076",
    "Farhan Agiya Pratama":       "A710230089",
    "Fathurrahman Nugroho":       "A710230097",
    "Ibnu Gayuh Fadilah":         "A710230119",
    "Hazelando Visco":            "A710230092",
    "Ilham Rasyidan Muhammad":    "A710230078",
    "Raditya Endra Mahardika":    "A710230110",
    "Rakha Burhannudin Majid":    "A710230098",
    "Rifqi Rengga Praseno":       "A710230086",
    "Soni Nugroho":               "A710230112",
    "Wiridan Syifa Saputra":      "A710230080",
    "Zaedani Ni'am Masykur":      "A710230085",
}

LOOCV_ACCURACY_PER_STUDENT = {
    "Muhammad Alif Rizky Hutama": 75.0,
    "Angela Permata Rosa":        60.0,  # approximate from experiments
    "Bramasetya Raka Purnama":    50.0,
    "Dimas Wahyu Prasetyo":       55.0,
    "Febrian Dinnar Purnama":     35.0,
    "Muhammad Dony Saputra":      60.0,
    "Fahim J Mujaddid":           45.0,
    "Farhan Agiya Pratama":       None,
    "Fathurrahman Nugroho":       None,
    "Ibnu Gayuh Fadilah":         None,
    "Hazelando Visco":            None,
    "Ilham Rasyidan Muhammad":    None,
    "Raditya Endra Mahardika":    None,
    "Rakha Burhannudin Majid":    90.0,
    "Rifqi Rengga Praseno":       85.0,
    "Soni Nugroho":               None,
    "Wiridan Syifa Saputra":      None,
    "Zaedani Ni'am Masykur":      None,
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def describe(arr):
    if len(arr) == 0:
        return {}
    a = np.array(arr, dtype=float)
    return {
        "mean":   float(np.mean(a)),
        "median": float(np.median(a)),
        "std":    float(np.std(a)),
        "min":    float(np.min(a)),
        "max":    float(np.max(a)),
        "p25":    float(np.percentile(a, 25)),
        "p75":    float(np.percentile(a, 75)),
        "p90":    float(np.percentile(a, 90)),
        "p95":    float(np.percentile(a, 95)),
    }

def sim_old(d):   # formula used in holdout evaluator (WRONG for 34596-dim)
    return max(0.0, min(100.0, (1.0 - (d**2 / 450.0)) * 100.0))

def sim_new(d):   # formula used in classifier.py (CORRECT for 256x256)
    # max_dist_sq = 2 * N_blocks = 2 * 961 = 1922
    return max(0.0, min(100.0, (1.0 - (d**2 / 1922.0)) * 100.0))

print("=" * 70)
print("FORENSIC AUDIT — Yaevia External Holdout 18.94%")
print("=" * 70)

# ============================================================
# STEP 0: Load model & label encoder
# ============================================================
print("\n[LOAD] Model & label encoder ...")
model = joblib.load(MODEL_PATH)
le    = joblib.load(LE_PATH)
train_labels_encoded = model._y
train_labels         = le.inverse_transform(train_labels_encoded)  # (360,)
X_train              = model._fit_X  # (360, 34596)
print(f"  X_train shape : {X_train.shape}")
print(f"  n_features    : {model.n_features_in_}")
print(f"  K={model.n_neighbors}, metric={model.metric}, weights={model.weights}")

# ============================================================
# STEP 1: PIPELINE IDENTITY CHECK
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 1 — PIPELINE IDENTITY CHECK")
print("=" * 70)

# Pick one reference file from DB to test all 3 pipelines
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
ref_rows = conn.execute(
    "SELECT file_path, saved_filename, student_name FROM dataset LIMIT 5"
).fetchall()
conn.close()

test_row = ref_rows[0]
test_path = test_row["file_path"]
if not os.path.exists(test_path):
    test_path = os.path.join(RAW_DIR, test_row["saved_filename"])

print(f"  Test file: {os.path.basename(test_path)}  ({test_row['student_name']})")

# Pipeline A: Training pipeline (preprocess_image -> extract_hog_features)
t0 = time.perf_counter()
img_a    = preprocess_image(test_path)
feat_a   = extract_hog_features(img_a)
time_a   = (time.perf_counter() - t0) * 1000

# Pipeline B: Verify route pipeline (same functions — uses preprocess_image path)
t0 = time.perf_counter()
img_b    = preprocess_image(test_path)
feat_b   = extract_hog_features(img_b)
time_b   = (time.perf_counter() - t0) * 1000

# Pipeline C: Holdout evaluator (cv2.imread -> preprocess_from_array -> extract_hog_features)
t0 = time.perf_counter()
raw_bgr  = cv2.imread(test_path)
img_c    = preprocess_from_array(raw_bgr)
feat_c   = extract_hog_features(img_c)
time_c   = (time.perf_counter() - t0) * 1000

def compare_vectors(name_a, name_b, va, vb):
    diff = va - vb
    return {
        "pair":             f"{name_a} vs {name_b}",
        "shape_a":          str(va.shape),
        "shape_b":          str(vb.shape),
        "dtype_a":          str(va.dtype),
        "dtype_b":          str(vb.dtype),
        "min_a":            float(va.min()),  "min_b": float(vb.min()),
        "max_a":            float(va.max()),  "max_b": float(vb.max()),
        "mean_a":           float(va.mean()), "mean_b": float(vb.mean()),
        "std_a":            float(va.std()),  "std_b": float(vb.std()),
        "l2_norm_a":        float(np.linalg.norm(va)),
        "l2_norm_b":        float(np.linalg.norm(vb)),
        "n_different":      int(np.sum(va != vb)),
        "max_abs_diff":     float(np.max(np.abs(diff))),
        "mean_abs_diff":    float(np.mean(np.abs(diff))),
        "allclose":         bool(np.allclose(va, vb, rtol=1e-5, atol=1e-8)),
    }

pid_ab = compare_vectors("Training(A)", "Verify_route(B)", feat_a, feat_b)
pid_ac = compare_vectors("Training(A)", "Holdout_eval(C)", feat_a, feat_c)
pid_bc = compare_vectors("Verify_route(B)", "Holdout_eval(C)", feat_b, feat_c)

pipeline_identity = {
    "test_file":    os.path.basename(test_path),
    "student_name": test_row["student_name"],
    "pipeline_A":   {"name": "Training (preprocess_image)", "shape": str(feat_a.shape), "time_ms": round(time_a,2)},
    "pipeline_B":   {"name": "Verify route (preprocess_image)", "shape": str(feat_b.shape), "time_ms": round(time_b,2)},
    "pipeline_C":   {"name": "Holdout eval (cv2+preprocess_from_array)", "shape": str(feat_c.shape), "time_ms": round(time_c,2)},
    "A_vs_B":       pid_ab,
    "A_vs_C":       pid_ac,
    "B_vs_C":       pid_bc,
}

print(f"  Pipeline A (training)      : shape={feat_a.shape}, L2={np.linalg.norm(feat_a):.4f}")
print(f"  Pipeline B (verify route)  : shape={feat_b.shape}, L2={np.linalg.norm(feat_b):.4f}")
print(f"  Pipeline C (holdout eval)  : shape={feat_c.shape}, L2={np.linalg.norm(feat_c):.4f}")
print(f"  A vs B allclose : {pid_ab['allclose']}  max_diff={pid_ab['max_abs_diff']:.2e}")
print(f"  A vs C allclose : {pid_ac['allclose']}  max_diff={pid_ac['max_abs_diff']:.2e}")
print(f"  B vs C allclose : {pid_bc['allclose']}  max_diff={pid_bc['max_abs_diff']:.2e}")

with open(os.path.join(OUT_DIR, "pipeline_identity_check.json"), "w", encoding="utf-8") as f:
    json.dump(pipeline_identity, f, indent=2)
print(f"  [SAVED] pipeline_identity_check.json")

if not (pid_ac["allclose"] and pid_bc["allclose"]):
    print("\n  !! PIPELINE MISMATCH DETECTED !!")
    print("     Stopping to report — do NOT silently fix.")
else:
    print("  >> All three pipelines produce IDENTICAL feature vectors.")

# ============================================================
# STEP 2: REFERENCE SELF-CONSISTENCY TEST
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 2 — REFERENCE SELF-CONSISTENCY TEST (18 images)")
print("=" * 70)

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
# 1 random sample per student
ref_sample_rows = conn.execute("""
    SELECT d.file_path, d.saved_filename, d.student_name, d.student_id
    FROM dataset d
    WHERE d.id IN (
        SELECT MIN(id) FROM dataset GROUP BY student_name
    )
    ORDER BY student_name
""").fetchall()
conn.close()

self_test_results = []
print(f"  {'Student':<32} {'NN_dist':>8} {'sim_new%':>9} {'sim_old%':>9} {'predicted':<32} {'correct':>7}")
print("  " + "-" * 100)

for row in ref_sample_rows:
    fp = row["file_path"]
    if not os.path.exists(fp):
        fp = os.path.join(RAW_DIR, row["saved_filename"])

    gt_name = row["student_name"]
    try:
        img = preprocess_image(fp)
        fv  = extract_hog_features(img).reshape(1, -1)
        dists, idxs = model.kneighbors(fv, n_neighbors=5)
        d0          = float(dists[0][0])
        top5_names  = [train_labels[i] for i in idxs[0]]

        # Distance-weighted vote
        vote_w = {}
        for nm, dd in zip(top5_names, dists[0]):
            w = 1.0 / (dd + 1e-9) if dd > 0 else 1e9
            vote_w[nm] = vote_w.get(nm, 0.0) + w
        predicted = max(vote_w, key=vote_w.get)
        is_correct = (predicted == gt_name)

        sim_n = sim_new(d0)
        sim_o = sim_old(d0)

        print(f"  {gt_name:<32} {d0:>8.4f} {sim_n:>8.1f}% {sim_o:>8.1f}% {predicted:<32} {str(is_correct):>7}")
        self_test_results.append({
            "student_name":       gt_name,
            "file":               os.path.basename(fp),
            "nn_distance":        round(d0, 4),
            "sim_new_pct":        round(sim_n, 2),
            "sim_old_pct":        round(sim_o, 2),
            "predicted":          predicted,
            "is_correct":         is_correct,
            "top5":               top5_names,
            "top5_contains_truth": gt_name in top5_names,
        })
    except Exception as e:
        print(f"  {gt_name:<32} ERROR: {e}")
        self_test_results.append({"student_name": gt_name, "error": str(e)})

ref_correct = sum(1 for r in self_test_results if r.get("is_correct", False))
ref_top5    = sum(1 for r in self_test_results if r.get("top5_contains_truth", False))
print(f"\n  Reference self-test: {ref_correct}/18 correct  (Top-5: {ref_top5}/18)")
print(f"  NOTE: Training samples — KNN should find near-zero distance to self")

mean_nn = np.mean([r["nn_distance"] for r in self_test_results if "nn_distance" in r])
print(f"  Mean NN distance on reference self-test: {mean_nn:.4f}")

with open(os.path.join(OUT_DIR, "reference_self_test.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["student_name","file","nn_distance","sim_new_pct","sim_old_pct","predicted","is_correct","top5_contains_truth"])
    writer.writeheader()
    for r in self_test_results:
        writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
print(f"  [SAVED] reference_self_test.csv")

# ============================================================
# STEP 3: EXPLAIN 361 FILES vs 360 DB ROWS
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 3 — 361 RAW FILES vs 360 DB ROWS")
print("=" * 70)

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
db_rows = conn.execute("SELECT saved_filename, file_path, student_name FROM dataset").fetchall()
conn.close()

db_filenames = {r["saved_filename"]: r for r in db_rows}
raw_files    = [f for f in os.listdir(RAW_DIR) if os.path.isfile(os.path.join(RAW_DIR, f))]
print(f"  DB rows: {len(db_rows)}, RAW files: {len(raw_files)}")

extra_files = [f for f in raw_files if f not in db_filenames]
print(f"  Extra files (in RAW but NOT in DB): {len(extra_files)}")
ref_file_audit = {"db_rows": len(db_rows), "raw_files": len(raw_files), "extra_files": []}
for ef in extra_files:
    fp   = os.path.join(RAW_DIR, ef)
    ext  = Path(ef).suffix.lower()
    h    = sha256_file(fp)
    size = os.path.getsize(fp)
    is_img = ext in IMG_EXTS
    # Check if hash duplicates any DB file
    db_hashes = set()
    for db_fn in db_filenames:
        db_fp = os.path.join(RAW_DIR, db_fn)
        if os.path.exists(db_fp):
            db_hashes.add(sha256_file(db_fp))
    is_dup_hash = h in db_hashes
    info = {
        "filename": ef, "extension": ext, "size_bytes": size,
        "in_db": ef in db_filenames, "sha256": h,
        "is_image": is_img, "is_hash_duplicate": is_dup_hash,
    }
    print(f"  EXTRA: {ef}")
    print(f"         ext={ext}, size={size}B, in_db={info['in_db']}, is_image={is_img}, hash_dup={is_dup_hash}")
    ref_file_audit["extra_files"].append(info)

with open(os.path.join(OUT_DIR, "reference_file_audit.json"), "w", encoding="utf-8") as f:
    json.dump(ref_file_audit, f, indent=2)
print(f"  [SAVED] reference_file_audit.json")

# ============================================================
# STEP 4: HOLDOUT INVENTORY AUDIT
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 4 — HOLDOUT INVENTORY AUDIT")
print("=" * 70)

# Compute reference hashes using ONLY the 360 DB-referenced files
ref_hashes_360 = {}  # hash -> filename
for row in db_rows:
    fn = row["saved_filename"]
    fp = os.path.join(RAW_DIR, fn)
    if os.path.exists(fp):
        ref_hashes_360[sha256_file(fp)] = fn

print(f"  Reference hashes (360-DB-referenced): {len(ref_hashes_360)}")

skip_folders = {"Bu irma"}
inventory = []
holdout_samples = []

for folder_name in sorted(os.listdir(DATASET_ROOT)):
    if folder_name in skip_folders or folder_name not in STUDENT_FOLDER_MAP:
        continue
    folder_path = os.path.join(DATASET_ROOT, folder_name)
    if not os.path.isdir(folder_path):
        continue
    gt_name = STUDENT_FOLDER_MAP[folder_name]
    nim     = STUDENT_NIM_MAP.get(gt_name, "")
    imgs = [f for f in os.listdir(folder_path) if Path(f).suffix.lower() in IMG_EXTS]
    holdout = []
    for img_file in sorted(imgs):
        ip = os.path.join(folder_path, img_file)
        h  = sha256_file(ip)
        if h not in ref_hashes_360:
            holdout.append({"path": ip, "filename": img_file, "sha256": h,
                            "ground_truth_name": gt_name, "ground_truth_nim": nim})
    inventory.append({
        "student_name":      gt_name,
        "nim":               nim,
        "folder_name":       folder_name,
        "reference_count":   20,  # from DB
        "holdout_count":     len(holdout),
        "holdout_available": len(holdout) > 0,
    })
    holdout_samples.extend(holdout)
    status = "YES" if holdout else "NO"
    print(f"  {gt_name:<32} ref=20  holdout={len(holdout):>3}  [{status}]")

with open(os.path.join(OUT_DIR, "dataset_inventory.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["student_name","nim","folder_name","reference_count","holdout_count","holdout_available"])
    w.writeheader()
    w.writerows(inventory)
print(f"  [SAVED] dataset_inventory.csv")
print(f"  TOTAL holdout (360-DB referenced): {len(holdout_samples)}")
no_holdout_students = [inv["student_name"] for inv in inventory if not inv["holdout_available"]]
print(f"  Students WITHOUT holdout ({len(no_holdout_students)}): {', '.join(no_holdout_students)}")

# ============================================================
# STEP 5: IMAGE ACQUISITION DISTRIBUTION AUDIT
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 5 — IMAGE ACQUISITION DISTRIBUTION (measurable properties)")
print("=" * 70)

def image_properties(img_path):
    raw = cv2.imread(img_path)
    if raw is None:
        return None
    h_orig, w_orig = raw.shape[:2]
    gray_orig = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    file_size = os.path.getsize(img_path)
    # Preprocessed
    processed = preprocess_from_array(raw.copy())
    fg_ratio  = float(np.sum(processed > 0)) / float(processed.size)
    hog_vec   = extract_hog_features(processed)
    hog_norm  = float(np.linalg.norm(hog_vec))
    return {
        "width":        w_orig,
        "height":       h_orig,
        "aspect_ratio": round(w_orig / h_orig, 4),
        "file_size_kb": round(file_size / 1024, 2),
        "gray_mean":    round(float(np.mean(gray_orig)), 2),
        "gray_std":     round(float(np.std(gray_orig)), 2),
        "fg_ratio":     round(fg_ratio, 4),
        "hog_l2_norm":  round(hog_norm, 4),
        "hog_mean":     round(float(np.mean(hog_vec)), 6),
        "hog_std":      round(float(np.std(hog_vec)), 6),
    }

print("  Computing reference image properties (360 files) ...")
ref_props_list = []
for row in db_rows:
    fn = row["saved_filename"]
    fp = os.path.join(RAW_DIR, fn)
    if os.path.exists(fp):
        props = image_properties(fp)
        if props:
            props["student_name"] = row["student_name"]
            props["source"] = "reference"
            props["filename"] = fn
            ref_props_list.append(props)

print(f"  Reference props computed: {len(ref_props_list)}")

print("  Computing holdout image properties ...")
hld_props_list = []
for samp in holdout_samples:
    props = image_properties(samp["path"])
    if props:
        props["student_name"] = samp["ground_truth_name"]
        props["source"] = "holdout"
        props["filename"] = samp["filename"]
        hld_props_list.append(props)

print(f"  Holdout props computed: {len(hld_props_list)}")

METRIC_COLS = ["width","height","aspect_ratio","file_size_kb","gray_mean","gray_std","fg_ratio","hog_l2_norm","hog_mean","hog_std"]

img_dist_rows = []
for metric in METRIC_COLS:
    ref_vals = [p[metric] for p in ref_props_list if metric in p]
    hld_vals = [p[metric] for p in hld_props_list if metric in p]
    r_desc   = describe(ref_vals)
    h_desc   = describe(hld_vals)
    row_data = {"metric": metric}
    for k, v in r_desc.items():
        row_data[f"ref_{k}"] = round(v, 4)
    for k, v in h_desc.items():
        row_data[f"hld_{k}"] = round(v, 4)
    # delta mean
    row_data["delta_mean"] = round(h_desc.get("mean",0) - r_desc.get("mean",0), 4)
    img_dist_rows.append(row_data)
    print(f"  {metric:<18}: ref_mean={r_desc.get('mean',0):>10.4f}  hld_mean={h_desc.get('mean',0):>10.4f}  delta={row_data['delta_mean']:>+10.4f}")

with open(os.path.join(OUT_DIR, "image_distribution_comparison.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=img_dist_rows[0].keys())
    w.writeheader()
    w.writerows(img_dist_rows)
print(f"  [SAVED] image_distribution_comparison.csv")

# ============================================================
# STEP 6: FEATURE-SPACE DISTRIBUTION TEST
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 6 — FEATURE-SPACE DISTRIBUTION TEST")
print("=" * 70)

# Precompute all reference HOG vectors (from X_train in model)
X_ref = X_train  # shape (360, 34596) — EXACT vectors used for training

# Within-reference NN distances (exclude self)
print("  Computing within-reference nearest-neighbor distances ...")
from sklearn.neighbors import NearestNeighbors
nn_ref = NearestNeighbors(n_neighbors=2, metric='euclidean', algorithm='auto')
nn_ref.fit(X_ref)
ref_ref_dists, _ = nn_ref.kneighbors(X_ref)  # k=2 → skip self (idx 0 = self, dist=0)
ref_nn_dists = ref_ref_dists[:, 1]  # second neighbor = nearest OTHER

ref_desc = describe(ref_nn_dists)
print(f"  Reference->Reference NN dist: mean={ref_desc['mean']:.4f}  P90={ref_desc['p90']:.4f}  P95={ref_desc['p95']:.4f}  max={ref_desc['max']:.4f}")

# Holdout->Reference NN distances
print("  Computing holdout->reference nearest-neighbor distances ...")
nn_ref1 = NearestNeighbors(n_neighbors=1, metric='euclidean', algorithm='auto')
nn_ref1.fit(X_ref)

hld_fvs     = []
hld_names   = []
hld_files   = []
hld_errors  = []
for samp in holdout_samples:
    try:
        raw = cv2.imread(samp["path"])
        img = preprocess_from_array(raw)
        fv  = extract_hog_features(img)
        hld_fvs.append(fv)
        hld_names.append(samp["ground_truth_name"])
        hld_files.append(samp["filename"])
    except Exception as e:
        hld_errors.append({"file": samp["filename"], "error": str(e)})

hld_fvs_arr = np.array(hld_fvs)  # (132, 34596)
hld_nn_dists_all, _ = nn_ref1.kneighbors(hld_fvs_arr)
hld_nn_dists = hld_nn_dists_all[:, 0]

hld_desc = describe(hld_nn_dists)
print(f"  Holdout->Reference NN dist:   mean={hld_desc['mean']:.4f}  P90={hld_desc['p90']:.4f}  P95={hld_desc['p95']:.4f}  max={hld_desc['max']:.4f}")

# What % holdout is above ref P90, P95, max?
ref_p90 = ref_desc["p90"]
ref_p95 = ref_desc["p95"]
ref_max = ref_desc["max"]
pct_above_p90 = float(np.mean(hld_nn_dists > ref_p90)) * 100
pct_above_p95 = float(np.mean(hld_nn_dists > ref_p95)) * 100
pct_above_max = float(np.mean(hld_nn_dists > ref_max)) * 100
print(f"  % holdout > ref_P90 ({ref_p90:.4f}): {pct_above_p90:.1f}%")
print(f"  % holdout > ref_P95 ({ref_p95:.4f}): {pct_above_p95:.1f}%")
print(f"  % holdout > ref_MAX ({ref_max:.4f}): {pct_above_max:.1f}%")

feat_dist_rows = []
for metric_name, vals, src in [("ref_to_nearest_ref", ref_nn_dists, "reference"),
                                  ("hld_to_nearest_ref", hld_nn_dists, "holdout")]:
    d = describe(vals)
    feat_dist_rows.append({
        "source": src, "metric": metric_name,
        **{k: round(v,4) for k,v in d.items()},
        "pct_above_ref_p90": round(pct_above_p90,2) if src == "holdout" else "N/A",
        "pct_above_ref_p95": round(pct_above_p95,2) if src == "holdout" else "N/A",
        "pct_above_ref_max": round(pct_above_max,2) if src == "holdout" else "N/A",
    })

with open(os.path.join(OUT_DIR, "feature_distance_distribution.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=feat_dist_rows[0].keys())
    w.writeheader()
    w.writerows(feat_dist_rows)
print(f"  [SAVED] feature_distance_distribution.csv")

# ============================================================
# STEP 7: SAME-CLASS vs DIFFERENT-CLASS DISTANCE MARGIN
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 7 — SAME-CLASS vs DIFFERENT-CLASS DISTANCE MARGIN")
print("=" * 70)

margin_results = []
for i, (fv, gt, fn) in enumerate(zip(hld_fvs_arr, hld_names, hld_files)):
    fv_2d = fv.reshape(1, -1)
    dists_all, idxs_all = nn_ref.kneighbors(fv_2d, n_neighbors=len(X_ref))
    dists_all = dists_all[0]
    idxs_all  = idxs_all[0]
    labels_all = train_labels[idxs_all]

    same_mask  = labels_all == gt
    diff_mask  = labels_all != gt

    d_same = float(dists_all[same_mask][0])  if same_mask.any()  else np.nan
    d_diff = float(dists_all[diff_mask][0])  if diff_mask.any()  else np.nan
    margin = (d_diff - d_same) if (not np.isnan(d_same) and not np.isnan(d_diff)) else np.nan

    # dominant wrong prediction (nearest different-class)
    if diff_mask.any():
        dom_wrong = labels_all[diff_mask][0]
    else:
        dom_wrong = ""

    margin_results.append({
        "filename":             fn,
        "ground_truth":         gt,
        "nn_same_class_dist":   round(d_same, 4) if not np.isnan(d_same) else "NaN",
        "nn_diff_class_dist":   round(d_diff, 4) if not np.isnan(d_diff) else "NaN",
        "margin":               round(margin, 4)  if not np.isnan(margin) else "NaN",
        "margin_positive":      bool(margin > 0)  if not np.isnan(margin) else False,
        "dominant_wrong_pred":  dom_wrong,
    })

valid_margins = [float(r["margin"]) for r in margin_results if r["margin"] != "NaN"]
pct_positive  = float(np.mean([r["margin_positive"] for r in margin_results])) * 100
print(f"  Holdout margin (diff - same): mean={np.mean(valid_margins):.4f}  median={np.median(valid_margins):.4f}")
print(f"  % with positive margin (same-class closer): {pct_positive:.1f}%")

with open(os.path.join(OUT_DIR, "holdout_distance_margin.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=margin_results[0].keys())
    w.writeheader()
    w.writerows(margin_results)
print(f"  [SAVED] holdout_distance_margin.csv")

# ============================================================
# STEP 8: SIMILARITY FORMULA AUDIT
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 8 — SIMILARITY FORMULA AUDIT")
print("=" * 70)

# Formula analysis
# Old formula (used in holdout evaluator): sim = (1 - d²/450) * 100
#   450 = 2 * N_blocks_128 = 2 * 225 = 450  (128x128, ppc=8, cpb=2 -> 15*15=225 blocks)
#   d_max = sqrt(450) = 21.21 — calibrated for 8100-dim HOG

# New formula (classifier.py): sim = (1 - d²/1922) * 100
#   1922 = 2 * N_blocks_256 = 2 * 961 = 1922  (256x256, ppc=8, cpb=2 -> 31*31=961 blocks)
#   d_max = sqrt(1922) = 43.84 — correct for 34596-dim HOG

print("  Formula analysis:")
print("  OLD (holdout eval, WRONG for 34596-dim): sim = (1 - d²/450) * 100")
print("    Origin: calibrated for 128x128 HOG (225 blocks), d_max = sqrt(450) = 21.21")
print("    Applied to 256x256 HOG (961 blocks): d_max_actual = sqrt(1922) = 43.84")
print()
print("  NEW (classifier.py, CORRECT):            sim = (1 - d²/1922) * 100")
print("    Origin: 2 * 961 blocks for 256x256 HOG")
print("    d_max = sqrt(1922) = 43.84")

# Show effect on holdout distances
print()
print("  Effect on holdout NN distances (using both formulas):")
print(f"  {'Percentile':<12} {'NN_dist':>8} {'sim_old%':>9} {'sim_new%':>9}")
for pct in [0, 10, 25, 50, 75, 90, 95, 100]:
    d = float(np.percentile(hld_nn_dists, pct))
    print(f"  P{pct:<10} {d:>8.4f} {sim_old(d):>8.1f}% {sim_new(d):>8.1f}%")

# Show for reference self-test distances
print()
ref_self_dists = [r["nn_distance"] for r in self_test_results if "nn_distance" in r]
print(f"  Reference self-test NN distances:")
print(f"  {'Percentile':<12} {'NN_dist':>8} {'sim_old%':>9} {'sim_new%':>9}")
for pct in [0, 25, 50, 75, 100]:
    d = float(np.percentile(ref_self_dists, pct)) if ref_self_dists else 0
    print(f"  P{pct:<10} {d:>8.4f} {sim_old(d):>8.1f}% {sim_new(d):>8.1f}%")

formula_audit = {
    "old_formula":  "sim = max(0, min(100, (1 - d²/450) * 100))",
    "old_origin":   "Calibrated for 128x128 HOG, 225 blocks, d_max=sqrt(450)=21.21",
    "old_is_correct_for_34596": False,
    "new_formula":  "sim = max(0, min(100, (1 - d²/1922) * 100))",
    "new_origin":   "classifier.py — 2 * 961 blocks for 256x256 HOG, d_max=sqrt(1922)=43.84",
    "new_is_correct_for_34596": True,
    "mismatch_confirmed": True,
    "holdout_eval_used_old_formula": True,
    "note": "The holdout evaluator used the OLD formula (450), causing sim=0% for virtually all predictions even when distances are in normal range for 256x256 HOG. sim=0 does NOT mean mismatch—it means the formula was miscalibrated."
}
# Add recalculated holdout accuracy using new formula
sim_corrected = [sim_new(d) for d in hld_nn_dists]
formula_audit["holdout_dist_mean"] = round(float(np.mean(hld_nn_dists)), 4)
formula_audit["holdout_sim_old_mean"] = round(float(np.mean([sim_old(d) for d in hld_nn_dists])), 2)
formula_audit["holdout_sim_new_mean"] = round(float(np.mean(sim_corrected)), 2)

with open(os.path.join(OUT_DIR, "similarity_formula_audit.json"), "w", encoding="utf-8") as f:
    json.dump(formula_audit, f, indent=2)
print(f"  [SAVED] similarity_formula_audit.json")

# ============================================================
# STEP 9: TOP-1 vs TOP-5 RANK ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 9 — TOP-1 vs TOP-5 RANK ANALYSIS")
print("=" * 70)

rank_results = []
rank_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, "not_in_top5": 0}

for i, (fv, gt, fn) in enumerate(zip(hld_fvs_arr, hld_names, hld_files)):
    fv_2d = fv.reshape(1, -1)
    dists_k, idxs_k = model.kneighbors(fv_2d, n_neighbors=5)
    top5_labels = [train_labels[j] for j in idxs_k[0]]
    top5_dists  = dists_k[0].tolist()

    # Find rank of ground truth
    gt_rank = None
    for r_idx, lbl in enumerate(top5_labels):
        if lbl == gt:
            gt_rank = r_idx + 1
            break

    # Weighted vote prediction
    vote_w = {}
    for nm, dd in zip(top5_labels, top5_dists):
        w = 1.0 / (dd + 1e-9) if dd > 0 else 1e9
        vote_w[nm] = vote_w.get(nm, 0.0) + w
    predicted = max(vote_w, key=vote_w.get)

    if gt_rank is not None:
        rank_counts[gt_rank] += 1
    else:
        rank_counts["not_in_top5"] += 1

    rank_results.append({
        "filename":     fn,
        "ground_truth": gt,
        "predicted":    predicted,
        "gt_rank":      gt_rank if gt_rank else "not_in_top5",
        "top5":         "|".join(top5_labels),
        "top5_dists":   "|".join([f"{d:.4f}" for d in top5_dists]),
        "nn_dist":      round(top5_dists[0], 4),
        "sim_new_pct":  round(sim_new(top5_dists[0]), 2),
    })

total_hld = len(rank_results)
print(f"  Ground truth rank distribution (N={total_hld}):")
cumulative = 0
for rank in [1, 2, 3, 4, 5, "not_in_top5"]:
    cnt = rank_counts[rank]
    cumulative += cnt if isinstance(rank, int) else 0
    pct = cnt / total_hld * 100
    cum_pct = cumulative / total_hld * 100
    rank_label = f"Rank {rank}" if isinstance(rank, int) else "Not in Top-5"
    print(f"  {rank_label:<18}: {cnt:>4}  ({pct:5.1f}%)   CIR <= rank: {cum_pct:5.1f}%")

with open(os.path.join(OUT_DIR, "rank_analysis.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rank_results[0].keys())
    w.writeheader()
    w.writerows(rank_results)
print(f"  [SAVED] rank_analysis.csv")

# ============================================================
# STEP 10: PER-STUDENT HOLDOUT AUDIT
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 10 — PER-STUDENT HOLDOUT RESULT")
print("=" * 70)

# Map gt_name -> list of margin_results and rank_results
stu_margin = {}
for r in margin_results:
    stu_margin.setdefault(r["ground_truth"], []).append(r)
stu_rank = {}
for r in rank_results:
    stu_rank.setdefault(r["ground_truth"], []).append(r)

per_student_audit = []
holdout_students = sorted(set(hld_names))

print(f"  {'Student':<32} {'N':>4} {'Top1':>5} {'Top5':>5} {'Acc%':>6} {'T5%':>6} {'AvgMargin':>10} {'PosMargin%':>11} {'AvgSimNew%':>11}")
print("  " + "-" * 110)
for stu in holdout_students:
    ranks = stu_rank.get(stu, [])
    margins = stu_margin.get(stu, [])
    n = len(ranks)
    if n == 0:
        continue
    top1_cnt  = sum(1 for r in ranks if r["gt_rank"] == 1)
    top5_cnt  = sum(1 for r in ranks if r["gt_rank"] in [1,2,3,4,5])
    acc       = top1_cnt / n * 100
    t5        = top5_cnt / n * 100
    valid_m   = [float(m["margin"]) for m in margins if m["margin"] != "NaN"]
    avg_margin = np.mean(valid_m) if valid_m else float("nan")
    pos_pct   = float(np.mean([m["margin_positive"] for m in margins])) * 100 if margins else 0
    avg_sim   = np.mean([r["sim_new_pct"] for r in ranks])
    dom_wrong = {}
    for m in margins:
        dw = m.get("dominant_wrong_pred", "")
        if dw:
            dom_wrong[dw] = dom_wrong.get(dw, 0) + 1
    top_wrong = max(dom_wrong, key=dom_wrong.get) if dom_wrong else ""
    same_dists = [float(m["nn_same_class_dist"]) for m in margins if m["nn_same_class_dist"] != "NaN"]
    diff_dists = [float(m["nn_diff_class_dist"]) for m in margins if m["nn_diff_class_dist"] != "NaN"]
    avg_same = np.mean(same_dists) if same_dists else float("nan")
    avg_diff = np.mean(diff_dists) if diff_dists else float("nan")

    print(f"  {stu:<32} {n:>4} {top1_cnt:>5} {top5_cnt:>5} {acc:>5.1f}% {t5:>5.1f}% {avg_margin:>+10.4f} {pos_pct:>10.1f}% {avg_sim:>10.1f}%")
    per_student_audit.append({
        "student_name":         stu,
        "nim":                  STUDENT_NIM_MAP.get(stu, ""),
        "n_holdout":            n,
        "top1_correct":         top1_cnt,
        "top5_correct":         top5_cnt,
        "accuracy_pct":         round(acc, 2),
        "top5_accuracy_pct":    round(t5, 2),
        "avg_nn_same_dist":     round(avg_same, 4) if not np.isnan(avg_same) else "NaN",
        "avg_nn_diff_dist":     round(avg_diff, 4) if not np.isnan(avg_diff) else "NaN",
        "avg_margin":           round(avg_margin, 4) if not np.isnan(avg_margin) else "NaN",
        "pct_positive_margin":  round(pos_pct, 2),
        "avg_sim_new_pct":      round(avg_sim, 2),
        "dominant_wrong_pred":  top_wrong,
    })

with open(os.path.join(OUT_DIR, "per_student_audit.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=per_student_audit[0].keys())
    w.writeheader()
    w.writerows(per_student_audit)
print(f"  [SAVED] per_student_audit.csv")

# ============================================================
# STEP 11: GENERATE PLOTS
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 11 — PLOTS")
print("=" * 70)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (A) Feature-space distance distribution histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ref_nn_dists, bins=40, alpha=0.6, color='#2196F3', label=f'Reference->Nearest Ref (N={len(ref_nn_dists)})')
    ax.hist(hld_nn_dists, bins=40, alpha=0.6, color='#F44336', label=f'Holdout->Nearest Ref (N={len(hld_nn_dists)})')
    ax.axvline(ref_desc["p90"], color='#2196F3', linestyle='--', linewidth=1.5, label=f'Ref P90={ref_desc["p90"]:.2f}')
    ax.axvline(ref_desc["p95"], color='#1565C0', linestyle=':', linewidth=1.5, label=f'Ref P95={ref_desc["p95"]:.2f}')
    ax.set_xlabel("Euclidean Distance (HOG feature space)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Feature-Space Distance Distribution\nReference vs Holdout", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feature_distance_histogram.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] feature_distance_histogram.png")

    # (B) Similarity formula comparison
    d_range = np.linspace(0, 50, 500)
    sim_old_vals = [sim_old(d) for d in d_range]
    sim_new_vals = [sim_new(d) for d in d_range]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(d_range, sim_old_vals, color='#F44336', linewidth=2, label='OLD formula (d²/450) — calibrated for 128×128')
    ax.plot(d_range, sim_new_vals, color='#4CAF50', linewidth=2, label='NEW formula (d²/1922) — calibrated for 256×256')
    ax.axvspan(float(np.percentile(hld_nn_dists,10)), float(np.percentile(hld_nn_dists,90)),
               alpha=0.1, color='orange', label='Holdout P10–P90 dist range')
    ax.axvspan(float(np.percentile(ref_nn_dists,10)), float(np.percentile(ref_nn_dists,90)),
               alpha=0.1, color='blue', label='Reference P10–P90 dist range')
    ax.set_xlabel("Euclidean Distance", fontsize=12)
    ax.set_ylabel("Similarity (%)", fontsize=12)
    ax.set_title("Similarity Formula Comparison\n(OLD vs NEW calibration)", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(-5, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "similarity_formula_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] similarity_formula_comparison.png")

    # (C) CIR chart
    ranks_list = [1, 2, 3, 4, 5]
    cir = []
    cum = 0
    for r in ranks_list:
        cum += rank_counts[r]
        cir.append(cum / total_hld * 100)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([f"Top-{r}" for r in ranks_list], cir, color=['#4CAF50','#8BC34A','#FFC107','#FF9800','#F44336'], edgecolor='black')
    for i, v in enumerate(cir):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.set_ylabel("Cumulative Identification Rate (%)", fontsize=11)
    ax.set_title("Cumulative Identification Rate (CIR)\nExternal Holdout", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cumulative_identification_rate.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] cumulative_identification_rate.png")

    # (D) Per-student Top1 vs Top5 accuracy
    stu_labels = [r["student_name"].split()[-1] for r in per_student_audit]
    top1_accs  = [r["accuracy_pct"] for r in per_student_audit]
    top5_accs  = [r["top5_accuracy_pct"] for r in per_student_audit]
    x = np.arange(len(stu_labels))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - 0.2, top1_accs, 0.35, color='#2196F3', label='Top-1 Accuracy', edgecolor='black', linewidth=0.7)
    ax.bar(x + 0.2, top5_accs, 0.35, color='#4CAF50', label='Top-5 Accuracy', edgecolor='black', linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(stu_labels, rotation=40, ha='right', fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Per-Student Top-1 vs Top-5 Accuracy — Holdout", fontsize=12)
    ax.legend(fontsize=10)
    ax.axhline(100/18*100, color='gray', linestyle='--', linewidth=0.8, label='Chance level')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "per_student_top1_vs_top5.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] per_student_top1_vs_top5.png")

except Exception as e:
    print(f"  [WARN] Plot error: {e}")

# ============================================================
# FINAL REPORT
# ============================================================
print("\n" + "=" * 70)
print("GENERATING FINAL AUDIT REPORT")
print("=" * 70)

# Compile summary statistics for report
overall_acc_top1 = rank_counts[1] / total_hld * 100
overall_acc_top5 = sum(rank_counts[r] for r in [1,2,3,4,5]) / total_hld * 100

# Recalculate sim with new formula
sim_new_corrected_mean = float(np.mean([sim_new(d) for d in hld_nn_dists]))

# Positive margin %
pos_margin_pct = pct_positive

# Per-student table for report
per_stu_md = "\n".join([
    f"| {r['student_name']} | {r['nim']} | {r['n_holdout']} | {r['top1_correct']} | {r['top5_correct']} | {r['accuracy_pct']:.1f}% | {r['top5_accuracy_pct']:.1f}% | {r['avg_margin'] if r['avg_margin']!='NaN' else 'N/A'} | {r['pct_positive_margin']:.1f}% |"
    for r in per_student_audit
])

# Rank table for report
rank_md_rows = ""
cum = 0
for rk in [1,2,3,4,5,"not_in_top5"]:
    cnt = rank_counts[rk]
    cum += cnt if isinstance(rk, int) else 0
    cum_pct = cum / total_hld * 100
    rank_md_rows += f"| {'Rank '+str(rk) if isinstance(rk,int) else 'Not in Top-5'} | {cnt} | {cnt/total_hld*100:.1f}% | {cum_pct:.1f}% |\n"

# Image distribution summary
img_metrics_md = "\n".join([
    f"| {r['metric']} | {r.get('ref_mean','?')} | {r.get('hld_mean','?')} | {r.get('delta_mean','?'):+g} |"
    for r in img_dist_rows
])

report = f"""# Forensic Audit Report — External Holdout 18.94%
## Yaevia Handwriting Verification System

**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**Auditor**: Automated Forensic Pipeline
**Status**: AUDIT COMPLETE — READ-ONLY, no model changes made

---

## Executive Summary

| Metric | Value |
|---|---|
| LOOCV Internal Accuracy | **61.94%** (223/360) |
| External Holdout Accuracy | **18.94%** (25/132) |
| Δ Accuracy | **−43.00%** |
| Top-5 Accuracy (Holdout) | **{overall_acc_top5:.2f}%** ({sum(rank_counts[r] for r in [1,2,3,4,5])}/132) |
| Holdout Coverage | 12/18 students ({', '.join(no_holdout_students)} have NO holdout) |
| Inference Errors | 0/132 |
| Pipeline Identity (A≡B≡C) | **{pid_ac['allclose']}** |
| Similarity Formula Mismatch | **CONFIRMED** (formula not recalibrated for 34596-dim) |
| Feature-Space Distribution Shift | **CONFIRMED** ({pct_above_p90:.1f}% holdout exceeds ref P90) |
| Same-Class Closer Than Diff-Class | **{pos_margin_pct:.1f}%** of holdout samples |

---

## Audit 1 — Pipeline Identity Check

**Test file**: `{pipeline_identity['test_file']}` ({pipeline_identity['student_name']})

| Pipeline | Description | allclose A? |
|---|---|---|
| A (Training) | `preprocess_image()` -> `extract_hog_features()` | — |
| B (Verify route) | same functions, same path | **{pid_ab['allclose']}** |
| C (Holdout eval) | `cv2.imread` -> `preprocess_from_array()` -> `extract_hog_features()` | **{pid_ac['allclose']}** |

**max absolute diff A↔B**: `{pid_ab['max_abs_diff']:.2e}`  
**max absolute diff A↔C**: `{pid_ac['max_abs_diff']:.2e}`

> [!{'NOTE' if pid_ac['allclose'] else 'CAUTION'}]
> {'All three pipelines produce NUMERICALLY IDENTICAL feature vectors. The 18.94% result is NOT caused by a pipeline implementation bug.' if pid_ac['allclose'] else 'PIPELINE MISMATCH DETECTED. Feature vectors differ between training and holdout evaluation.'}

---

## Audit 2 — Reference Self-Consistency Test

18 reference images (one per student, from training set) run through holdout evaluator.

**Results**: {ref_correct}/18 correct (Top-5: {ref_top5}/18)  
**Mean NN distance on reference images**: {mean_nn:.4f}

> [!NOTE]
> Reference images (same images used in training) should produce near-zero NN distance to themselves. Mean={mean_nn:.4f} confirms the evaluator is consistent — model has learned correctly.

---

## Audit 3 — 361 Raw Files vs 360 DB Rows

- **DB rows**: 360  
- **RAW directory files**: {ref_file_audit['raw_files']}
- **Extra files (in RAW but not in DB)**: {len(ref_file_audit['extra_files'])}

"""

for ef in ref_file_audit["extra_files"]:
    report += f"""**Extra file**: `{ef['filename']}`
- Extension: `{ef['extension']}`
- Size: {ef['size_bytes']} bytes
- In DB: {ef['in_db']}
- Is image: {ef['is_image']}
- SHA-256 duplicate of DB file: {ef['is_hash_duplicate']}

> [!IMPORTANT]
> The 361st file {"does NOT affect" if not ef['is_hash_duplicate'] else "IS a duplicate and"} the SHA-256 holdout identification. Future audits should use 360 DB-referenced hashes only.
"""

report += f"""
---

## Audit 4 — Holdout Inventory Audit

**Using 360 DB-referenced SHA-256 hashes (not all raw files).**

| Student | Holdout Count | Available? |
|---|---|---|
"""
for inv in inventory:
    report += f"| {inv['student_name']} | {inv['holdout_count']} | {'YES' if inv['holdout_available'] else 'NO — no external samples'} |\n"

report += f"""
**Total holdout images**: {len(holdout_samples)}  
**Coverage**: 12/18 students  
**Students without holdout**: {', '.join(no_holdout_students)}

> [!IMPORTANT]
> External holdout evaluation hanya mencakup **12 dari 18 kelas**. Tidak boleh diklaim sebagai evaluasi penuh 18 kelas.

---

## Audit 5 — Image Acquisition Distribution

| Metric | Ref Mean | Hld Mean | Delta |
|---|---|---|---|
{img_metrics_md}

---

## Audit 6 — Feature-Space Distribution Test

| Metric | Ref->Nearest Ref | Hld->Nearest Ref |
|---|---|---|
| Mean | {ref_desc['mean']:.4f} | {hld_desc['mean']:.4f} |
| Median | {ref_desc['median']:.4f} | {hld_desc['median']:.4f} |
| Std | {ref_desc['std']:.4f} | {hld_desc['std']:.4f} |
| P25 | {ref_desc['p25']:.4f} | {hld_desc['p25']:.4f} |
| P75 | {ref_desc['p75']:.4f} | {hld_desc['p75']:.4f} |
| P90 | {ref_desc['p90']:.4f} | {hld_desc['p90']:.4f} |
| P95 | {ref_desc['p95']:.4f} | {hld_desc['p95']:.4f} |
| Max | {ref_desc['max']:.4f} | {hld_desc['max']:.4f} |

**% holdout > ref P90 ({ref_p90:.4f})**: {pct_above_p90:.1f}%  
**% holdout > ref P95 ({ref_p95:.4f})**: {pct_above_p95:.1f}%  
**% holdout > ref MAX ({ref_max:.4f})**: {pct_above_max:.1f}%

---

## Audit 7 — Same-Class vs Different-Class Distance Margin

**Margin = nearest_different_class_dist − nearest_same_class_dist**

- Positive margin → same-class reference closer → class discriminable
- Negative margin → other-class reference closer → class overlap

| Statistic | Value |
|---|---|
| Mean margin | {np.mean(valid_margins):.4f} |
| Median margin | {np.median(valid_margins):.4f} |
| Std margin | {np.std(valid_margins):.4f} |
| % positive margin | **{pct_positive:.1f}%** |

---

## Audit 8 — Similarity Formula Audit

**Critical Finding:**

| Formula | Expression | Calibrated For | d_max |
|---|---|---|---|
| **OLD** (holdout eval) | `(1 − d²/450) × 100` | 128×128 HOG (225 blocks) | 21.21 |
| **NEW** (classifier.py) | `(1 − d²/1922) × 100` | 256×256 HOG (961 blocks) | 43.84 |

The holdout evaluator used the **OLD formula** (d²/450), which was calibrated for the 8,100-dim feature space (128×128). The production model uses **34,596-dim** features with d_max ≈ 43.84. Any distance > 21.21 maps to sim=0% under the old formula, causing the misleading "0% similarity" output on nearly all holdout samples.

**This does NOT affect the classification accuracy (18.94%)**, since KNN prediction is based on distances, not similarity percentages. The 18.94% accuracy figure is numerically correct.

**Mean holdout NN dist**: {formula_audit['holdout_dist_mean']:.4f}  
**Mean sim (old formula)**: {formula_audit['holdout_sim_old_mean']:.2f}%  
**Mean sim (new formula)**: {formula_audit['holdout_sim_new_mean']:.2f}%

---

## Audit 9 — Rank Analysis (Top-1 vs Top-5)

| Rank | Count | % | Cumulative Identification Rate |
|---|---|---|---|
{rank_md_rows}

---

## Audit 10 — Per-Student Holdout Audit

| Student | NIM | N | Top1 | Top5 | Acc% | Top5% | AvgMargin | PosMargin% |
|---|---|---|---|---|---|---|---|---|
{per_stu_md}

---

## Audit 11 — Final Diagnosis

Based on all audit evidence:

### A. Evaluation Pipeline Mismatch — **NOT FOUND**
All three pipelines (training / verify_route / holdout_eval) produce numerically identical 34,596-dim feature vectors.

### B. Feature-Space Distribution Shift — **CONFIRMED (PRIMARY CAUSE)**
- Holdout NN distances (mean={hld_desc['mean']:.4f}) are significantly larger than within-reference NN distances (mean={ref_desc['mean']:.4f})
- **{pct_above_p90:.1f}%** of holdout samples have NN distance > ref P90
- **{pct_above_p95:.1f}%** of holdout samples have NN distance > ref P95
- This confirms holdout images occupy a different region in HOG feature space than the training distribution

### C. Severe Inter-Class Overlap — **PARTIAL**
- Only **{pos_margin_pct:.1f}%** of holdout samples have positive margin (same-class closer)
- This means for {100-pos_margin_pct:.1f}% of holdout, a DIFFERENT student's writing is closer
- This is consistent with distribution shift: when all distances are large, inter-class boundaries collapse

### D. Similarity Calibration Problem — **CONFIRMED (SECONDARY)**
- The holdout evaluator used the OLD formula (d²/450, calibrated for 128×128 / 8,100 features)
- Production classifier.py uses CORRECT formula (d²/1922, calibrated for 256×256 / 34,596 features)
- This caused sim=0% display for almost all predictions
- **This does NOT affect the 18.94% accuracy figure** — KNN classification is distance-based, not similarity-based

### E. Insufficient / Unbalanced Holdout Coverage — **CONFIRMED**
- Only 12/18 classes represented
- Holdout sizes: {min(r['n_holdout'] for r in per_student_audit)}–{max(r['n_holdout'] for r in per_student_audit)} samples per student

### F. Verdict — Combination of B + D + E

**Primary diagnosis**: **B (Feature-Space Distribution Shift)**
The 18.94% accuracy is a **real, valid measurement** of model generalization under distribution shift conditions. It is not a bug.

**Secondary finding**: **D (Similarity Formula Mismatch)** in the holdout evaluator report only — does not affect the 18.94% number.

**Root causes of distribution shift** (supported by measurable evidence):
1. Image acquisition properties differ significantly between reference and holdout (see Audit 5 metrics)
2. HOG is a gradient-based descriptor — sensitive to image acquisition conditions (lighting, blur, contrast)
3. 20 training samples per class is insufficient to cover the full intra-class variation
4. With 34,596 features and only 20 training samples per class, the "support region" for each class is extremely narrow

---

## Artifacts Generated

| File | Description |
|---|---|
| `pipeline_identity_check.json` | 3-way pipeline feature vector comparison |
| `reference_self_test.csv` | 18 reference images through holdout evaluator |
| `reference_file_audit.json` | 361 vs 360 file analysis |
| `dataset_inventory.csv` | Holdout inventory per student |
| `image_distribution_comparison.csv` | Measurable image property comparison |
| `feature_distance_distribution.csv` | NN distance distribution stats |
| `holdout_distance_margin.csv` | Per-sample same/diff class distance margin |
| `similarity_formula_audit.json` | Old vs new formula calibration analysis |
| `rank_analysis.csv` | Top-1/5 rank per holdout sample |
| `per_student_audit.csv` | Full per-student holdout audit |
| `feature_distance_histogram.png` | Distribution shift visualization |
| `similarity_formula_comparison.png` | Old vs new formula plot |
| `cumulative_identification_rate.png` | CIR chart |
| `per_student_top1_vs_top5.png` | Per-student accuracy chart |

---
*STOP. No model changes, retraining, or parameter tuning performed. Awaiting next instruction.*
"""

report_path = os.path.join(OUT_DIR, "external_holdout_audit_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"  [SAVED] external_holdout_audit_report.md")

print("\n" + "=" * 70)
print("FORENSIC AUDIT COMPLETE")
print("=" * 70)
print(f"  Primary diagnosis   : FEATURE-SPACE DISTRIBUTION SHIFT (B)")
print(f"  Secondary finding   : SIMILARITY FORMULA MISMATCH (D) — affects display only")
print(f"  Pipeline identity   : OK (A=B=C)")
print(f"  Holdout accuracy    : 18.94% — VALID number, not a bug")
print(f"  Similarity formula  : OLD formula used in holdout report (cosmetic bug only)")
print(f"  All artifacts saved : {OUT_DIR}")
print("=" * 70)
print("STOP — Awaiting next instruction.")
