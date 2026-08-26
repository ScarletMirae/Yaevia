"""
model/trainer.py — Pipeline Training Model KNN (Diperluas)
============================================================
Modul ini mengimplementasikan pipeline pelatihan model KNN lengkap
sesuai dengan metodologi skripsi BAB IV.

Alur training (Pipeline):
    Upload Dataset
        -> Load & Preprocessing Citra
        -> Ekstraksi Fitur HOG (batch)
        -> Feature Vector Matrix
        -> Validasi Dataset (min 2 sampel per mahasiswa)
        -> Train/Test Split 80:20 (stratified)
        -> Training KNN (Euclidean Distance, Weighted)
        -> Evaluasi Model (Accuracy, Precision, Recall, F1)
        -> Simpan Model (.joblib)
        -> Simpan Training Features (untuk kneighbors di inference)
        -> Simpan Metadata JSON (untuk Model Information)
        -> Simpan Metadata ke Database

BAB IV -- Implementasi:
    Metode K-Nearest Neighbor (KNN) dipilih karena:
    - Algoritma non-parametrik, tidak membuat asumsi distribusi data
    - Sederhana namun efektif untuk klasifikasi tulisan tangan
    - Interpretable: keputusan berdasarkan kedekatan di ruang fitur HOG
    - Metric Euclidean distance mengukur 'jarak' antar feature vector

Referensi:
    Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification.
    IEEE Transactions on Information Theory, 13(1), 21-27.
"""

import os
import sys
import json
import time
import numpy as np
import joblib
from datetime import datetime
from collections import Counter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_SAVED_DIR, DATASET_RAW_DIR, DATASET_PROCESSED_DIR,
    KNN_N_NEIGHBORS, KNN_METRIC, KNN_WEIGHTS, KNN_ALGORITHM,
    HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK, HOG_BLOCK_NORM,
    TEST_SIZE, RANDOM_STATE, IMAGE_SIZE,
    MIN_SAMPLES_PER_CLASS,
)
from preprocessing.image_processor import preprocess_image
from features.hog_extractor import extract_hog_features
from database import get_connection, deactivate_all_models

# Alias untuk kompatibilitas (MODEL_DIR lama -> MODEL_SAVED_DIR baru)
MODEL_DIR = MODEL_SAVED_DIR


# ==============================================================================
# VALIDASI DATASET
# ==============================================================================

def validate_dataset_min_samples() -> dict:
    """
    BAB IV - Validasi Dataset Sebelum Training:

    Memastikan setiap mahasiswa (kelas) memiliki minimal MIN_SAMPLES_PER_CLASS
    gambar tulisan tangan sebelum training dimulai.

    Alasan: KNN memerlukan minimal 2 sampel per kelas untuk dapat membentuk
    ruang fitur yang representatif dan melakukan train/test split yang valid.

    Returns:
        dict:
            valid (bool): True jika semua kelas memenuhi syarat
            insufficient_students (list): Nama mahasiswa yang kurang sampel
            counts (dict): Jumlah sampel per mahasiswa
            total (int): Total dataset
            n_classes (int): Jumlah kelas/mahasiswa
            message (str): Pesan hasil validasi
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT student_name, COUNT(*) as cnt FROM dataset GROUP BY student_name"
    ).fetchall()
    total_row = conn.execute("SELECT COUNT(*) as cnt FROM dataset").fetchone()
    conn.close()

    counts               = {r["student_name"]: r["cnt"] for r in rows}
    insufficient         = [name for name, cnt in counts.items() if cnt < MIN_SAMPLES_PER_CLASS]
    total                = total_row["cnt"] if total_row else 0
    n_classes            = len(counts)

    if insufficient:
        return {
            "valid": False,
            "insufficient_students": insufficient,
            "counts": counts,
            "total": total,
            "n_classes": n_classes,
            "message": (
                f"Training dibatalkan. Terdapat {len(insufficient)} mahasiswa yang memiliki "
                f"kurang dari {MIN_SAMPLES_PER_CLASS} sampel tulisan tangan. "
                "Silakan lengkapi dataset terlebih dahulu."
            ),
        }

    if total == 0:
        return {
            "valid": False,
            "insufficient_students": [],
            "counts": {},
            "total": 0,
            "n_classes": 0,
            "message": "Dataset kosong. Upload citra tulisan tangan terlebih dahulu.",
        }

    if n_classes < 2:
        return {
            "valid": False,
            "insufficient_students": [],
            "counts": counts,
            "total": total,
            "n_classes": n_classes,
            "message": "Diperlukan minimal 2 mahasiswa berbeda untuk training.",
        }

    return {
        "valid": True,
        "insufficient_students": [],
        "counts": counts,
        "total": total,
        "n_classes": n_classes,
        "message": f"Validasi OK: {total} citra dari {n_classes} mahasiswa siap untuk training.",
    }


# ==============================================================================
# LOAD DATASET DARI DATABASE & DISK
# ==============================================================================

def load_dataset_from_db() -> tuple:
    """
    BAB IV - Memuat Dataset Tulisan Tangan:

    Membaca seluruh citra dari database dan melakukan preprocessing.
    Preprocessing mencakup: grayscale -> thresholding -> normalisasi -> resize.

    Returns:
        tuple:
            images (list of ndarray): Citra terpreproses (128x128 grayscale)
            labels (list of str): Nama mahasiswa per citra
            student_ids (list of str): NIM mahasiswa per citra
            mata_kuliah_list (list of str): Mata kuliah per citra

    Raises:
        ValueError: Jika dataset kosong atau tidak cukup data
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT file_path, student_name, student_id, mata_kuliah FROM dataset"
    ).fetchall()
    conn.close()

    if not rows:
        raise ValueError("Dataset kosong! Upload minimal 2 citra dari 2 mahasiswa berbeda.")

    images           = []
    labels           = []
    student_ids      = []
    mata_kuliah_list = []
    failed           = []

    for i, row in enumerate(rows):
        file_path    = row["file_path"]
        student_name = row["student_name"]
        nim          = row["student_id"] or ""
        mk           = row["mata_kuliah"]

        # Resolusi path dinamis jika path absolut lama berubah (misal folder project di-rename)
        if not os.path.exists(file_path):
            basename = os.path.basename(file_path)
            candidate_path = os.path.join(DATASET_RAW_DIR, basename)
            if os.path.exists(candidate_path):
                file_path = candidate_path

        try:
            # Preprocessing: grayscale -> blur -> threshold -> denoise -> ROI -> letterbox 128x128
            processed = preprocess_image(file_path)
            images.append(processed)
            labels.append(student_name)
            student_ids.append(nim)
            mata_kuliah_list.append(mk)
        except Exception as e:
            print(f"[TRAINER] Skip file {file_path}: {e}")
            failed.append(i)

    if len(images) < 2:
        raise ValueError(
            f"Jumlah citra valid terlalu sedikit ({len(images)}). "
            "Diperlukan minimal 2 citra dari 2 kelas berbeda."
        )

    print(f"[TRAINER] Dataset dimuat: {len(images)} citra, {len(set(labels))} kelas")
    return images, labels, student_ids, mata_kuliah_list


# ==============================================================================
# PIPELINE TRAINING UTAMA
# ==============================================================================

def train_model(
    n_neighbors: int  = KNN_N_NEIGHBORS,
    metric: str       = KNN_METRIC,
    weights: str      = KNN_WEIGHTS,
    orientations: int       = HOG_ORIENTATIONS,
    pixels_per_cell: tuple  = HOG_PIXELS_PER_CELL,
    cells_per_block: tuple  = HOG_CELLS_PER_BLOCK,
    test_size: float  = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    BAB IV - Pipeline Training Model KNN Lengkap:

    Menjalankan 9 tahap pipeline sesuai metodologi skripsi:
        1. Validasi Dataset (min 2 sampel/kelas)
        2. Load Dataset dari Database
        3. Preprocessing Citra (grayscale, thresholding, resize)
        4. Ekstraksi Fitur HOG (batch)
        5. Encode Label (LabelEncoder)
        6. Train/Test Split 80:20 (stratified)
        7. Training KNN (fit dengan X_train)
        8. Evaluasi Model (Accuracy, Precision, Recall, F1)
        9. Simpan Model, Training Features, dan Metadata

    Args:
        n_neighbors (int): Nilai K untuk KNN (default dari config)
        metric (str): Metrik jarak ('euclidean' | 'manhattan' | 'minkowski')
        weights (str): Bobot KNN ('uniform' | 'distance')
        orientations (int): Jumlah orientasi HOG
        pixels_per_cell (tuple): Ukuran sel HOG (px, px)
        cells_per_block (tuple): Ukuran blok HOG (sel, sel)
        test_size (float): Proporsi data uji (0.0-1.0), default 0.2
        random_state (int): Seed reproduksibilitas

    Returns:
        dict: Hasil training dengan semua metrik dan metadata
    """
    print("=" * 65)
    print("[TRAINER] Memulai pipeline training...")
    print("=" * 65)
    t_pipeline_start = time.time()

    # =========================================================
    # LANGKAH 1: VALIDASI DATASET
    # =========================================================
    print("[TRAINER] Langkah 1/9: Validasi dataset...")
    validation = validate_dataset_min_samples()
    if not validation["valid"]:
        return {
            "success": False,
            "message": validation["message"],
            "insufficient_students": validation.get("insufficient_students", []),
            "counts": validation.get("counts", {}),
        }
    print(f"[TRAINER]   Validasi OK: {validation['total']} citra, {validation['n_classes']} kelas")

    # =========================================================
    # LANGKAH 2: LOAD DATASET
    # =========================================================
    print("[TRAINER] Langkah 2/9: Memuat dataset dari database...")
    images, labels, student_ids, mata_kuliah_list = load_dataset_from_db()

    # =========================================================
    # LANGKAH 3: PREPROCESSING (sudah dilakukan di load_dataset)
    # =========================================================
    print("[TRAINER] Langkah 3/9: Preprocessing selesai (grayscale, threshold, resize).")

    # =========================================================
    # LANGKAH 4: EKSTRAKSI FITUR HOG (BATCH)
    # =========================================================
    print("[TRAINER] Langkah 4/9: Mengekstraksi fitur HOG...")
    print(f"[TRAINER]   Parameter HOG: orientations={orientations}, "
          f"pixels_per_cell={pixels_per_cell}, cells_per_block={cells_per_block}")

    X_list       = []
    valid_labels = []

    for i, (img, lbl) in enumerate(zip(images, labels)):
        try:
            # Ekstraksi Histogram of Oriented Gradients (HOG)
            # Output: feature vector 1D dengan panjang bergantung parameter HOG
            feat = extract_hog_features(
                img,
                orientations    = orientations,
                pixels_per_cell = pixels_per_cell,
                cells_per_block = cells_per_block,
                visualize       = False,
            )
            X_list.append(feat)
            valid_labels.append(lbl)
        except Exception as e:
            print(f"[TRAINER]   Skip indeks {i}: {e}")

    X = np.array(X_list)       # Feature matrix (n_samples, n_features)
    y = valid_labels
    feature_size = X.shape[1] if len(X) > 0 else 0
    print(f"[TRAINER]   Feature matrix: {X.shape} "
          f"(panjang feature vector: {feature_size} dimensi)")

    # =========================================================
    # LANGKAH 5: ENCODE LABEL
    # =========================================================
    print("[TRAINER] Langkah 5/9: Encoding label mahasiswa...")
    le = LabelEncoder()
    y_encoded   = le.fit_transform(y)
    class_names = list(le.classes_)
    print(f"[TRAINER]   {len(class_names)} kelas: {class_names}")

    # =========================================================
    # LANGKAH 6: TRAIN/TEST SPLIT 80:20 (STRATIFIED)
    # =========================================================
    # Stratified split memastikan distribusi kelas yang proporsional
    # di data training dan data testing, mencegah bias evaluasi.
    print(f"[TRAINER] Langkah 6/9: Train/Test Split "
          f"{int((1-test_size)*100)}:{int(test_size*100)} (stratified)...")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size    = test_size,
            random_state = random_state,
            stratify     = y_encoded,   # Stratified: distribusi kelas proporsional
        )
    except ValueError:
        # Fallback jika stratified tidak bisa (kelas dengan 1 sampel)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size    = test_size,
            random_state = random_state,
        )

    print(f"[TRAINER]   Data training : {len(X_train)} sampel ({int((1-test_size)*100)}%)")
    print(f"[TRAINER]   Data testing  : {len(X_test)} sampel ({int(test_size*100)}%)")

    # =========================================================
    # LANGKAH 7: TRAINING KNN
    # =========================================================
    # KNN dilatih dengan data training menggunakan Euclidean Distance
    # sebagai metric jarak antar feature vector HOG.
    print(f"[TRAINER] Langkah 7/9: Melatih KNN "
          f"(K={n_neighbors}, metric={metric}, weights={weights})...")
    t_train_start = time.time()

    knn = KNeighborsClassifier(
        n_neighbors = n_neighbors,
        metric      = metric,
        weights     = weights,
        algorithm   = KNN_ALGORITHM,
    )
    knn.fit(X_train, y_train)   # KNN 'training' = menyimpan X_train dan y_train

    training_time = round(time.time() - t_train_start, 3)
    print(f"[TRAINER]   KNN training selesai dalam {training_time}s")

    # =========================================================
    # LANGKAH 8: EVALUASI MODEL
    # =========================================================
    # Evaluasi pada data testing yang tidak dilihat selama training.
    # Metrik: Accuracy, Precision (macro), Recall (macro), F1 (macro)
    print("[TRAINER] Langkah 8/9: Evaluasi model...")

    y_train_pred = knn.predict(X_train)
    y_test_pred  = knn.predict(X_test)

    train_acc = float(accuracy_score(y_train, y_train_pred))
    test_acc  = float(accuracy_score(y_test,  y_test_pred))

    # Macro averaging: rata-rata per kelas tanpa mempertimbangkan jumlah sampel
    # Sesuai untuk dataset yang mungkin tidak seimbang
    precision = float(precision_score(y_test, y_test_pred, average="macro", zero_division=0))
    recall    = float(recall_score   (y_test, y_test_pred, average="macro", zero_division=0))
    f1        = float(f1_score       (y_test, y_test_pred, average="macro", zero_division=0))

    print(f"[TRAINER]   Akurasi data latih  : {train_acc*100:.2f}%")
    print(f"[TRAINER]   Akurasi data uji    : {test_acc*100:.2f}%")
    print(f"[TRAINER]   Precision (macro)   : {precision*100:.2f}%")
    print(f"[TRAINER]   Recall    (macro)   : {recall*100:.2f}%")
    print(f"[TRAINER]   F1 Score  (macro)   : {f1*100:.2f}%")

    # Confusion matrix (untuk endpoint evaluasi)
    cm = confusion_matrix(y_test, y_test_pred)

    # =========================================================
    # LANGKAH 9: SIMPAN MODEL, DATA, DAN METADATA
    # =========================================================
    print("[TRAINER] Langkah 9/9: Menyimpan model dan metadata...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Path file-file yang disimpan
    model_filename   = f"knn_model_{timestamp}.joblib"
    le_filename      = f"label_encoder_{timestamp}.joblib"
    Xtrain_filename  = f"train_features_{timestamp}.joblib"
    ytrain_filename  = f"train_labels_{timestamp}.joblib"
    Xtest_filename   = f"test_features_{timestamp}.joblib"
    ytest_filename   = f"test_labels_{timestamp}.joblib"
    meta_filename    = f"metadata_{timestamp}.json"

    model_path  = os.path.join(MODEL_DIR, model_filename)
    le_path     = os.path.join(MODEL_DIR, le_filename)
    Xtrain_path = os.path.join(MODEL_DIR, Xtrain_filename)
    ytrain_path = os.path.join(MODEL_DIR, ytrain_filename)
    Xtest_path  = os.path.join(MODEL_DIR, Xtest_filename)
    ytest_path  = os.path.join(MODEL_DIR, ytest_filename)
    meta_path   = os.path.join(MODEL_DIR, meta_filename)
    latest_meta_path = os.path.join(MODEL_DIR, "latest_metadata.json")

    # Simpan model KNN
    joblib.dump(knn, model_path)

    # Simpan label encoder
    joblib.dump(le, le_path)

    # Simpan training features (WAJIB untuk kneighbors() di inference)
    # Tanpa ini, classifier.py tidak bisa menghitung Euclidean Distance per kelas
    joblib.dump(X_train, Xtrain_path)
    joblib.dump(y_train, ytrain_path)

    # Simpan testing features (untuk evaluasi ulang tanpa training)
    joblib.dump(X_test, Xtest_path)
    joblib.dump(y_test, ytest_path)

    print(f"[TRAINER]   Model disimpan : {model_path}")
    print(f"[TRAINER]   Training data  : {Xtrain_path}")
    print(f"[TRAINER]   Testing data   : {Xtest_path}")

    # Total waktu pipeline
    total_time = round(time.time() - t_pipeline_start, 3)

    # Hitung distribusi sampel per kelas
    label_counts = Counter(valid_labels)

    # Susun metadata lengkap (untuk Model Information di dashboard)
    metadata = {
        # Informasi Responden
        "n_respondents":       len(class_names),
        "class_names":         class_names,
        "label_counts":        dict(label_counts),

        # Informasi Dataset
        "n_total_dataset":     len(X),
        "n_train_samples":     len(X_train),
        "n_test_samples":      len(X_test),
        "test_size":           test_size,
        "train_size":          round(1.0 - test_size, 2),

        # Parameter HOG
        "hog_orientations":    orientations,
        "hog_pixels_per_cell": list(pixels_per_cell),
        "hog_cells_per_block": list(cells_per_block),
        "hog_block_norm":      HOG_BLOCK_NORM,
        "feature_vector_size": feature_size,
        "image_size":          list(IMAGE_SIZE),

        # Parameter KNN
        "knn_k":               n_neighbors,
        "knn_metric":          metric,
        "knn_weights":         weights,

        # Metrik Evaluasi
        "train_accuracy":      round(train_acc * 100, 4),
        "test_accuracy":       round(test_acc  * 100, 4),
        "precision_macro":     round(precision * 100, 4),
        "recall_macro":        round(recall    * 100, 4),
        "f1_macro":            round(f1        * 100, 4),

        # Waktu
        "training_time_seconds": total_time,
        "train_timestamp":       datetime.now().isoformat(),

        # Path file
        "model_path":            model_path,
        "le_path":               le_path,
        "train_features_path":   Xtrain_path,
        "test_features_path":    Xtest_path,
        "timestamp":             timestamp,
    }

    # Simpan metadata sebagai JSON
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Simpan juga sebagai latest (mudah diakses oleh API)
    with open(latest_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[TRAINER]   Metadata disimpan: {meta_path}")

    # Simpan ke database
    deactivate_all_models()
    conn = get_connection()
    conn.execute("""
        INSERT INTO model_meta (
            model_filename, model_path,
            train_accuracy, test_accuracy, precision_score, recall_score, f1_score,
            n_train_samples, n_test_samples, n_classes, n_total_dataset,
            knn_k, knn_metric, knn_weights,
            hog_orientations, hog_pixels_per_cell, hog_cells_per_block, hog_block_norm,
            feature_vector_size, test_size, training_time_seconds,
            metadata_json_path, train_timestamp, is_active
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)
    """, (
        model_filename, model_path,
        train_acc, test_acc, precision, recall, f1,
        len(X_train), len(X_test), len(class_names), len(X),
        n_neighbors, metric, weights,
        orientations, str(pixels_per_cell), str(cells_per_block), HOG_BLOCK_NORM,
        feature_size, test_size, total_time,
        meta_path, datetime.now().isoformat(),
    ))
    conn.commit()
    conn.close()

    print("=" * 65)
    print(f"[TRAINER] Training selesai! Akurasi uji: {test_acc*100:.2f}% | Waktu: {total_time}s")
    print("=" * 65)

    return {
        "success":           True,
        "message":           "Model berhasil dilatih",
        "model_path":        model_path,
        "timestamp":         timestamp,

        # Metrik
        "train_accuracy":    round(train_acc * 100, 2),
        "test_accuracy":     round(test_acc  * 100, 2),
        "precision":         round(precision * 100, 2),
        "recall":            round(recall    * 100, 2),
        "f1_score":          round(f1        * 100, 2),

        # Dataset info
        "n_train":           len(X_train),
        "n_test":            len(X_test),
        "n_classes":         len(class_names),
        "n_total":           len(X),
        "class_names":       class_names,
        "feature_vector_size": feature_size,
        "knn_k":             n_neighbors,
        "metric":            metric,
        "training_time":     total_time,
    }


# ==============================================================================
# FUNGSI HELPER
# ==============================================================================

def get_latest_model_paths() -> dict:
    """
    Mengambil path model terakhir yang tersimpan dari database.

    Returns:
        dict: Path model dan file pendukung, atau None jika belum ada model.
    """
    conn = get_connection()
    row  = conn.execute(
        "SELECT model_path, metadata_json_path FROM model_meta "
        "WHERE is_active=1 ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()

    if not row:
        return None

    model_path = row["model_path"]
    row_dict   = dict(row)
    timestamp  = os.path.basename(model_path).replace("knn_model_", "").replace(".joblib", "")

    return {
        "model_path":    model_path,
        "le_path":       os.path.join(MODEL_DIR, f"label_encoder_{timestamp}.joblib"),
        "Xtrain_path":   os.path.join(MODEL_DIR, f"train_features_{timestamp}.joblib"),
        "ytrain_path":   os.path.join(MODEL_DIR, f"train_labels_{timestamp}.joblib"),
        "Xtest_path":    os.path.join(MODEL_DIR, f"test_features_{timestamp}.joblib"),
        "ytest_path":    os.path.join(MODEL_DIR, f"test_labels_{timestamp}.joblib"),
        "meta_path":     row_dict.get("metadata_json_path") or os.path.join(MODEL_DIR, "latest_metadata.json"),
        "timestamp":     timestamp,
    }


def get_latest_metadata() -> dict:
    """
    Membaca metadata model terlatih terbaru dari file JSON.
    Digunakan oleh endpoint /api/model/info.

    Returns:
        dict: Metadata model atau None jika belum ada.
    """
    latest_path = os.path.join(MODEL_DIR, "latest_metadata.json")
    if not os.path.exists(latest_path):
        return None
    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
