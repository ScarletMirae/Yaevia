"""
model/classifier.py — Klasifikasi KNN Berbasis Euclidean Distance & HOG
========================================================================
Modul ini mengimplementasikan proses verifikasi tulisan tangan menggunakan
K-Nearest Neighbor (KNN, K=5, distance-weighted) dan pengukuran kemiripan
sampel geometris berbasis Euclidean Distance pada ruang fitur HOG.

BAB IV — Implementasi Verifikasi:
    Proses verifikasi dilakukan dengan langkah berikut:
    1. Feature vector HOG diterima dari modul ekstraksi fitur
    2. Model KNN menghitung tetangga terdekat menggunakan kneighbors() dan
       menentukan kelas prediksi via distance-weighted majority voting (K=5).
    3. Bobot voting (Vote Share) dihitung melalui predict_proba() sebagai
       proporsi kontribusi bobot (w = 1/d) pada lingkungan K=5.
    4. Jarak Euclidean minimum (d_min) ke sampel terdekat dari kelas pemenang
       dikonversi ke Sample Similarity (%) menggunakan formula Cosine/Normalized HOG:
       similarity(%) = max(0.0, min(100.0, (1 - (d^2 / 450)) * 100))
    5. Status kemiripan ditentukan berdasarkan threshold di config.py
    6. Daftar Top Kandidat diurutkan berdasarkan konsensus voting KNN (primer:
       vote_weight desc, sekunder: distance asc).

CATATAN METODOLOGI:
    - KNN Weighted Vote Share: Menunjukkan proporsi konsensus voting KNN (K=5).
    - Sample Similarity: Menunjukkan kedekatan geometris query terhadap sampel
      terdekat dari suatu kelas.
    - Keduanya disajikan secara transparan dan berdampingan.

Referensi:
    Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification.
    IEEE Transactions on Information Theory, 13(1), 21-27.
    Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for
    human detection. CVPR.
"""

import os
import sys
import time
import glob
import json
import logging
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_SAVED_DIR, SIMILARITY_THRESHOLDS
from database import get_connection

logger = logging.getLogger(__name__)


# ==============================================================================
# FUNGSI SIMILARITY BERBASIS EUCLIDEAN DISTANCE
# ==============================================================================

def euclidean_to_similarity(distance: float, max_dist_sq: float = 450.0) -> float:
    """
    BAB IV - Konversi Euclidean Distance ke Similarity Score (Cosine / Normalized HOG Space):

    Dasar Metodologi:
        Pada HOG dengan normalisasi blok L2-Hys (128x128, 8x8 pixels_per_cell, 2x2 cells_per_block),
        vektor fitur memiliki N_blocks = 225 blok ternormalisasi L2.
        Kuadrat panjang vektor: ||x||^2 ≈ 225.0.

        Hubungan antara Euclidean Distance (d) dan Cosine Similarity (cos_sim) untuk vektor L2-norm:
            d^2 = ||x||^2 + ||y||^2 - 2*(x·y) = 225 + 225 - 2*(225 * cos_sim) = 450 * (1 - cos_sim)
            cos_sim = 1 - (d^2 / 450)

        Formula Similarity (%):
            similarity(%) = max(0.0, min(100.0, (1 - (distance^2 / 450.0)) * 100))

    Sifat Formula:
        - distance = 0.00   -> similarity = 100.0% (identik sempurna)
        - distance ≈ 11.44  -> similarity ≈ 70.9%  (sangat mirip / same writer)
        - distance ≈ 14.27  -> similarity ≈ 54.8%  (mirip / variasi wajar satu penulis)
        - distance >= 21.21 -> similarity = 0.0%   (ortogonal / totally dissimilar)
        - Monoton menurun proporsional terhadap ruang fitur HOG

    Args:
        distance (float): Euclidean Distance antara dua feature vector HOG.
        max_dist_sq (float): Kuadrat jarak maksimum teoritis (2 * N_blocks = 450.0).

    Returns:
        float: Similarity Score dalam persentase (0.0 - 100.0).
    """
    if distance <= 0.0:
        return 100.0
    sim_ratio = 1.0 - (float(distance) ** 2) / float(max_dist_sq)
    return round(float(np.clip(sim_ratio * 100.0, 0.0, 100.0)), 2)


def get_similarity_status(similarity_pct: float) -> str:
    """
    BAB IV - Klasifikasi Status Kemiripan Tulisan Tangan:

    Mengklasifikasikan similarity score menjadi 4 kategori kemiripan.
    Threshold dapat dikonfigurasi di config.py -> SIMILARITY_THRESHOLDS.

    Threshold default (berbasis kalibrasi eksperimental):
        >= 70% -> SANGAT MIRIP   (highly likely same writer)
        >= 50% -> MIRIP          (possibly same writer)
        >= 30% -> KURANG MIRIP   (uncertain)
        <  30% -> TIDAK MIRIP    (likely different writer)

    Args:
        similarity_pct (float): Similarity Score dalam persen.

    Returns:
        str: Status kemiripan ('SANGAT MIRIP' | 'MIRIP' | 'KURANG MIRIP' | 'TIDAK MIRIP')
    """
    t = SIMILARITY_THRESHOLDS
    if similarity_pct >= t.get("sangat_mirip", 70.0):
        return "SANGAT MIRIP"
    elif similarity_pct >= t.get("mirip", 50.0):
        return "MIRIP"
    elif similarity_pct >= t.get("kurang_mirip", 30.0):
        return "KURANG MIRIP"
    else:
        return "TIDAK MIRIP"


# ==============================================================================
# FUNGSI KLASIFIKASI UTAMA
# ==============================================================================

def classify_handwriting(
    query_feature: np.ndarray,
    knn_model,
    label_encoder,
    X_train: np.ndarray,
    y_train_encoded: np.ndarray,
) -> dict:
    """
    BAB IV - Proses Klasifikasi K-Nearest Neighbor + Euclidean Distance:

    Pipeline klasifikasi lengkap:
    1. Reshape feature vector query menjadi bentuk (1, n_features)
    2. KNN predict(): distance-weighted majority voting K tetangga -> kelas prediksi
    3. KNN predict_proba(): hitung proporsi bobot voting (vote share) untuk tiap kelas
    4. KNN kneighbors(): hitung Euclidean Distance ke K tetangga terdekat
    5. Hitung jarak sampel minimum (d_min) dan Sample Similarity (%) untuk tiap kelas
    6. Urutkan Top Kandidat:
       - Primer: KNN Vote Share (%) descending (pemenang voting selalu #1)
       - Sekunder: Euclidean Distance minimum ascending
    7. Parameter utama (distance, similarity, status) diambil dari sampel terbaik milik predicted_name.

    Pemisahan Dua Metrik:
    - KNN Weighted Vote Share (%): Menunjukkan persentase perolehan bobot voting KNN (K=5).
    - Sample Similarity (%): Menunjukkan kemiripan geometris sampel terdekat berbasis normalized HOG distance.

    Args:
        query_feature: Feature vector HOG dari citra query (shape: (n_features,))
        knn_model: Model KNeighborsClassifier yang sudah difit
        label_encoder: LabelEncoder untuk decode nama mahasiswa
        X_train: Matrix fitur data training (shape: (n_samples, n_features))
        y_train_encoded: Label terenkode data training (shape: (n_samples,))

    Returns:
        dict: Hasil klasifikasi lengkap dengan keys:
            - predicted_name (str): Nama mahasiswa yang diprediksi oleh KNN voting
            - predicted_vote_weight (float): Persentase perolehan voting KNN (0.0 - 100.0%)
            - euclidean_distance (float): Jarak Euclidean ke sampel terdekat milik predicted_name
            - similarity_percent (float): Sample similarity score 0-100%
            - similarity_status (str): Kategori kemiripan
            - k_neighbors (int): Nilai K yang digunakan
            - top_matches (list): Top-5 kandidat per kelas dengan vote_percent & distance
            - k_distances (list): Jarak ke K tetangga terdekat
    """
    # --- Step 1: Siapkan feature vector query ---
    query = np.array(query_feature).reshape(1, -1)

    # --- Step 2: Prediksi kelas dengan KNN (voting mayoritas K tetangga) ---
    predicted_label = knn_model.predict(query)[0]
    predicted_name  = label_encoder.inverse_transform([predicted_label])[0]

    # --- Step 3: Probabilitas/bobot voting KNN untuk seluruh kelas ---
    probs = knn_model.predict_proba(query)[0]
    prob_dict = {cls_idx: float(probs[i]) for i, cls_idx in enumerate(knn_model.classes_)}

    # --- Step 4: Ambil K-nearest neighbors dan jarak Euclideannya ---
    k_distances_arr, k_indices_arr = knn_model.kneighbors(query)
    k_distances = [round(float(d), 4) for d in k_distances_arr[0]]

    # --- Step 5: Evaluasi jarak terbaik (minimum distance) per kelas ---
    unique_labels = np.unique(y_train_encoded)
    class_results = []

    for lbl in unique_labels:
        class_mask     = (y_train_encoded == lbl)
        class_features = X_train[class_mask]   # shape: (n_class_samples, n_features)

        # Hitung Euclidean Distance dari query ke setiap sampel kelas ini
        diffs = class_features - query          # broadcasting
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))   # Euclidean: sqrt(sum((xi-yi)^2))

        # Ambil jarak minimum (sampel paling dekat dari kelas ini)
        min_dist_class = float(np.min(dists))
        sim_class      = euclidean_to_similarity(min_dist_class)
        class_name     = label_encoder.inverse_transform([lbl])[0]
        raw_vote_prob  = prob_dict.get(lbl, 0.0)
        vote_percent   = round(float(raw_vote_prob) * 100.0, 2)

        class_results.append({
            "name":         class_name,
            "distance":     round(min_dist_class, 4),
            "percent":      round(sim_class, 2),
            "vote_weight":  round(raw_vote_prob, 4),
            "vote_percent": vote_percent,
            "label_id":     lbl,
        })

    # Urutkan kandidat secara konsisten dengan mekanisme KNN:
    # 1. Primer: Bobot voting KNN (descending) -> Pemenang voting (predicted_name) SELALU peringkat #1
    # 2. Sekunder: Jarak minimum sampel (ascending)
    class_results.sort(key=lambda x: (-x["vote_weight"], x["distance"]))
    top_matches = class_results[:5]

    # --- Step 6: Parameter utama dihitung langsung dari sampel milik predicted_name (Top #1) ---
    top1 = top_matches[0]
    predicted_distance    = float(top1["distance"])
    predicted_similarity  = float(top1["percent"])
    predicted_vote_weight = float(top1["vote_percent"])
    status = get_similarity_status(predicted_similarity)

    # Format top_matches untuk output JSON (tanpa label_id internal)
    clean_top_matches = [
        {
            "name":         m["name"],
            "distance":     m["distance"],
            "percent":      m["percent"],          # Sample Similarity %
            "vote_percent": m["vote_percent"],     # KNN Weighted Vote Share %
            "vote_weight":  m["vote_weight"],      # Normalized vote share (0.0 - 1.0)
        }
        for m in top_matches
    ]

    return {
        "predicted_name":        predicted_name,
        "predicted_vote_weight": predicted_vote_weight,
        "euclidean_distance":    round(predicted_distance, 4),
        "similarity_percent":    round(predicted_similarity, 2),
        "similarity_status":     status,
        "k_neighbors":           int(knn_model.n_neighbors),
        "top_matches":           clean_top_matches,
        "k_distances":           k_distances,
    }


# ==============================================================================
# LOAD MODEL DARI DISK
# ==============================================================================

def _load_latest_model_files() -> tuple:
    """
    Memuat model KNN terbaru dan file pendukungnya dari direktori saved.

    Returns:
        tuple: (knn_model, label_encoder, X_train, y_train, timestamp)
               Raises FileNotFoundError jika belum ada model.
    """
    model_files = glob.glob(os.path.join(MODEL_SAVED_DIR, "knn_model_*.joblib"))
    if not model_files:
        raise FileNotFoundError(
            "Belum ada model terlatih. Jalankan training terlebih dahulu."
        )

    # Ambil model terbaru berdasarkan waktu modifikasi file
    latest_model = max(model_files, key=os.path.getmtime)
    timestamp    = os.path.basename(latest_model).replace("knn_model_", "").replace(".joblib", "")

    encoder_path = os.path.join(MODEL_SAVED_DIR, f"label_encoder_{timestamp}.joblib")
    Xtrain_path  = os.path.join(MODEL_SAVED_DIR, f"train_features_{timestamp}.joblib")
    ytrain_path  = os.path.join(MODEL_SAVED_DIR, f"train_labels_{timestamp}.joblib")

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"Label encoder tidak ditemukan untuk model {timestamp}. Lakukan training ulang."
        )
    if not os.path.exists(Xtrain_path) or not os.path.exists(ytrain_path):
        raise FileNotFoundError(
            f"Data training tidak ditemukan untuk model {timestamp}. Lakukan training ulang."
        )

    knn_model     = joblib.load(latest_model)
    label_encoder = joblib.load(encoder_path)
    X_train       = joblib.load(Xtrain_path)
    y_train       = joblib.load(ytrain_path)

    return knn_model, label_encoder, X_train, y_train, timestamp


# ==============================================================================
# FUNGSI VERIFIKASI UTAMA (dipanggil dari API)
# ==============================================================================

def verify_image(
    feature_vector: np.ndarray,
    model_version: str = None,
    query_filename: str = "unknown",
    query_path: str = "",
) -> dict:
    """
    BAB IV - Fungsi Utama Verifikasi Gambar:

    Dipanggil dari api/verify_routes.py setelah preprocessing & HOG extraction.

    Pipeline yang dipanggil:
        feature_vector HOG
            -> load model KNN + training features
            -> classify_handwriting() [KNN + Euclidean Distance]
            -> similarity score + status
            -> simpan ke database verifications
            -> return hasil lengkap

    Args:
        feature_vector: Feature vector HOG dari gambar query.
        model_version: Identifier versi model (optional, untuk logging DB).
        query_filename: Nama file gambar asli.
        query_path: Path lengkap file gambar.

    Returns:
        dict: Hasil verifikasi dengan semua field yang diperlukan frontend.
    """
    try:
        knn_model, label_encoder, X_train, y_train, timestamp = _load_latest_model_files()
    except FileNotFoundError as e:
        return {"success": False, "message": str(e)}
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return {"success": False, "message": f"Gagal memuat model: {str(e)}"}

    # Catat waktu mulai analisis
    t_start = time.time()

    try:
        # Jalankan klasifikasi KNN + Euclidean Distance
        result = classify_handwriting(
            feature_vector, knn_model, label_encoder, X_train, y_train
        )
    except Exception as e:
        logger.error(f"Error classifying: {e}")
        return {"success": False, "message": f"Gagal klasifikasi: {str(e)}"}

    # Hitung total waktu analisis
    analysis_time = round(time.time() - t_start, 4)
    result["analysis_time_seconds"] = analysis_time
    result["feature_vector_length"] = len(feature_vector)

    # Konversi top_matches ke JSON string untuk database
    top_matches_json = json.dumps(result["top_matches"], ensure_ascii=False)

    # Simpan hasil ke database
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO verifications (
                query_filename, query_path, predicted_name,
                similarity_percent, euclidean_distance, verification_status, similarity_status,
                top_matches_json, model_version, feature_vector_length, knn_k,
                analysis_time, verification_timestamp
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,datetime('now','localtime'))
        """, (
            query_filename,
            query_path,
            result["predicted_name"],
            result["similarity_percent"],
            result["euclidean_distance"],
            result["similarity_status"],   # status utama
            result["similarity_status"],   # kolom legacy
            top_matches_json,
            model_version or timestamp,
            result["feature_vector_length"],
            result["k_neighbors"],
            analysis_time,
        ))
        result["verification_id"] = cur.lastrowid
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Gagal simpan ke database: {e}")
        result["verification_id"] = None

    result["success"]       = True
    result["model_version"] = model_version or timestamp

    return result
