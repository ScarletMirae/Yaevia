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
       dikonversi ke Sample Similarity (%) menggunakan formula Normalized HOG Space:
       similarity(%) = max(0.0, min(100.0, (1 - (d^2 / 1922)) * 100))
       [256x256 HOG: 961 blok L2-Hys, max_dist_sq = 2 x 961 = 1922]
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
from config import (
    MODEL_SAVED_DIR, SIMILARITY_THRESHOLDS, VERIFICATION_CONFIDENCE_THRESHOLDS,
)
from database import get_connection
from model.trainer import get_latest_model_paths
from model.verifier import verify_claim

logger = logging.getLogger(__name__)


EXPECTED_FEATURE_DIM = 34596
EXPECTED_K = 5
EXPECTED_METRIC = "euclidean"


# ==============================================================================
# FUNGSI SIMILARITY BERBASIS EUCLIDEAN DISTANCE
# ==============================================================================

def euclidean_to_similarity(distance: float, max_dist_sq: float = None, feature_dim: int = None) -> float:
    """
    BAB IV - Konversi Euclidean Distance ke Similarity Score (Cosine / Normalized HOG Space):

    Dasar Metodologi:
        Pada HOG 256x256 dengan pixels_per_cell (8,8) dan cells_per_block (2,2),
        vektor fitur memiliki N_blocks = 31 x 31 = 961 blok ternormalisasi L2-Hys.
        Kuadrat panjang vektor: ||x||^2 ≈ 961.0.
        Kuadrat jarak maksimum teoritis (vektor ortogonal): max_dist_sq = 2 * N_blocks = 1922.0.

        Formula Similarity (%):
            similarity(%) = max(0.0, min(100.0, (1 - (distance^2 / max_dist_sq)) * 100))

    Args:
        distance (float): Euclidean Distance antara dua feature vector HOG.
        max_dist_sq (float, optional): Kuadrat jarak maksimum teoritis (2 * N_blocks).
        feature_dim (int, optional): Panjang vektor fitur untuk menghitung N_blocks secara dinamis.

    Returns:
        float: Similarity Score dalam persentase (0.0 - 100.0).
    """
    if distance <= 0.0:
        return 100.0

    if max_dist_sq is None:
        if feature_dim is not None and feature_dim > 0:
            n_blocks = feature_dim / 36.0
            max_dist_sq = 2.0 * n_blocks
        else:
            # Default frozen 256x256 representation (34,596 features / 961 blocks): 2 * 961 = 1922.0
            max_dist_sq = 1922.0

    sim_ratio = 1.0 - (float(distance) ** 2) / float(max_dist_sq)
    return round(float(np.clip(sim_ratio * 100.0, 0.0, 100.0)), 2)


def get_similarity_status(similarity_pct: float) -> str:
    """
    BAB IV - Klasifikasi Status Kemiripan Tulisan Tangan:

    Mengklasifikasikan similarity score menjadi 4 kategori kemiripan geometris.
    Threshold dikonfigurasi di config.py -> SIMILARITY_THRESHOLDS.

    Threshold default:
        >= 65% -> SANGAT MIRIP   (d <= 12.55)
        >= 50% -> MIRIP          (d <= 15.00)
        >= 40% -> KURANG MIRIP   (d <= 16.43)
        <  40% -> TIDAK MIRIP    (d > 16.43)

    Args:
        similarity_pct (float): Similarity Score dalam persen.

    Returns:
        str: Status kemiripan ('SANGAT MIRIP' | 'MIRIP' | 'KURANG MIRIP' | 'TIDAK MIRIP')
    """
    t = SIMILARITY_THRESHOLDS
    if similarity_pct >= t.get("sangat_mirip", 65.0):
        return "SANGAT MIRIP"
    elif similarity_pct >= t.get("mirip", 50.0):
        return "MIRIP"
    elif similarity_pct >= t.get("kurang_mirip", 40.0):
        return "KURANG MIRIP"
    else:
        return "TIDAK MIRIP"


def get_verification_status(
    similarity_pct: float,
    vote_share_pct: float = None,
    top_matches: list = None,
) -> str:
    """
    BAB IV — Mekanisme Keputusan Verifikasi / Identifikasi (Uncertainty & Rejection):

    Memisahkan status keputusan verifikasi dari sekadar kemiripan geometris:
    - TERIDENTIFIKASI    : similarity >= min_similarity_accept (50.0%) DAN vote_share >= min_vote_share_accept (35.0%)
    - TIDAK PASTI        : similarity antara 40.0% - 50.0% ATAU vote share terbagi/ambigu
    - TIDAK TERIDENTIFIKASI: similarity < min_similarity_uncertain (40.0%)

    Args:
        similarity_pct (float): Nilai kemiripan sampel terbaik (%)
        vote_share_pct (float, optional): Persentase voting KNN (%)
        top_matches (list, optional): Daftar top matches

    Returns:
        str: 'TERIDENTIFIKASI' | 'TIDAK PASTI' | 'TIDAK TERIDENTIFIKASI'
    """
    cfg = VERIFICATION_CONFIDENCE_THRESHOLDS
    min_accept = cfg.get("min_similarity_accept", 50.0)
    min_uncertain = cfg.get("min_similarity_uncertain", 40.0)
    min_vote_share = cfg.get("min_vote_share_accept", 35.0)

    if similarity_pct < min_uncertain:
        return "TIDAK TERIDENTIFIKASI"
    
    if similarity_pct >= min_accept:
        if vote_share_pct is None or vote_share_pct >= min_vote_share:
            return "TERIDENTIFIKASI"
        else:
            # Kemiripan cukup tinggi tapi konsensus voting KNN terpecah antar kandidat
            return "TIDAK PASTI"
    
    # 40.0 <= similarity_pct < 50.0
    return "TIDAK PASTI"


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
    4. KNN kneighbors(): hitung Euclidean Distance dan indeks ke K tetangga terdekat
    5. Hitung rincian K tetangga terdekat (diagnostik transparansi KNN)
    6. Hitung jarak sampel minimum (d_min) dan Sample Similarity (%) untuk tiap kelas
    7. Urutkan Top Kandidat:
       - Primer: KNN Vote Share (%) descending (pemenang voting selalu #1)
       - Sekunder: Euclidean Distance minimum ascending
    8. Parameter utama (distance, similarity, status) diambil dari sampel terbaik milik predicted_name.

    Pemisahan Metrik:
    - KNN Weighted Vote Share (%): Menunjukkan persentase perolehan bobot voting KNN (K=5).
    - Sample Similarity (%): Menunjukkan kemiripan geometris sampel terdekat berbasis normalized HOG distance.
    - Verification Status: Keputusan verifikasi (TERIDENTIFIKASI / TIDAK PASTI / TIDAK TERIDENTIFIKASI).
    - Similarity Status: Level kemiripan visual (SANGAT MIRIP / MIRIP / KURANG MIRIP / TIDAK MIRIP).
    """
    # --- Step 1: Siapkan feature vector query & validasi defensif ---
    query = np.array(query_feature).reshape(1, -1)

    # Validasi dimensi feature vector
    if query.shape[1] != EXPECTED_FEATURE_DIM:
        raise ValueError(
            f"Dimensi feature vector query tidak valid: diharapkan {EXPECTED_FEATURE_DIM}, "
            f"tetapi diterima {query.shape[1]}. Pastikan gambar dipreproses pada resolusi 256x256."
        )

    if X_train is not None and X_train.shape[1] != EXPECTED_FEATURE_DIM:
        raise ValueError(
            f"Dimensi feature vector X_train tidak valid: diharapkan {EXPECTED_FEATURE_DIM}, "
            f"tetapi ditemukan {X_train.shape[1]}."
        )

    if hasattr(knn_model, "n_neighbors") and knn_model.n_neighbors != EXPECTED_K:
        raise ValueError(
            f"Konfigurasi KNN K tidak valid: diharapkan K={EXPECTED_K}, "
            f"tetapi model memiliki K={knn_model.n_neighbors}."
        )

    if hasattr(knn_model, "metric") and knn_model.metric != EXPECTED_METRIC:
        raise ValueError(
            f"Metrik KNN tidak valid: diharapkan '{EXPECTED_METRIC}', "
            f"tetapi model menggunakan '{knn_model.metric}'."
        )

    # --- Step 2: Prediksi kelas dengan KNN (voting mayoritas berbobot jarak) ---
    predicted_label = knn_model.predict(query)[0]
    predicted_name  = label_encoder.inverse_transform([predicted_label])[0]

    # --- Step 3: Probabilitas/bobot voting KNN untuk seluruh kelas ---
    probs = knn_model.predict_proba(query)[0]
    prob_dict = {cls_idx: float(probs[i]) for i, cls_idx in enumerate(knn_model.classes_)}

    # --- Step 4: Ambil K-nearest neighbors dan jarak Euclideannya ---
    k_distances_arr, k_indices_arr = knn_model.kneighbors(query)
    k_dists = [float(d) for d in k_distances_arr[0]]
    k_idxs  = [int(idx) for idx in k_indices_arr[0]]
    
    # Hitung bobot per tetangga w_i = 1 / (d_i + eps)
    eps = 1e-7
    raw_weights = [1.0 / max(d, eps) for d in k_dists]
    total_weight = sum(raw_weights) if sum(raw_weights) > 0 else 1.0

    k_neighbors_detail = []
    neighbor_class_counts = {}
    for rank_idx, (d, s_idx, w) in enumerate(zip(k_dists, k_idxs, raw_weights), start=1):
        lbl = y_train_encoded[s_idx]
        name = label_encoder.inverse_transform([lbl])[0]
        neighbor_class_counts[name] = neighbor_class_counts.get(name, 0) + 1
        
        contrib_pct = round((w / total_weight) * 100.0, 2)
        sim_pct = euclidean_to_similarity(d)
        k_neighbors_detail.append({
            "rank": rank_idx,
            "sample_index": s_idx,
            "name": name,
            "distance": round(d, 4),
            "similarity_percent": sim_pct,
            "weight": round(w, 5),
            "vote_contrib_pct": contrib_pct,
        })

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
        n_k_count      = neighbor_class_counts.get(class_name, 0)

        class_results.append({
            "name":           class_name,
            "distance":       round(min_dist_class, 4),
            "percent":        round(sim_class, 2),
            "vote_weight":    round(raw_vote_prob, 4),
            "vote_percent":   vote_percent,
            "neighbor_count": n_k_count,
            "label_id":       lbl,
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
    
    sim_status  = get_similarity_status(predicted_similarity)
    verif_status = get_verification_status(predicted_similarity, predicted_vote_weight, top_matches)

    # Format top_matches untuk output JSON
    clean_top_matches = [
        {
            "name":           m["name"],
            "distance":       m["distance"],
            "percent":        m["percent"],          # Sample Similarity %
            "vote_percent":   m["vote_percent"],     # KNN Weighted Vote Share %
            "vote_weight":    m["vote_weight"],      # Normalized vote share (0.0 - 1.0)
            "neighbor_count": m["neighbor_count"],   # Jumlah tetangga di K=5
        }
        for m in top_matches
    ]

    return {
        "predicted_name":        predicted_name,
        "predicted_vote_weight": predicted_vote_weight,
        "euclidean_distance":    round(predicted_distance, 4),
        "similarity_percent":    round(predicted_similarity, 2),
        "similarity_status":     sim_status,
        "verification_status":   verif_status,
        "k_neighbors":           int(knn_model.n_neighbors),
        "top_matches":           clean_top_matches,
        "k_distances":           [round(d, 4) for d in k_dists],
        "k_neighbors_detail":    k_neighbors_detail,
    }


# ==============================================================================
# LOAD MODEL DARI DISK (UNIFIED & DATABASE-AWARE)
# ==============================================================================

def _load_latest_model_files() -> tuple:
    """
    Memuat model KNN aktif dan file pendukungnya.
    Menggunakan get_latest_model_paths() sebagai single source of truth.

    Returns:
        tuple: (knn_model, label_encoder, X_train, y_train, timestamp)
               Raises FileNotFoundError jika belum ada model.
    """
    paths = get_latest_model_paths()
    if paths and os.path.exists(paths["model_path"]):
        model_path = paths["model_path"]
        encoder_path = paths["le_path"]
        Xtrain_path  = paths["Xtrain_path"]
        ytrain_path  = paths["ytrain_path"]
        timestamp    = paths["timestamp"]
    else:
        # Fallback ke pencarian direktori jika model_meta belum terisi
        model_files = glob.glob(os.path.join(MODEL_SAVED_DIR, "knn_model_*.joblib"))
        if not model_files:
            raise FileNotFoundError(
                "Belum ada model terlatih. Jalankan training terlebih dahulu."
            )
        model_path = max(model_files, key=os.path.getmtime)
        timestamp  = os.path.basename(model_path).replace("knn_model_", "").replace(".joblib", "")
        encoder_path = os.path.join(MODEL_SAVED_DIR, f"label_encoder_{timestamp}.joblib")
        Xtrain_path  = os.path.join(MODEL_SAVED_DIR, f"train_features_{timestamp}.joblib")
        ytrain_path  = os.path.join(MODEL_SAVED_DIR, f"train_labels_{timestamp}.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} tidak ditemukan.")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Label encoder {encoder_path} tidak ditemukan.")
    if not os.path.exists(Xtrain_path) or not os.path.exists(ytrain_path):
        raise FileNotFoundError(f"Data training untuk model {timestamp} tidak ditemukan.")

    knn_model     = joblib.load(model_path)
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
    ground_truth_name: str = None,
    ground_truth_nim: str = None,
    claimed_writer: str = None,
) -> dict:
    """
    BAB IV - Fungsi Utama Verifikasi Gambar:

    Dipanggil dari api/verify_routes.py setelah preprocessing & HOG extraction.

    Args:
        feature_vector: Feature vector HOG dari gambar query.
        model_version: Identifier versi model (optional, untuk logging DB).
        query_filename: Nama file gambar asli.
        query_path: Path lengkap file gambar.
        ground_truth_name: Nama label sebenarnya (ground truth) jika diketahui.
        ground_truth_nim: NIM label sebenarnya jika diketahui.
        claimed_writer: Nama penulis yang diklaim untuk 1-to-1 verification (Experiment K1).

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
        # Jalankan klasifikasi KNN + Euclidean Distance (1-to-N identification)
        result = classify_handwriting(
            feature_vector, knn_model, label_encoder, X_train, y_train
        )
    except Exception as e:
        logger.error(f"Error classifying: {e}")
        return {"success": False, "message": f"Gagal klasifikasi: {str(e)}"}

    # Claimed-Identity Verification (1-to-1 verification mode)
    verification_data = None
    if claimed_writer and str(claimed_writer).strip():
        try:
            verification_data = verify_claim(
                query_feature=feature_vector,
                claimed_writer=claimed_writer,
                X_train=X_train,
                y_train_encoded=y_train,
                label_encoder=label_encoder,
                model_version=model_version or timestamp,
            )
            result["verification"] = verification_data
        except Exception as e:
            logger.error(f"Error in verify_claim: {e}")
            return {"success": False, "message": f"Gagal verifikasi klaim identitas: {str(e)}"}

    # Hitung total waktu analisis
    analysis_time = round(time.time() - t_start, 4)
    result["analysis_time_seconds"] = analysis_time
    result["feature_vector_length"] = len(feature_vector)

    # Hitung is_correct jika ground_truth_name tersedia
    is_correct = None
    if ground_truth_name is not None and str(ground_truth_name).strip() != "":
        pred_clean = str(result["predicted_name"]).strip().lower()
        gt_clean = str(ground_truth_name).strip().lower()
        is_correct = 1 if pred_clean == gt_clean else 0

    result["ground_truth_name"] = ground_truth_name
    result["ground_truth_nim"]  = ground_truth_nim
    result["is_correct"]        = is_correct

    # Konversi top_matches ke JSON string untuk database
    top_matches_json = json.dumps(result["top_matches"], ensure_ascii=False)

    # Ekstrak data verifikasi klaim untuk disimpan ke DB
    db_claimed_writer = verification_data["claimed_writer"] if verification_data else None
    db_verif_score    = verification_data["score"] if verification_data else None
    db_verif_thresh   = verification_data["threshold"] if verification_data else None
    db_verif_decision = verification_data["decision"] if verification_data else None
    db_verif_status   = verification_data["status"] if verification_data else None
    db_verif_method   = verification_data["method"] if verification_data else None
    db_top_claimed_dists = json.dumps(verification_data["neighbor_distances"]) if verification_data else None

    # Simpan hasil ke database verifications
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO verifications (
                query_filename, query_path, predicted_name,
                similarity_percent, euclidean_distance, verification_status, similarity_status,
                top_matches_json, model_version, feature_vector_length, knn_k,
                analysis_time, ground_truth_name, ground_truth_nim, is_correct,
                claimed_writer, verification_score, verification_threshold,
                verification_decision, verification_status_verif, verification_method,
                top_claimed_distances_json,
                verification_timestamp
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now','localtime'))
        """, (
            query_filename,
            query_path,
            result["predicted_name"],
            result["similarity_percent"],
            result["euclidean_distance"],
            result["verification_status"],   # TERIDENTIFIKASI / TIDAK PASTI / TIDAK TERIDENTIFIKASI
            result["similarity_status"],     # SANGAT MIRIP / MIRIP / KURANG MIRIP / TIDAK MIRIP
            top_matches_json,
            model_version or timestamp,
            result["feature_vector_length"],
            result["k_neighbors"],
            analysis_time,
            ground_truth_name,
            ground_truth_nim,
            is_correct,
            db_claimed_writer,
            db_verif_score,
            db_verif_thresh,
            db_verif_decision,
            db_verif_status,
            db_verif_method,
            db_top_claimed_dists,
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

