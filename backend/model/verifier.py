"""
model/verifier.py — Claimed-Identity Verification Service (Experiment K1 Frozen Protocol)
==========================================================================================
Modul ini mengimplementasikan verifikasi 1-to-1 berbasis klaim identitas (Claimed Identity)
menggunakan protokol terstandarisasi dan beku dari Experiment K1:
  - Skor Verifikasi: Mean Top-5 Claimed-Writer Euclidean Distance
  - Frozen Global Threshold (EER Operating Point): THETA = 25.1291
  - Keputusan:
      Score <= 25.1291 -> ACCEPT (VALID)
      Score >  25.1291 -> REJECT (TIDAK VALID)

PENTING:
  - Keputusan verifikasi TIDAK ditentukan oleh predicted_name == claimed_writer.
  - Keputusan verifikasi TIDAK bergantung pada voting kompetitif 20-kelas.
  - Verifikasi mengevaluasi kedekatan jarak absolut terhadap ruang fitur penulis yang diklaim.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from config import (
    VERIFICATION_METHOD,
    VERIFICATION_K,
    VERIFICATION_THRESHOLD,
    VERIFICATION_THRESHOLD_SOURCE,
    VERIFICATION_DECISION_ACCEPT,
    VERIFICATION_DECISION_REJECT,
    VERIFICATION_STATUS_VALID,
    VERIFICATION_STATUS_INVALID,
)

logger = logging.getLogger(__name__)


def verify_claim(
    query_feature: np.ndarray,
    claimed_writer: str,
    X_train: np.ndarray,
    y_train_encoded: np.ndarray,
    label_encoder: Any,
    k: int = VERIFICATION_K,
    threshold: float = VERIFICATION_THRESHOLD,
    model_version: str = "20260926_190940",
) -> Dict[str, Any]:
    """
    Menjalankan verifikasi klaim identitas (Claimed-Identity Verification).

    Args:
        query_feature (np.ndarray): Vektor fitur HOG 34,596 dimensi dari citra query.
        claimed_writer (str): Nama penulis yang diklaim sebagai pemilik tulisan.
        X_train (np.ndarray): Matriks fitur referensi (N, D).
        y_train_encoded (np.ndarray): Array label integer untuk sampel referensi.
        label_encoder: Objek LabelEncoder yang memetakan nama penulis ke integer.
        k (int): Jumlah tetangga terdekat yang dirata-ratakan (default: 5).
        threshold (float): Batas ambang jarak global (default: 25.1291).
        model_version (str): Versi timestamp model aktif.

    Returns:
        dict: Struktur hasil verifikasi lengkap.
    """
    if not claimed_writer or not str(claimed_writer).strip():
        raise ValueError("Claimed writer tidak boleh kosong.")

    claimed_writer_clean = str(claimed_writer).strip()

    # Validasi apakah claimed_writer terdaftar pada label encoder
    known_classes = list(label_encoder.classes_)
    if claimed_writer_clean not in known_classes:
        raise ValueError(
            f"Penulis yang diklaim '{claimed_writer_clean}' tidak terdaftar dalam dataset/model. "
            f"Pilihan yang valid: {', '.join(known_classes)}"
        )

    # Ambil label integer penulis yang diklaim
    claimed_label = label_encoder.transform([claimed_writer_clean])[0]

    # Filter referensi HOG HANYA milik claimed_writer
    class_mask = (y_train_encoded == claimed_label)
    claimed_ref_features = X_train[class_mask]

    n_available_refs = claimed_ref_features.shape[0]
    if n_available_refs < k:
        raise ValueError(
            f"Jumlah sampel referensi untuk '{claimed_writer_clean}' ({n_available_refs}) "
            f"kurang dari K={k} yang disyaratkan oleh protokol verifikasi."
        )

    # Reshape query feature jika diperlukan
    q_feat = np.array(query_feature, dtype=np.float32).reshape(1, -1)

    # Hitung Euclidean Distance ke seluruh sampel referensi milik claimed_writer
    diffs = claimed_ref_features - q_feat
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))

    # Urutkan jarak ascending
    sorted_dists = np.sort(dists)

    # Ambil K jarak terkecil
    top_k_dists = sorted_dists[:k]

    # Hitung Mean Top-5 Claimed-Writer Euclidean Distance
    verification_score = float(np.mean(top_k_dists))

    # Evaluasi keputusan terhadap frozen threshold
    # Gunakan presisi 4 desimal sesuai spesifikasi protocol K1
    score_rounded = round(verification_score, 4)
    threshold_rounded = round(threshold, 4)

    if score_rounded <= threshold_rounded:
        decision = VERIFICATION_DECISION_ACCEPT
        status   = VERIFICATION_STATUS_VALID
    else:
        decision = VERIFICATION_DECISION_REJECT
        status   = VERIFICATION_STATUS_INVALID

    return {
        "claimed_writer": claimed_writer_clean,
        "score": round(verification_score, 4),
        "raw_score": verification_score,
        "threshold": threshold,
        "threshold_source": VERIFICATION_THRESHOLD_SOURCE,
        "decision": decision,
        "status": status,
        "method": VERIFICATION_METHOD,
        "k_neighbors": k,
        "available_reference_count": int(n_available_refs),
        "neighbor_distances": [round(float(d), 4) for d in top_k_dists],
        "model_version": model_version,
    }
