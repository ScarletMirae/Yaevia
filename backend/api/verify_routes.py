"""
api/verify_routes.py — Endpoint Verifikasi Tulisan Tangan
===========================================================
Menyediakan endpoint untuk:
  POST /api/verify             — verifikasi gambar (Euclidean Distance based)
  GET  /api/verify/history     — riwayat verifikasi
  GET  /api/verify/<id>        — detail satu record verifikasi
  DELETE /api/verify/<id>      — hapus satu record
"""

import os
import sys
import json
import uuid
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_RAW_DIR, ALLOWED_EXTENSIONS
from preprocessing.image_processor import preprocess_image
from features.hog_extractor import extract_hog_features
from model.classifier import verify_image
from database import get_connection

verify_bp = Blueprint("verify", __name__)
logger    = logging.getLogger(__name__)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@verify_bp.route("/api/verify", methods=["POST"])
def api_verify():
    """
    POST /api/verify
    Menerima gambar tulisan tangan dan mengembalikan hasil verifikasi.

    Pipeline yang dijalankan:
      1. Terima file gambar
      2. Simpan sementara ke disk
      3. Preprocessing (grayscale, thresholding, resize 128x128)
      4. Ekstraksi fitur HOG (feature vector)
      5. Klasifikasi KNN + Euclidean Distance (via classifier.py)
      6. Return hasil lengkap

    Response JSON:
        success             (bool)
        predicted_name      (str)   : Nama mahasiswa yang diprediksi
        euclidean_distance  (float) : Jarak Euclidean ke nearest neighbor
        similarity_percent  (float) : Similarity score 0-100% (1/(1+d)*100)
        similarity_status   (str)   : SANGAT MIRIP | MIRIP | KURANG MIRIP | TIDAK MIRIP
        verification_status (str)   : Sama dengan similarity_status
        top_matches         (list)  : Top-5 kandidat dengan distance & similarity
        k_neighbors         (int)   : Nilai K yang digunakan
        feature_vector_length (int) : Panjang feature vector HOG
        analysis_time_seconds (float): Waktu analisis dalam detik
        verification_id     (int)   : ID record di database
        model_version       (str)   : Timestamp model yang digunakan
    """
    if "file" not in request.files:
        return jsonify({"success": False, "message": "Tidak ada file yang diupload"}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"success": False, "message": "File tidak valid"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "message": f"Format file tidak didukung. Gunakan: {', '.join(ALLOWED_EXTENSIONS)}",
        }), 400

    # Simpan file sementara
    ext           = file.filename.rsplit(".", 1)[1].lower()
    unique_name   = f"verify_{uuid.uuid4().hex}.{ext}"
    query_dir     = os.path.join(DATASET_RAW_DIR, "queries")
    os.makedirs(query_dir, exist_ok=True)
    query_path    = os.path.join(query_dir, unique_name)
    file.save(query_path)

    try:
        # --- Step 1: Preprocessing citra ---
        # preprocess_image() menerima path file dan melakukan:
        # grayscale -> thresholding Otsu -> noise removal -> resize 128x128
        processed_img = preprocess_image(query_path)

        # --- Step 2: Ekstraksi Fitur HOG ---
        feature_vector = extract_hog_features(processed_img)

        # --- Step 3: Klasifikasi KNN + Euclidean Distance ---
        result = verify_image(
            feature_vector = feature_vector,
            query_filename = file.filename,
            query_path     = query_path,
        )

        if not result.get("success"):
            return jsonify(result), 400

        # Tambahkan field compatibility untuk frontend lama
        result["verification_status"] = result.get("similarity_status", "TIDAK MIRIP")
        result["similarity_score"]     = result.get("similarity_percent", 0) / 100.0

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Verifikasi error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Gagal memproses gambar: {str(e)}",
        }), 500


@verify_bp.route("/api/verify/history", methods=["GET"])
def api_verify_history():
    """
    GET /api/verify/history?limit=20&offset=0
    Mengambil riwayat verifikasi dengan pagination.
    """
    try:
        limit  = int(request.args.get("limit",  20))
        offset = int(request.args.get("offset", 0))

        conn  = get_connection()
        rows  = conn.execute("""
            SELECT id, query_filename, predicted_name,
                   similarity_percent, euclidean_distance, verification_status, similarity_status,
                   top_matches_json, model_version, feature_vector_length, knn_k,
                   analysis_time, verification_timestamp
            FROM verifications
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """, (limit, offset)).fetchall()

        total = conn.execute("SELECT COUNT(*) FROM verifications").fetchone()[0]
        conn.close()

        data = []
        for r in rows:
            rec = dict(r)
            # Parse top_matches_json jika ada
            try:
                rec["top_matches"] = json.loads(rec.get("top_matches_json") or "[]")
            except Exception:
                rec["top_matches"] = []
            data.append(rec)

        return jsonify({"success": True, "data": data, "total": total,
                        "limit": limit, "offset": offset}), 200

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@verify_bp.route("/api/verify/<int:verify_id>", methods=["GET"])
def api_verify_detail(verify_id):
    """GET /api/verify/<id> — detail satu record verifikasi."""
    try:
        conn = get_connection()
        row  = conn.execute(
            "SELECT * FROM verifications WHERE id=?", (verify_id,)
        ).fetchone()
        conn.close()

        if not row:
            return jsonify({"success": False, "message": "Record tidak ditemukan"}), 404

        rec = dict(row)
        try:
            rec["top_matches"] = json.loads(rec.get("top_matches_json") or "[]")
        except Exception:
            rec["top_matches"] = []

        return jsonify({"success": True, "data": rec}), 200

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@verify_bp.route("/api/verify/<int:verify_id>", methods=["DELETE"])
def api_verify_delete(verify_id):
    """DELETE /api/verify/<id> — hapus satu record verifikasi."""
    try:
        conn = get_connection()
        row  = conn.execute(
            "SELECT id FROM verifications WHERE id=?", (verify_id,)
        ).fetchone()

        if not row:
            conn.close()
            return jsonify({"success": False, "message": "Record tidak ditemukan"}), 404

        conn.execute("DELETE FROM verifications WHERE id=?", (verify_id,))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": "Record berhasil dihapus"}), 200

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
