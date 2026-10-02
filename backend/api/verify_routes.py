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
from config import DATASET_RAW_DIR, ALLOWED_EXTENSIONS, PUBLIC_EVALUATION_MODE
from preprocessing.image_processor import preprocess_image
from features.hog_extractor import extract_hog_features
from model.classifier import verify_image
from database import get_connection

verify_bp = Blueprint("verify", __name__)
logger    = logging.getLogger(__name__)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@verify_bp.route("/api/verify/validate-identity", methods=["GET", "POST"], strict_slashes=False)
def api_validate_identity():
    """
    POST /api/verify/validate-identity
    Memvalidasi apakah pasangan Nama Mahasiswa dan NIM yang diinput user
    terdaftar secara sah pada database dataset Yaevia (strict pair matching).
    """
    try:
        data = request.get_json(silent=True) or request.form or {}
        name = str(data.get("student_name") or data.get("name") or "").strip()
        nim  = str(data.get("student_id") or data.get("student_nim") or data.get("nim") or "").strip()

        if not name or not nim:
            return jsonify({"success": True, "valid": False}), 200

        conn = get_connection()
        count = conn.execute("""
            SELECT COUNT(*) FROM dataset
            WHERE LOWER(TRIM(student_name)) = ? AND LOWER(TRIM(student_id)) = ?
        """, (name.lower(), nim.lower())).fetchone()[0]
        conn.close()

        return jsonify({"success": True, "valid": bool(count > 0)}), 200

    except Exception as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"success": False, "valid": False, "message": str(e)}), 500



@verify_bp.route("/api/verify/claimed-writers", methods=["GET"], strict_slashes=False)
def api_get_claimed_writers():
    """
    GET /api/verify/claimed-writers
    Mengembalikan daftar penulis terdaftar (20 writers) untuk opsi dropdown Claimed Identity.
    """
    try:
        conn = get_connection()
        rows = conn.execute("""
            SELECT DISTINCT student_name, student_id
            FROM dataset
            ORDER BY student_name ASC
        """).fetchall()
        conn.close()

        writers = [
            {
                "name": r["student_name"],
                "nim": r["student_id"] or ""
            }
            for r in rows
        ]

        return jsonify({"success": True, "writers": writers, "total": len(writers)}), 200
    except Exception as e:
        logger.error(f"Error fetching claimed writers: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@verify_bp.route("/api/verify", methods=["POST"], strict_slashes=False)
def api_verify():
    """
    POST /api/verify
    Menerima gambar tulisan tangan dan mengembalikan hasil verifikasi.

    Pipeline yang dijalankan:
      1. Terima file gambar
      2. Simpan sementara ke disk
      3. Preprocessing (grayscale, thresholding, resize 256x256 letterbox)
      4. Ekstraksi fitur HOG (34,596 dim)
      5. Klasifikasi KNN + Claimed-Identity Verification (via classifier.py & verifier.py)
      6. Return hasil lengkap
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

    # Ambil claimed writer dan ground truth jika disediakan
    claimed_writer    = request.form.get("claimed_writer") or request.form.get("claimed_name")
    ground_truth_name = request.form.get("ground_truth_name") or request.form.get("ground_truth") or request.form.get("student_name")
    ground_truth_nim  = request.form.get("ground_truth_nim") or request.form.get("student_id")

    try:
        # --- Step 1: Preprocessing citra ---
        # preprocess_image() melakukan:
        # grayscale -> blur -> thresholding Otsu -> noise removal -> ROI -> letterbox 256x256
        processed_img = preprocess_image(query_path)

        # --- Step 2: Ekstraksi Fitur HOG ---
        feature_vector = extract_hog_features(processed_img)

        # --- Step 3: Klasifikasi KNN + Claimed-Identity Verification ---
        result = verify_image(
            feature_vector    = feature_vector,
            query_filename    = file.filename,
            query_path        = query_path,
            ground_truth_name = ground_truth_name,
            ground_truth_nim  = ground_truth_nim,
            claimed_writer    = claimed_writer,
        )

        if not result.get("success"):
            return jsonify(result), 400

        # Tambahkan field compatibility untuk frontend / script evaluasi
        result["similarity_score"] = result.get("similarity_percent", 0.0) / 100.0

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
                   analysis_time, ground_truth_name, ground_truth_nim, is_correct,
                   claimed_writer, verification_score, verification_threshold,
                   verification_decision, verification_status_verif, verification_method,
                   top_claimed_distances_json,
                   verification_timestamp
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
            try:
                rec["top_claimed_distances"] = json.loads(rec.get("top_claimed_distances_json") or "[]")
            except Exception:
                rec["top_claimed_distances"] = []
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
        try:
            rec["top_claimed_distances"] = json.loads(rec.get("top_claimed_distances_json") or "[]")
        except Exception:
            rec["top_claimed_distances"] = []

        return jsonify({"success": True, "data": rec}), 200

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@verify_bp.route("/api/verify/<int:verify_id>", methods=["DELETE"])
def api_verify_delete(verify_id):
    """DELETE /api/verify/<id> — hapus satu record verifikasi."""
    if PUBLIC_EVALUATION_MODE:
        return jsonify({
            "success": False,
            "error": "PUBLIC_EVALUATION_MODE_ACTIVE",
            "message": "Penghapusan riwayat verifikasi dinonaktifkan dalam Mode Evaluasi Media / Public Demo."
        }), 403

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
