"""
api/dataset_routes.py — Endpoint Manajemen Dataset
====================================================
Menyediakan REST API endpoint untuk:
    - POST /api/dataset/upload   — Upload citra + label mahasiswa
    - GET  /api/dataset/list     — Daftar semua dataset
    - GET  /api/dataset/stats    — Statistik dataset (per mahasiswa, per MK)
    - DELETE /api/dataset/<id>   — Hapus satu entri dataset
    - DELETE /api/dataset/all    — Hapus semua dataset (reset)
"""

import os
import uuid
import sys
from datetime import datetime
from flask import Blueprint, request, jsonify
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_RAW_DIR, DATASET_PROCESSED_DIR, ALLOWED_EXTENSIONS
from database import get_connection
from preprocessing.image_processor import preprocess_image

dataset_bp = Blueprint("dataset", __name__, url_prefix="/api/dataset")


def allowed_file(filename: str) -> bool:
    """Memeriksa apakah ekstensi file termasuk yang diizinkan."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ==============================================================================
# POST /api/dataset/upload
# ==============================================================================
@dataset_bp.route("/upload", methods=["POST"])
def upload_image():
    """
    Endpoint untuk upload satu citra tulisan tangan ke dataset.

    Form data yang diperlukan:
        - file (multipart): File gambar (JPG/PNG/BMP/TIFF)
        - student_name (str): Nama lengkap mahasiswa
        - student_id (str, opsional): NIM mahasiswa
        - mata_kuliah (str): Nama mata kuliah
        - notes (str, opsional): Catatan tambahan

    Returns:
        JSON: { success, message, data: { id, filename, student_name, ... } }
    """
    # --- Validasi form data ---
    if "file" not in request.files:
        return jsonify({"success": False, "message": "Tidak ada file dalam request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "Nama file kosong"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "message": f"Ekstensi file tidak didukung. Gunakan: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    student_name = request.form.get("student_name", "").strip()
    if not student_name:
        return jsonify({"success": False, "message": "Nama mahasiswa wajib diisi"}), 400

    student_id  = request.form.get("student_id", "").strip()
    mata_kuliah = request.form.get("mata_kuliah", "").strip()
    notes       = request.form.get("notes", "").strip()

    if not mata_kuliah:
        return jsonify({"success": False, "message": "Mata kuliah wajib dipilih"}), 400

    # --- Simpan file ---
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    raw_path = os.path.join(DATASET_RAW_DIR, unique_filename)

    os.makedirs(DATASET_RAW_DIR, exist_ok=True)
    file.save(raw_path)

    # --- Preprocessing & simpan citra terproses ---
    processed_path = None
    is_processed   = 0
    try:
        proc_filename  = f"proc_{unique_filename.rsplit('.', 1)[0]}.png"
        processed_path = os.path.join(DATASET_PROCESSED_DIR, proc_filename)
        os.makedirs(DATASET_PROCESSED_DIR, exist_ok=True)
        preprocess_image(raw_path, save_path=processed_path)
        is_processed = 1
    except Exception as e:
        print(f"[DATASET] Preprocessing gagal untuk {unique_filename}: {e}")
        processed_path = None
        is_processed   = 0

    # --- Simpan metadata ke database ---
    timestamp = datetime.now().isoformat()
    conn = get_connection()
    cursor = conn.execute("""
        INSERT INTO dataset (
            student_name, student_id, mata_kuliah,
            original_filename, saved_filename, file_path,
            processed_path, is_processed, upload_timestamp, notes
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        student_name, student_id, mata_kuliah,
        file.filename, unique_filename, raw_path,
        processed_path, is_processed, timestamp, notes,
    ))
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return jsonify({
        "success": True,
        "message": f"Citra berhasil diupload untuk: {student_name}",
        "data": {
            "id":             new_id,
            "filename":       unique_filename,
            "original_name":  file.filename,
            "student_name":   student_name,
            "student_id":     student_id,
            "mata_kuliah":    mata_kuliah,
            "is_processed":   bool(is_processed),
            "upload_timestamp": timestamp,
        }
    }), 201


# ==============================================================================
# GET /api/dataset/list
# ==============================================================================
@dataset_bp.route("/list", methods=["GET"])
def list_dataset():
    """
    Mengambil daftar semua citra dalam dataset.

    Query params (opsional):
        - mata_kuliah (str): Filter berdasarkan mata kuliah
        - student_name (str): Filter berdasarkan nama mahasiswa

    Returns:
        JSON: { success, total, data: [ {...} ] }
    """
    mata_kuliah_filter = request.args.get("mata_kuliah", "").strip()
    student_filter     = request.args.get("student_name", "").strip()

    query = "SELECT * FROM dataset"
    params = []
    conditions = []

    if mata_kuliah_filter:
        conditions.append("mata_kuliah = ?")
        params.append(mata_kuliah_filter)
    if student_filter:
        conditions.append("student_name LIKE ?")
        params.append(f"%{student_filter}%")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY upload_timestamp DESC"

    conn = get_connection()
    rows = conn.execute(query, params).fetchall()
    conn.close()

    data = [dict(row) for row in rows]
    return jsonify({"success": True, "total": len(data), "data": data}), 200


# ==============================================================================
# GET /api/dataset/stats
# ==============================================================================
@dataset_bp.route("/stats", methods=["GET"])
def dataset_stats():
    """
    Mengambil statistik dataset: jumlah per mahasiswa, per mata kuliah.

    Returns:
        JSON: { total, per_student, per_mata_kuliah, classes_count }
    """
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) as cnt FROM dataset").fetchone()["cnt"]
    per_student = conn.execute(
        "SELECT student_name, student_id, COUNT(*) as count FROM dataset GROUP BY student_name"
    ).fetchall()
    per_mk = conn.execute(
        "SELECT mata_kuliah, COUNT(*) as count FROM dataset GROUP BY mata_kuliah"
    ).fetchall()
    conn.close()

    return jsonify({
        "success": True,
        "total": total,
        "classes_count": len(per_student),
        "per_student":    [dict(r) for r in per_student],
        "per_mata_kuliah": [dict(r) for r in per_mk],
    }), 200


# ==============================================================================
# DELETE /api/dataset/<id>
# ==============================================================================
@dataset_bp.route("/<int:dataset_id>", methods=["DELETE"])
def delete_dataset_item(dataset_id: int):
    """
    Menghapus satu entri dataset berdasarkan ID.
    File fisik (raw + processed) juga akan dihapus.
    """
    conn = get_connection()
    row = conn.execute("SELECT * FROM dataset WHERE id=?", (dataset_id,)).fetchone()

    if not row:
        conn.close()
        return jsonify({"success": False, "message": f"Dataset ID {dataset_id} tidak ditemukan"}), 404

    # Hapus file fisik
    for path_key in ("file_path", "processed_path"):
        path = row[path_key]
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print(f"[DATASET] Gagal hapus file {path}: {e}")

    conn.execute("DELETE FROM dataset WHERE id=?", (dataset_id,))
    conn.commit()
    conn.close()

    return jsonify({"success": True, "message": f"Dataset ID {dataset_id} berhasil dihapus"}), 200


# ==============================================================================
# DELETE /api/dataset/all
# ==============================================================================
@dataset_bp.route("/all", methods=["DELETE"])
def delete_all_dataset():
    """Menghapus seluruh dataset dan semua file citra terkait."""
    conn = get_connection()
    rows = conn.execute("SELECT file_path, processed_path FROM dataset").fetchall()

    deleted_files = 0
    for row in rows:
        for path_key in ("file_path", "processed_path"):
            path = row[path_key]
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    deleted_files += 1
                except Exception:
                    pass

    conn.execute("DELETE FROM dataset")
    conn.commit()
    conn.close()

    return jsonify({
        "success": True,
        "message": f"Seluruh dataset dihapus. {deleted_files} file dihapus dari disk."
    }), 200


# ==============================================================================
# GET /api/dataset/summary
# Dataset Summary: per-mahasiswa + ringkasan untuk halaman Upload
# ==============================================================================
@dataset_bp.route("/summary", methods=["GET"])
def get_dataset_summary():
    """
    GET /api/dataset/summary
    Mengembalikan ringkasan dataset untuk card Dataset Summary:
        - Total mahasiswa, total dataset, mata kuliah
        - Estimasi total training & testing data
        - Rata-rata sampel per mahasiswa
        - Tabel: nama mahasiswa -> jumlah sampel
    """
    try:
        conn = get_connection()

        # Jumlah per mahasiswa
        per_student = conn.execute("""
            SELECT student_name, COUNT(*) as count
            FROM dataset
            GROUP BY student_name
            ORDER BY student_name
        """).fetchall()

        # Jumlah per mata kuliah
        per_mk = conn.execute("""
            SELECT mata_kuliah, COUNT(*) as count
            FROM dataset
            GROUP BY mata_kuliah
        """).fetchall()

        # Total dataset
        total_row = conn.execute("SELECT COUNT(*) as cnt FROM dataset").fetchone()
        conn.close()

        total       = total_row["cnt"] if total_row else 0
        n_students  = len(per_student)

        student_list = [
            {"name": r["student_name"], "count": r["count"]}
            for r in per_student
        ]
        mk_list = [
            {"mata_kuliah": r["mata_kuliah"], "count": r["count"]}
            for r in per_mk
        ]

        # Estimasi training & testing dengan split 80:20
        n_test_est  = max(0, round(total * 0.20))
        n_train_est = total - n_test_est
        avg_per_student = round(total / n_students, 1) if n_students > 0 else 0

        return jsonify({
            "success":          True,
            "total":            total,
            "n_students":       n_students,
            "n_train_estimate": n_train_est,
            "n_test_estimate":  n_test_est,
            "avg_per_student":  avg_per_student,
            "students":         student_list,
            "mata_kuliah":      mk_list,
        }), 200

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

