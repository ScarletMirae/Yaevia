"""
app.py — Entry Point Flask Server
====================================
Server utama sistem verifikasi keaslian tulisan tangan.

Menjalankan:
    python app.py

Atau dalam mode produksi:
    waitress-serve --host=0.0.0.0 --port=5000 app:app

Akses frontend:
    Buka D:/handwriting-verification/frontend/index.html di browser.
    Frontend menggunakan fetch() API untuk berkomunikasi dengan server ini.
"""

import os
import sys

from flask import Flask, jsonify, send_from_directory, send_file
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, SECRET_KEY, MAX_CONTENT_LENGTH
from database import init_db
from api.dataset_routes  import dataset_bp
from api.train_routes    import train_bp
from api.verify_routes   import verify_bp
from api.evaluate_routes import evaluate_bp



# Path ke folder frontend (satu level di atas backend)
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

# ==============================================================================
# INISIALISASI FLASK APP
# ==============================================================================
app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path=""
)
app.config["SECRET_KEY"]        = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# CORS: izinkan akses dari frontend lokal (file:// atau localhost)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ==============================================================================
# REGISTRASI BLUEPRINT (API ROUTES)
# ==============================================================================
app.register_blueprint(dataset_bp)   # /api/dataset/*
app.register_blueprint(train_bp)     # /api/train/* + /api/model/*
app.register_blueprint(verify_bp)    # /api/verify/*
app.register_blueprint(evaluate_bp)  # /api/evaluate/*



# ==============================================================================
# SERVE FRONTEND (HTML Pages + Static Assets)
# Akses lewat: http://localhost:5000/
# ==============================================================================
@app.route("/")
def serve_index():
    """Redirect root ke index.html frontend."""
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def serve_frontend(filename):
    """
    Melayani semua file frontend (HTML, CSS, JS, gambar).
    - *.html  → halaman frontend
    - css/*   → stylesheet
    - js/*    → script
    - Fallback ke index.html untuk SPA routing
    """
    import os
    # Cek apakah file ada
    filepath = os.path.join(FRONTEND_DIR, filename)
    if os.path.isfile(filepath):
        return send_from_directory(FRONTEND_DIR, filename)
    # Fallback: halaman HTML langsung (tanpa ekstensi)
    html_file = filename + ".html"
    if os.path.isfile(os.path.join(FRONTEND_DIR, html_file)):
        return send_from_directory(FRONTEND_DIR, html_file)
    # 404 fallback
    return send_from_directory(FRONTEND_DIR, "index.html")


# ==============================================================================
# ROUTE UTILITAS
# ==============================================================================
@app.route("/api/health", methods=["GET"])

def health_check():
    """
    Health check endpoint untuk memeriksa status server.
    Digunakan oleh frontend untuk memastikan server berjalan.
    """
    return jsonify({
        "status":  "ok",
        "message": "Sistem Verifikasi Tulisan Tangan aktif",
        "version": "2.0.0",
    }), 200



@app.route("/api/config", methods=["GET"])
def get_config():
    """
    Mengembalikan konfigurasi aktif sistem (untuk tampilan di UI).
    Berguna untuk dokumentasi dan debugging skripsi.
    """
    from config import (
        KNN_N_NEIGHBORS, KNN_METRIC, KNN_WEIGHTS,
        HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK,
        IMAGE_SIZE, TEST_SIZE, MATA_KULIAH_OPTIONS,
    )
    return jsonify({
        "knn": {
            "n_neighbors": KNN_N_NEIGHBORS,
            "metric":      KNN_METRIC,
            "weights":     KNN_WEIGHTS,
        },
        "hog": {
            "orientations":    HOG_ORIENTATIONS,
            "pixels_per_cell": HOG_PIXELS_PER_CELL,
            "cells_per_block": HOG_CELLS_PER_BLOCK,
        },
        "preprocessing": {
            "image_size": IMAGE_SIZE,
        },
        "training": {
            "test_size": TEST_SIZE,
        },
        "mata_kuliah_options": MATA_KULIAH_OPTIONS,
    }), 200


# ==============================================================================
# ERROR HANDLERS
# ==============================================================================
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"success": False, "message": "Bad Request", "error": str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "message": "Endpoint tidak ditemukan"}), 404

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"success": False, "message": "Ukuran file melebihi batas (16 MB)"}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"success": False, "message": "Internal server error", "error": str(e)}), 500


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Sistem Verifikasi Keaslian Tulisan Tangan")
    print("  HOG + KNN | Flask Server")
    print("=" * 60)

    # Inisialisasi database SQLite
    init_db()

    print(f"\n[SERVER] Berjalan di:   http://localhost:{FLASK_PORT}")
    print(f"[SERVER] Buka browser:  http://localhost:{FLASK_PORT}/")
    print(f"[SERVER] Health check:  http://localhost:{FLASK_PORT}/api/health")
    print("=" * 60)


    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
        use_reloader=False,   # Matikan reloader untuk menghindari double init_db
    )
