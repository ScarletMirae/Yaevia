"""
api/train_routes.py — Endpoint Training Model KNN
==================================================
Menyediakan endpoint untuk:
  POST /api/train          — jalankan training pipeline
  GET  /api/train/status   — status dan metadata model aktif
  GET  /api/model/info     — metadata lengkap untuk Model Information card
"""

import os
import sys
from flask import Blueprint, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.trainer import train_model, get_latest_metadata, validate_dataset_min_samples
from database import get_active_model_meta

train_bp = Blueprint("train", __name__)


@train_bp.route("/api/train/validate", methods=["GET"])
def api_validate_dataset():
    """
    GET /api/train/validate
    Memvalidasi apakah dataset memenuhi syarat untuk training.
    Digunakan sebelum tombol 'Mulai Training' di-submit.
    """
    try:
        result = validate_dataset_min_samples()
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"valid": False, "message": str(e)}), 500


@train_bp.route("/api/train", methods=["POST"])
def api_train():
    """
    POST /api/train
    Menjalankan pipeline training KNN.

    Request JSON (opsional, semua ada default dari config.py):
        knn_k           (int)   : Nilai K
        metric          (str)   : 'euclidean' | 'manhattan' | 'minkowski'
        weights         (str)   : 'uniform' | 'distance'
        orientations    (int)   : HOG orientations
        pixels_per_cell (int)   : HOG pixels per cell (N -> [N,N])
        test_size       (float) : proporsi data uji (0.1-0.4)

    Response JSON:
        success         (bool)
        message         (str)
        train_accuracy  (float)
        test_accuracy   (float)
        precision       (float)
        recall          (float)
        f1_score        (float)
        n_train, n_test, n_classes, n_total (int)
        knn_k, metric   (mixed)
        feature_vector_size (int)
        training_time   (float)
        -- Jika gagal validasi:
        insufficient_students (list[str])
        counts (dict)
    """
    data = request.get_json(silent=True) or {}

    # Ambil parameter dari request (fallback ke config default)
    n_neighbors = int(data.get("knn_k", 5))
    metric      = str(data.get("metric", "euclidean"))
    weights     = str(data.get("weights", "distance"))
    orientations = int(data.get("orientations", 9))

    ppc_raw = data.get("pixels_per_cell", 8)
    if isinstance(ppc_raw, (list, tuple)):
        pixels_per_cell = tuple(int(v) for v in ppc_raw[:2])
    else:
        ppc = int(ppc_raw)
        pixels_per_cell = (ppc, ppc)

    cpb_raw = data.get("cells_per_block", 2)
    if isinstance(cpb_raw, (list, tuple)):
        cells_per_block = tuple(int(v) for v in cpb_raw[:2])
    else:
        cpb = int(cpb_raw)
        cells_per_block = (cpb, cpb)

    test_size = float(data.get("test_size", 0.2))

    try:
        result = train_model(
            n_neighbors     = n_neighbors,
            metric          = metric,
            weights         = weights,
            orientations    = orientations,
            pixels_per_cell = pixels_per_cell,
            cells_per_block = cells_per_block,
            test_size       = test_size,
        )
        status_code = 200 if result.get("success") else 400
        return jsonify(result), status_code

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Training error: {str(e)}",
        }), 500


@train_bp.route("/api/train/status", methods=["GET"])
def api_train_status():
    """
    GET /api/train/status
    Status model yang sedang aktif (ringkas, untuk dashboard).
    """
    try:
        meta = get_active_model_meta()
        if not meta:
            return jsonify({
                "has_model": False,
                "message":   "Belum ada model terlatih.",
                "model_meta": None,
            }), 200

        return jsonify({
            "has_model":  True,
            "message":    "Model aktif ditemukan.",
            "model_meta": meta,
        }), 200

    except Exception as e:
        return jsonify({"has_model": False, "message": str(e)}), 500


@train_bp.route("/api/model/info", methods=["GET"])
def api_model_info():
    """
    GET /api/model/info
    Metadata model lengkap dari file JSON (untuk Model Information card).
    Mencakup: Accuracy, Precision, Recall, F1, HOG params, KNN params,
    feature vector length, training time, dan distribusi sampel.
    """
    try:
        meta = get_latest_metadata()
        if not meta:
            return jsonify({
                "success": False,
                "message": "Belum ada metadata model. Lakukan training terlebih dahulu.",
                "data":    None,
            }), 200

        return jsonify({
            "success": True,
            "data":    meta,
        }), 200

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
