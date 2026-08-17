"""
api/evaluate_routes.py — Endpoint Evaluasi Model
==================================================
Menyediakan endpoint untuk halaman Evaluasi:
  GET /api/evaluate          — metrik + data chart (tanpa retraining)
  GET /api/evaluate/chart-data — raw data untuk Chart.js
"""

import os
import sys
import json
import logging
import numpy as np
import joblib
import glob

from flask import Blueprint, jsonify

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.trainer import get_latest_model_paths, get_latest_metadata
from database import get_connection

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix,
    )
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

evaluate_bp = Blueprint("evaluate", __name__)
logger      = logging.getLogger(__name__)


def _load_eval_data():
    """
    Memuat data testing dari file joblib untuk evaluasi ulang.
    Returns (X_test, y_test, label_encoder, knn_model) atau None jika belum ada.
    """
    paths = get_latest_model_paths()
    if not paths:
        return None

    Xtest_path = paths.get("Xtest_path")
    ytest_path = paths.get("ytest_path")
    le_path    = paths.get("le_path")
    model_path = paths.get("model_path")

    if not all(p and os.path.exists(p) for p in [Xtest_path, ytest_path, le_path, model_path]):
        return None

    return {
        "X_test":  joblib.load(Xtest_path),
        "y_test":  joblib.load(ytest_path),
        "le":      joblib.load(le_path),
        "model":   joblib.load(model_path),
    }


@evaluate_bp.route("/api/evaluate", methods=["GET"])
def api_evaluate():
    """
    GET /api/evaluate
    Mengembalikan semua data evaluasi model:
    - Akurasi, Precision, Recall, F1 Score (dari metadata JSON)
    - Confusion Matrix (per-kelas)
    - Distribusi data training vs testing per kelas
    - Distribusi jumlah sampel per mahasiswa
    """
    try:
        # Prioritas: baca dari metadata JSON (cepat, tanpa recompute)
        meta = get_latest_metadata()
        if not meta:
            return jsonify({
                "success": False,
                "message": "Belum ada model terlatih. Lakukan training terlebih dahulu.",
            }), 200

        # ── Metrik utama dari metadata ─────────────────────────
        metrics = {
            "train_accuracy":  meta.get("train_accuracy", 0),
            "test_accuracy":   meta.get("test_accuracy",  0),
            "precision_macro": meta.get("precision_macro", 0),
            "recall_macro":    meta.get("recall_macro",    0),
            "f1_macro":        meta.get("f1_macro",        0),
        }

        # ── Distribusi sampel per mahasiswa ─────────────────────
        label_counts = meta.get("label_counts", {})
        class_names  = meta.get("class_names", [])
        n_train      = meta.get("n_train_samples", 0)
        n_test       = meta.get("n_test_samples",  0)
        n_total      = meta.get("n_total_dataset",  0)
        test_size    = meta.get("test_size", 0.2)

        # Estimasi jumlah training/testing per kelas
        per_class_chart = []
        for name in class_names:
            total_class = label_counts.get(name, 0)
            n_test_cls  = max(1, round(total_class * test_size))
            n_train_cls = total_class - n_test_cls
            per_class_chart.append({
                "name":    name,
                "train":   n_train_cls,
                "test":    n_test_cls,
                "total":   total_class,
            })

        # ── Confusion matrix (recompute dari test data jika tersedia) ──
        cm_data = None
        try:
            eval_data = _load_eval_data()
            if eval_data and SKLEARN_OK:
                X_test = eval_data["X_test"]
                y_test = eval_data["y_test"]
                model  = eval_data["model"]
                le     = eval_data["le"]

                y_pred = model.predict(X_test)
                cm     = confusion_matrix(y_test, y_pred)
                labels = [le.inverse_transform([i])[0] for i in range(len(le.classes_))]

                cm_data = {
                    "matrix":  cm.tolist(),
                    "labels":  labels,
                }
        except Exception as cm_err:
            logger.warning(f"Tidak bisa hitung confusion matrix: {cm_err}")

        return jsonify({
            "success":         True,
            "metrics":         metrics,
            "per_class_chart": per_class_chart,
            "confusion_matrix": cm_data,
            "model_info": {
                "n_respondents":       meta.get("n_respondents", 0),
                "n_total_dataset":     n_total,
                "n_train_samples":     n_train,
                "n_test_samples":      n_test,
                "test_size":           test_size,
                "knn_k":               meta.get("knn_k", 5),
                "knn_metric":          meta.get("knn_metric", "euclidean"),
                "hog_orientations":    meta.get("hog_orientations", 9),
                "hog_pixels_per_cell": meta.get("hog_pixels_per_cell", [8,8]),
                "hog_cells_per_block": meta.get("hog_cells_per_block", [2,2]),
                "feature_vector_size": meta.get("feature_vector_size", 0),
                "training_time":       meta.get("training_time_seconds", 0),
                "train_timestamp":     meta.get("train_timestamp", ""),
            },
            "label_counts": label_counts,
        }), 200

    except Exception as e:
        logger.error(f"Evaluate error: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500
