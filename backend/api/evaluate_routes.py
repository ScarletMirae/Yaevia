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
from model.trainer import get_latest_model_paths, get_latest_metadata, evaluate_k_comparison
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


def _load_loocv_confusion_matrix():
    """Memuat confusion matrix LOOCV K=5 resmi dari berkas evaluasi eksperimen."""
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests", "evaluation_results", "knn_final_experiment", "confusion_k5.csv"
    )
    if not os.path.exists(csv_path):
        return None

    try:
        matrix = []
        labels = []
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        
        # Header baris pertama
        header = [h.strip() for h in lines[0].split(",")[1:]]
        labels = header

        for line in lines[1:]:
            parts = [p.strip() for p in line.split(",")]
            row_vals = [int(v) for v in parts[1:]]
            matrix.append(row_vals)

        return {"matrix": matrix, "labels": labels}
    except Exception as e:
        logger.warning(f"Gagal memuat LOOCV confusion matrix dari CSV: {e}")
        return None


@evaluate_bp.route("/api/evaluate", methods=["GET"])
def api_evaluate():
    """
    GET /api/evaluate
    BAB IV / BAB V — Metodologi Evaluasi Model Resmi:
    
    Menampilkan metrik evaluasi ilmiah yang valid secara metodologis:
    1. Validasi Internal (LOOCV):
       - Metrik: Akurasi (61.94%), Macro Precision (63.21%), Macro Recall (61.94%), Macro F1 (61.29%)
       - 360 iterasi di mana sampel query dikeluarkan dari himpunan referensi (Bebas Data Leakage).
       - Performa Halaman #1 (33.33%) vs Halaman #2–20 (63.45%).
    2. Dataset Referensi Produksi:
       - 18 mahasiswa, 20 citra/mahasiswa, total 360 citra.
       - Digunakan sebagai reference set penuh untuk inferensi KNN produksi.
    3. Confusion Matrix: 18x18 matrix hasil pengujian LOOCV K=5.
    4. Per-Class LOOCV Performance: Akurasi per mahasiswa dari LOOCV.
    """
    try:
        meta = get_latest_metadata() or {}
        loocv_meta = meta.get("loocv_benchmark", {})

        # ── Metrik LOOCV Authoritative Benchmark ────────────────
        loocv_metrics = {
            "accuracy":            loocv_meta.get("loocv_accuracy", 61.94),
            "precision_macro":     loocv_meta.get("loocv_precision_macro", 63.21),
            "recall_macro":        loocv_meta.get("loocv_recall_macro", 61.94),
            "f1_macro":            loocv_meta.get("loocv_f1_macro", 61.29),
            "evaluated_samples":   loocv_meta.get("loocv_n_samples", 360),
            "correct_samples":     loocv_meta.get("loocv_correct_count", 223),
            "wrong_samples":       loocv_meta.get("loocv_wrong_count", 137),
            "position_1_accuracy": loocv_meta.get("position_1_accuracy", 33.33),
            "position_2_20_acc":   loocv_meta.get("position_2_20_accuracy", 63.45),
            "methodology":         "Leave-One-Out Cross-Validation (LOOCV, 360 Folds)",
            "leakage_safe":        True,
        }

        # ── Data Per Mahasiswa (LOOCV K=5) ──────────────────────
        per_class_loocv = [
            {"name": "Angela Permata Rosa",        "total": 20, "correct": 11, "wrong": 9,  "accuracy": 55.0},
            {"name": "Bramasetya Raka Purnama",    "total": 20, "correct": 10, "wrong": 10, "accuracy": 50.0},
            {"name": "Dimas Wahyu Prasetyo",       "total": 20, "correct": 11, "wrong": 9,  "accuracy": 55.0},
            {"name": "Fahim J Mujaddid",           "total": 20, "correct": 9,  "wrong": 11, "accuracy": 45.0},
            {"name": "Farhan Agiya Pratama",       "total": 20, "correct": 10, "wrong": 10, "accuracy": 50.0},
            {"name": "Fathurrahman Nugroho",       "total": 20, "correct": 16, "wrong": 4,  "accuracy": 80.0},
            {"name": "Febrian Dinnar Purnama",     "total": 20, "correct": 7,  "wrong": 13, "accuracy": 35.0},
            {"name": "Hazelando Visco",            "total": 20, "correct": 14, "wrong": 6,  "accuracy": 70.0},
            {"name": "Ibnu Gayuh Fadilah",         "total": 20, "correct": 11, "wrong": 9,  "accuracy": 55.0},
            {"name": "Ilham Rasyidan Muhammad",    "total": 20, "correct": 10, "wrong": 10, "accuracy": 50.0},
            {"name": "Muhammad Alif Rizky Hutama", "total": 20, "correct": 17, "wrong": 3,  "accuracy": 85.0},
            {"name": "Muhammad Dony Saputra",      "total": 20, "correct": 12, "wrong": 8,  "accuracy": 60.0},
            {"name": "Raditya Endra Mahardika",    "total": 20, "correct": 11, "wrong": 9,  "accuracy": 55.0},
            {"name": "Rakha Burhannudin Majid",    "total": 20, "correct": 18, "wrong": 2,  "accuracy": 90.0},
            {"name": "Rifqi Rengga Praseno",       "total": 20, "correct": 17, "wrong": 3,  "accuracy": 85.0},
            {"name": "Soni Nugroho",               "total": 20, "correct": 13, "wrong": 7,  "accuracy": 65.0},
            {"name": "Wiridan Syifa Saputra",      "total": 20, "correct": 9,  "wrong": 11, "accuracy": 45.0},
            {"name": "Zaedani Ni'am Masykur",      "total": 20, "correct": 17, "wrong": 3,  "accuracy": 85.0},
        ]

        # ── Confusion Matrix LOOCV ──────────────────────────────
        cm_data = _load_loocv_confusion_matrix()

        # ── Dataset Referensi Info ──────────────────────────────
        ref_dataset_info = {
            "n_respondents":        meta.get("n_respondents", 18),
            "n_total_dataset":      meta.get("n_total_dataset", 360),
            "samples_per_student":  20,
            "description":          "360 citra tulisan tangan yang digunakan sebagai basis data referensi model KNN produksi.",
        }

        # ── Model Hyperparameters ───────────────────────────────
        model_info = {
            "knn_k":               meta.get("knn_k", 5),
            "knn_metric":          meta.get("knn_metric", "euclidean"),
            "knn_weights":         meta.get("knn_weights", "distance"),
            "hog_orientations":    meta.get("hog_orientations", 9),
            "hog_pixels_per_cell": meta.get("hog_pixels_per_cell", [8, 8]),
            "hog_cells_per_block": meta.get("hog_cells_per_block", [2, 2]),
            "hog_block_norm":      meta.get("hog_block_norm", "L2-Hys"),
            "feature_vector_size": meta.get("feature_vector_size", 34596),
            "image_size":          meta.get("image_size", [256, 256]),
            "training_time":       meta.get("training_time_seconds", 0),
            "train_timestamp":     meta.get("train_timestamp", ""),
        }

        # ── Data Pengujian Verifikasi Riwayat Berlabel (Ground Truth) ──
        conn = get_connection()
        verif_rows = conn.execute("""
            SELECT ground_truth_name, predicted_name, is_correct
            FROM verifications
            WHERE ground_truth_name IS NOT NULL AND TRIM(ground_truth_name) != ''
        """).fetchall()
        conn.close()

        n_live = len(verif_rows)
        if n_live > 0:
            n_correct = sum(1 for r in verif_rows if r["is_correct"] == 1)
            n_wrong   = n_live - n_correct
            live_acc  = round((n_correct / n_live) * 100.0, 2)
            live_eval = {
                "has_records":      True,
                "n_samples":        n_live,
                "correct_count":    n_correct,
                "wrong_count":      n_wrong,
                "accuracy":         live_acc,
                "description":      f"Metrik pengujian riil dari {n_live} record verifikasi pengguna berlabel.",
            }
        else:
            live_eval = {
                "has_records":      False,
                "n_samples":        0,
                "correct_count":    0,
                "wrong_count":      0,
                "accuracy":         0.0,
                "description":      "Belum tersedia data pengujian berlabel yang cukup.",
            }

        return jsonify({
            "success":                     True,
            "loocv_metrics":               loocv_metrics,
            "live_verification_eval":      live_eval,
            "metrics": {
                "test_accuracy":   loocv_metrics["accuracy"],
                "precision_macro": loocv_metrics["precision_macro"],
                "recall_macro":    loocv_metrics["recall_macro"],
                "f1_macro":        loocv_metrics["f1_macro"],
            },
            "reference_dataset":   ref_dataset_info,
            "per_class_chart":     per_class_loocv,
            "confusion_matrix":    cm_data,
            "model_info":          model_info,
        }), 200

    except Exception as e:
        logger.error(f"Evaluate error: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500


@evaluate_bp.route("/api/evaluate/compare-k", methods=["GET"])
def api_evaluate_compare_k():
    """
    GET /api/evaluate/compare-k
    Mengevaluasi dan membandingkan performa berbagai nilai K ganjil (K=3, 5, 7, 9)
    pada data pengujian yang sama tanpa memicu retraining ataupun data leakage.
    """
    try:
        res = evaluate_k_comparison()
        status_code = 200 if res.get("success") else 400
        return jsonify(res), status_code
    except Exception as e:
        logger.error(f"Compare-K error: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500

