"""
tests/evaluate_model.py -- Skrip Evaluasi Model KNN
====================================================
Skrip ini mengevaluasi performa model KNN terlatih menggunakan
20% data uji yang disimpan saat training.

Metrik yang dihitung:
    - Accuracy (Akurasi)
    - Precision (per kelas + rata-rata weighted)
    - Recall (per kelas + rata-rata weighted)
    - F1-Score (per kelas + rata-rata weighted)
    - Confusion Matrix

Output:
    - Tabel laporan di terminal
    - File CSV: evaluation_results/evaluation_report_{timestamp}.csv
    - Gambar confusion matrix: evaluation_results/confusion_matrix_{timestamp}.png

BAB IV -- Hasil Penelitian:
    Evaluasi dilakukan pada 20% data uji yang tidak digunakan saat training,
    mengikuti protokol evaluasi standar machine learning untuk memastikan
    hasil yang objektif dan representatif.

Cara menjalankan:
    cd D:/handwriting-verification/backend
    python tests/evaluate_model.py
"""

import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")    # Non-interactive backend (tidak butuh display)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import joblib
from config import MODEL_DIR, EVALUATION_DIR
from model.trainer import get_latest_model_paths
from database import get_active_model_meta


def run_evaluation(save_results: bool = True) -> dict:
    """
    Menjalankan evaluasi lengkap terhadap model KNN aktif.

    Args:
        save_results (bool): Jika True, simpan laporan ke file CSV & PNG

    Returns:
        dict: Hasil evaluasi lengkap (accuracy, precision, recall, f1, dll.)
    """
    print("=" * 70)
    print("  EVALUASI MODEL -- Sistem Verifikasi Keaslian Tulisan Tangan")
    print("=" * 70)

    # --- Load model, label encoder, dan test data ---
    paths = get_latest_model_paths()
    if not paths:
        print("[EVAL] ERROR: Belum ada model terlatih. Jalankan training terlebih dahulu.")
        sys.exit(1)

    if not paths["test_data_path"] or not os.path.exists(paths["test_data_path"]):
        print("[EVAL] ERROR: File test data tidak ditemukan.")
        print(f"       Path: {paths['test_data_path']}")
        sys.exit(1)

    print(f"[EVAL] Memuat model dari: {paths['model_path']}")
    knn = joblib.load(paths["model_path"])

    le = None
    if paths["le_path"] and os.path.exists(paths["le_path"]):
        le = joblib.load(paths["le_path"])

    # Load test data
    test_data   = np.load(paths["test_data_path"], allow_pickle=True)
    X_test      = test_data["X_test"]
    y_test      = test_data["y_test"]
    class_names = list(test_data["class_names"])

    print(f"[EVAL] Data uji: {len(X_test)} sampel, {len(class_names)} kelas")

    # --- Prediksi ---
    y_pred = knn.predict(X_test)

    # --- Hitung metrik ---
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Per-kelas metrik
    prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec_per_class  = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class   = f1_score(y_test, y_pred, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # --- Tampilkan hasil ---
    print("\n" + "-" * 70)
    print(f"  AKURASI KESELURUHAN : {accuracy*100:.2f}%")
    print(f"  PRECISION (weighted): {precision*100:.2f}%")
    print(f"  RECALL (weighted)   : {recall*100:.2f}%")
    print(f"  F1-SCORE (weighted) : {f1*100:.2f}%")
    print("-" * 70)
    print("\n  LAPORAN PER KELAS:")
    print("-" * 70)
    print(f"  {'Nama Mahasiswa':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 70)

    for i, name in enumerate(class_names):
        if i < len(prec_per_class):
            print(f"  {name:<30} {prec_per_class[i]*100:>9.2f}% {rec_per_class[i]*100:>9.2f}% {f1_per_class[i]*100:>9.2f}%")

    print("-" * 70)
    print("\n  CLASSIFICATION REPORT (sklearn):")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # --- Simpan hasil ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp":   timestamp,
        "n_test":      len(X_test),
        "n_classes":   len(class_names),
        "accuracy":    round(accuracy * 100, 2),
        "precision":   round(precision * 100, 2),
        "recall":      round(recall * 100, 2),
        "f1_score":    round(f1 * 100, 2),
        "class_names": class_names,
        "per_class": {
            name: {
                "precision": round(float(prec_per_class[i]) * 100, 2) if i < len(prec_per_class) else 0,
                "recall":    round(float(rec_per_class[i])  * 100, 2) if i < len(rec_per_class) else 0,
                "f1_score":  round(float(f1_per_class[i])   * 100, 2) if i < len(f1_per_class) else 0,
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
    }

    if save_results:
        os.makedirs(EVALUATION_DIR, exist_ok=True)

        # --- Simpan CSV ---
        csv_path = os.path.join(EVALUATION_DIR, f"evaluation_report_{timestamp}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metrik", "Nilai (%)"])
            writer.writerow(["Accuracy",    results["accuracy"]])
            writer.writerow(["Precision",   results["precision"]])
            writer.writerow(["Recall",      results["recall"]])
            writer.writerow(["F1-Score",    results["f1_score"]])
            writer.writerow([])
            writer.writerow(["Nama Mahasiswa", "Precision (%)", "Recall (%)", "F1-Score (%)"])
            for name, metrics in results["per_class"].items():
                writer.writerow([name, metrics["precision"], metrics["recall"], metrics["f1_score"]])
        print(f"\n[EVAL] Laporan CSV disimpan: {csv_path}")

        # --- Simpan Confusion Matrix (PNG) ---
        cm_path = os.path.join(EVALUATION_DIR, f"confusion_matrix_{timestamp}.png")
        _plot_confusion_matrix(cm, class_names, cm_path)
        print(f"[EVAL] Confusion matrix disimpan: {cm_path}")
        results["csv_path"] = csv_path
        results["cm_path"]  = cm_path

    print("\n" + "=" * 70)
    return results


def _plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str):
    """
    Membuat dan menyimpan visualisasi confusion matrix sebagai heatmap.

    Args:
        cm (np.ndarray): Confusion matrix dari sklearn
        class_names (list): Nama setiap kelas
        save_path (str): Path untuk menyimpan gambar PNG
    """
    fig_size = max(8, len(class_names))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    # Heatmap dengan warna pink-maroon sesuai tema sistem
    cmap = sns.diverging_palette(15, 340, as_cmap=True)
    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap="RdPu",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title("Confusion Matrix -- Verifikasi Tulisan Tangan (KNN)", fontsize=14, pad=15)
    ax.set_xlabel("Prediksi", fontsize=12)
    ax.set_ylabel("Aktual", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_evaluation(save_results=True)
