"""
config.py — Konfigurasi Global Sistem Verifikasi Tulisan Tangan
=================================================================
Semua hyperparameter HOG, KNN, dan threshold similarity dipusatkan di sini.
Untuk keperluan eksperimen skripsi, ubah nilai di bagian ini.

BAB IV — Implementasi:
    Semua hyperparameter HOG dan KNN dipusatkan di sini sehingga
    eksperimenter cukup mengubah nilai pada file ini tanpa menyentuh
    kode inti di modul lain.
"""

import os

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_RAW_DIR       = os.path.join(BASE_DIR, "dataset", "raw")
DATASET_PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
MODEL_DIR             = os.path.join(BASE_DIR, "model", "saved")
MODEL_SAVED_DIR       = MODEL_DIR   # alias
DB_PATH               = os.path.join(BASE_DIR, "database.db")
EVALUATION_DIR        = os.path.join(BASE_DIR, "tests", "evaluation_results")

# ==============================================================================
# IMAGE PREPROCESSING PARAMETERS
# ==============================================================================
IMAGE_SIZE           = (128, 128)
GAUSSIAN_BLUR_KERNEL = (5, 5)
MEDIAN_BLUR_KERNEL   = 3
MORPH_KERNEL_SIZE    = (3, 3)

# ==============================================================================
# HOG FEATURE EXTRACTION PARAMETERS
# Referensi: Dalal & Triggs (2005)
# EKSPERIMEN: Ubah nilai di bawah untuk membandingkan performa pada skripsi
# ==============================================================================
HOG_ORIENTATIONS    = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM      = "L2-Hys"
HOG_VISUALIZE       = False
HOG_FEATURE_VECTOR  = True

# ==============================================================================
# KNN CLASSIFIER PARAMETERS
# EKSPERIMEN: Ubah K dan metric untuk analisis perbandingan di skripsi
# ==============================================================================
KNN_N_NEIGHBORS = 5
KNN_METRIC      = "euclidean"
KNN_WEIGHTS     = "distance"
KNN_ALGORITHM   = "auto"

# ==============================================================================
# TRAIN/TEST SPLIT
# BAB IV: Rasio 80:20 digunakan sesuai dengan metodologi penelitian
# ==============================================================================
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ==============================================================================
# SIMILARITY THRESHOLDS (Berbasis Euclidean Distance)
# BAB IV — Klasifikasi Status Kemiripan:
#   Threshold dikalibrasi berdasarkan distribusi jarak HOG feature vector.
#   Nilai dapat disesuaikan berdasarkan hasil eksperimen validasi.
# ==============================================================================
SIMILARITY_THRESHOLDS = {
    "sangat_mirip":  70.0,   # >= 70% → SANGAT MIRIP
    "mirip":         50.0,   # >= 50% → MIRIP
    "kurang_mirip":  30.0,   # >= 30% → KURANG MIRIP
    # < 30%  → TIDAK MIRIP
}

# Minimum sampel per mahasiswa agar bisa diikutkan training
MIN_SAMPLES_PER_CLASS = 2

# ==============================================================================
# DATASET METADATA
# ==============================================================================
MATA_KULIAH_OPTIONS = [
    "Algoritma Pemrograman (Alpro)",
    "Pemrograman Berorientasi Objek (PBO)",
    "Sistem Operasi",
]

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

# ==============================================================================
# FLASK SERVER CONFIGURATION
# ==============================================================================
FLASK_HOST  = "0.0.0.0"
FLASK_PORT  = 5000
FLASK_DEBUG = True
SECRET_KEY  = "handwriting-verification-secret-key-2024"
