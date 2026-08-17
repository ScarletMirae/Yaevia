"""
features/hog_extractor.py — Ekstraksi Fitur HOG (Histogram of Oriented Gradients)
====================================================================================
Modul ini mengimplementasikan ekstraksi fitur HOG dari citra tulisan tangan
yang telah dipreproses.

HOG bekerja dengan cara:
    1. Membagi citra menjadi sel-sel kecil (pixels_per_cell)
    2. Menghitung histogram gradien dalam setiap sel
    3. Mengelompokkan sel menjadi blok (cells_per_block) dan menormalisasinya
    4. Menggabungkan semua histogram menjadi satu feature vector

HOG dipilih karena:
    - Tahan terhadap perubahan pencahayaan lokal (karena normalisasi blok)
    - Efektif menangkap pola struktural dan bentuk karakter tulisan
    - Telah terbukti efektif untuk pengenalan pola visual

BAB IV — Implementasi:
    Feature vector HOG merepresentasikan karakteristik unik gaya tulisan
    setiap mahasiswa. Panjang vector = (H/py)*(W/px) * (cells_per_block)^2 * orientations
    Contoh: (128/8)*(128/8)*(2*2)*9 = 16*16*4*9 = 9216 fitur

Referensi:
    Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection.
    IEEE CVPR 2005. DOI: 10.1109/CVPR.2005.177
"""

import numpy as np
import os
import sys

from skimage.feature import hog

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    HOG_CELLS_PER_BLOCK,
    HOG_BLOCK_NORM,
    HOG_VISUALIZE,
    HOG_FEATURE_VECTOR,
)


# ==============================================================================
# FUNGSI EKSTRAKSI FITUR TUNGGAL
# ==============================================================================
def extract_hog_features(
    image: np.ndarray,
    orientations: int  = HOG_ORIENTATIONS,
    pixels_per_cell: tuple = HOG_PIXELS_PER_CELL,
    cells_per_block: tuple = HOG_CELLS_PER_BLOCK,
    block_norm: str  = HOG_BLOCK_NORM,
    visualize: bool  = HOG_VISUALIZE,
) -> np.ndarray:
    """
    Mengekstraksi feature vector HOG dari satu citra.

    Args:
        image (np.ndarray): Citra hasil preprocessing (grayscale, binary)
        orientations (int): Jumlah bin histogram orientasi gradien (default: 9)
        pixels_per_cell (tuple): Ukuran satu sel HOG dalam piksel (default: (8,8))
        cells_per_block (tuple): Ukuran blok untuk normalisasi (default: (2,2))
        block_norm (str): Metode normalisasi blok (default: 'L2-Hys')
        visualize (bool): Jika True, kembalikan juga gambar visualisasi HOG

    Returns:
        np.ndarray: Feature vector 1D dengan panjang = (image_h/py * image_w/px
                    * cpb_h * cpb_w * orientations)

        Jika visualize=True: tuple (feature_vector, hog_image)

    Raises:
        ValueError: Jika citra input tidak valid (None atau dimensi salah)
    """
    if image is None or image.size == 0:
        raise ValueError("Citra input tidak valid untuk ekstraksi HOG")

    # Normalisasi tipe data ke float64 untuk kompatibilitas skimage
    if image.dtype != np.float64:
        image_float = image.astype(np.float64) / 255.0
    else:
        image_float = image

    result = hog(
        image_float,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        visualize=visualize,
        feature_vector=HOG_FEATURE_VECTOR,
    )

    return result   # tuple jika visualize=True, ndarray jika visualize=False


# ==============================================================================
# FUNGSI EKSTRAKSI BATCH (SELURUH DATASET)
# ==============================================================================
def extract_features_batch(
    image_list: list,
    labels: list,
    orientations: int  = HOG_ORIENTATIONS,
    pixels_per_cell: tuple = HOG_PIXELS_PER_CELL,
    cells_per_block: tuple = HOG_CELLS_PER_BLOCK,
) -> tuple:
    """
    Mengekstraksi fitur HOG dari seluruh dataset sekaligus (batch processing).

    Args:
        image_list (list of np.ndarray): Daftar citra yang sudah dipreproses
        labels (list of str): Daftar label (nama mahasiswa) untuk setiap citra
        orientations, pixels_per_cell, cells_per_block: Parameter HOG

    Returns:
        tuple:
            - X (np.ndarray): Matrix fitur shape (n_samples, n_features)
            - y (list): Daftar label
            - failed_indices (list): Indeks citra yang gagal diekstraksi
    """
    X = []
    y = []
    failed_indices = []

    for i, (image, label) in enumerate(zip(image_list, labels)):
        try:
            features = extract_hog_features(
                image,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                visualize=False,
            )
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"[HOG] Gagal mengekstraksi fitur untuk indeks {i}: {e}")
            failed_indices.append(i)

    return np.array(X), y, failed_indices


# ==============================================================================
# SIMPAN & LOAD FEATURE VECTORS
# ==============================================================================
def save_features(X: np.ndarray, y: list, save_dir: str, prefix: str = "features"):
    """
    Menyimpan feature matrix dan labels ke file numpy (.npy).

    File yang disimpan:
        - {prefix}_X.npy : Matrix fitur shape (n_samples, n_features)
        - {prefix}_y.npy : Array label

    Args:
        X (np.ndarray): Matrix fitur
        y (list): Daftar label
        save_dir (str): Direktori penyimpanan
        prefix (str): Prefix nama file
    """
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{prefix}_X.npy"), X)
    np.save(os.path.join(save_dir, f"{prefix}_y.npy"), np.array(y))
    print(f"[HOG] Fitur disimpan: {X.shape} samples × {X.shape[1]} features")
    print(f"[HOG] Labels disimpan: {len(y)} entri")


def load_features(save_dir: str, prefix: str = "features") -> tuple:
    """
    Memuat feature matrix dan labels dari file numpy.

    Args:
        save_dir (str): Direktori penyimpanan
        prefix (str): Prefix nama file

    Returns:
        tuple: (X: np.ndarray, y: list) — feature matrix dan daftar label

    Raises:
        FileNotFoundError: Jika file features tidak ditemukan
    """
    x_path = os.path.join(save_dir, f"{prefix}_X.npy")
    y_path = os.path.join(save_dir, f"{prefix}_y.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"File features tidak ditemukan di: {save_dir}. "
            "Jalankan training terlebih dahulu."
        )

    X = np.load(x_path, allow_pickle=True)
    y = list(np.load(y_path, allow_pickle=True))
    print(f"[HOG] Fitur dimuat: {X.shape} samples × features, {len(y)} labels")
    return X, y


def get_feature_vector_length(
    image_size: tuple,
    pixels_per_cell: tuple = HOG_PIXELS_PER_CELL,
    cells_per_block: tuple = HOG_CELLS_PER_BLOCK,
    orientations: int = HOG_ORIENTATIONS,
) -> int:
    """
    Menghitung panjang feature vector HOG secara teoritis.
    Berguna untuk dokumentasi skripsi dan validasi konfigurasi.

    Formula:
        n_cells_row = image_h / pixels_per_cell[0]
        n_cells_col = image_w / pixels_per_cell[1]
        n_blocks_row = n_cells_row - cells_per_block[0] + 1
        n_blocks_col = n_cells_col - cells_per_block[1] + 1
        feature_length = n_blocks_row * n_blocks_col * prod(cells_per_block) * orientations

    Returns:
        int: Panjang feature vector
    """
    h, w = image_size
    n_blocks_row = (h // pixels_per_cell[0]) - cells_per_block[0] + 1
    n_blocks_col = (w // pixels_per_cell[1]) - cells_per_block[1] + 1
    feature_length = (
        n_blocks_row * n_blocks_col
        * cells_per_block[0] * cells_per_block[1]
        * orientations
    )
    return feature_length
