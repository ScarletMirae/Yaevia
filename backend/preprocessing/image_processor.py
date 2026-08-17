"""
preprocessing/image_processor.py — Pipeline Preprocessing Citra
=================================================================
Modul ini mengimplementasikan seluruh tahapan preprocessing citra
tulisan tangan sebelum ekstraksi fitur HOG dilakukan.

Tahapan preprocessing (berurutan):
    1. Grayscale Conversion  — konversi ke citra abu-abu
    2. Resize                — normalisasi ukuran
    3. Gaussian Blur         — smoothing untuk reduksi noise awal
    4. Otsu's Thresholding   — binarisasi adaptif
    5. Noise Removal         — median blur + morphological operations
    6. ROI Segmentation      — deteksi dan pemotongan area tulisan

BAB IV — Implementasi:
    Preprocessing dilakukan untuk meningkatkan kualitas citra sebelum
    ekstraksi fitur, sehingga fitur yang dihasilkan lebih representatif
    dan tidak terganggu oleh artefak (bayangan, lipatan kertas, dll).

Referensi:
    - Gonzalez, R.C. & Woods, R.E. (2018). Digital Image Processing, 4th Ed.
    - OpenCV Documentation: https://docs.opencv.org/
"""

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    IMAGE_SIZE,
    GAUSSIAN_BLUR_KERNEL,
    MEDIAN_BLUR_KERNEL,
    MORPH_KERNEL_SIZE,
)


# ==============================================================================
# TAHAP 1: GRAYSCALE CONVERSION
# ==============================================================================
def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Mengkonversi citra BGR ke grayscale.

    Citra tulisan tangan tidak memerlukan informasi warna untuk identifikasi;
    konversi ke grayscale menyederhanakan representasi dan mengurangi
    dimensi data dari 3-channel menjadi 1-channel.

    Args:
        image (np.ndarray): Citra input dalam format BGR (hasil cv2.imread)

    Returns:
        np.ndarray: Citra grayscale 1-channel
    """
    if len(image.shape) == 2:
        return image   # Sudah grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ==============================================================================
# TAHAP 2: RESIZE / NORMALISASI UKURAN
# ==============================================================================
def resize_image(image: np.ndarray, size: tuple = IMAGE_SIZE) -> np.ndarray:
    """
    Mengubah ukuran citra menjadi dimensi seragam (IMAGE_SIZE dari config.py).

    Normalisasi ukuran diperlukan agar feature vector HOG yang dihasilkan
    selalu memiliki panjang yang sama untuk semua citra dalam dataset.

    Args:
        image (np.ndarray): Citra input
        size (tuple): Target ukuran (width, height), default dari config.py

    Returns:
        np.ndarray: Citra dengan ukuran seragam
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


# ==============================================================================
# TAHAP 3: GAUSSIAN BLUR (Smoothing Awal)
# ==============================================================================
def apply_gaussian_blur(image: np.ndarray,
                        kernel: tuple = GAUSSIAN_BLUR_KERNEL) -> np.ndarray:
    """
    Menerapkan Gaussian blur untuk mengurangi noise frekuensi tinggi.

    Blur dilakukan sebelum thresholding agar transisi tepi lebih halus
    sehingga hasil binarisasi lebih bersih.

    Args:
        image (np.ndarray): Citra grayscale
        kernel (tuple): Ukuran kernel Gaussian, default dari config.py

    Returns:
        np.ndarray: Citra setelah Gaussian blur
    """
    return cv2.GaussianBlur(image, kernel, sigmaX=0)


# ==============================================================================
# TAHAP 4: OTSU'S THRESHOLDING (Binarisasi)
# ==============================================================================
def apply_otsu_threshold(image: np.ndarray) -> np.ndarray:
    """
    Menerapkan metode Otsu untuk thresholding adaptif otomatis.

    Metode Otsu secara otomatis menentukan nilai threshold optimal dengan
    memaksimalkan variansi antar kelas (piksel latar vs. piksel tulisan).
    Hasilnya adalah citra biner: piksel tulisan = putih (255), latar = hitam (0).

    Referensi:
        Otsu, N. (1979). A threshold selection method from gray-level histograms.
        IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62–66.

    Args:
        image (np.ndarray): Citra grayscale setelah blur

    Returns:
        np.ndarray: Citra biner setelah thresholding
    """
    # THRESH_BINARY_INV membalik warna: tulisan jadi putih di atas latar hitam
    # Ini memudahkan deteksi kontur untuk segmentasi ROI
    _, binary = cv2.threshold(
        image, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


# ==============================================================================
# TAHAP 5: NOISE REMOVAL
# ==============================================================================
def remove_noise(image: np.ndarray,
                 median_k: int = MEDIAN_BLUR_KERNEL,
                 morph_k: tuple = MORPH_KERNEL_SIZE) -> np.ndarray:
    """
    Menghilangkan noise residual menggunakan median blur + morphological operations.

    - Median blur efektif menghilangkan salt-and-pepper noise tanpa
      memblur tepi tulisan.
    - Morphological closing (dilasi lalu erosi) mengisi lubang kecil
      dalam karakter tulisan yang terputus.

    Args:
        image (np.ndarray): Citra biner setelah thresholding
        median_k (int): Ukuran kernel median blur, default dari config.py
        morph_k (tuple): Ukuran kernel morphological, default dari config.py

    Returns:
        np.ndarray: Citra biner setelah noise removal
    """
    # Median blur untuk menghilangkan titik-titik noise kecil
    denoised = cv2.medianBlur(image, median_k)

    # Morphological closing: tutup lubang kecil di karakter tulisan
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_k)
    closed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

    # Morphological opening: hapus noise kecil yang masih tersisa
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    return opened


# ==============================================================================
# TAHAP 6: ROI SEGMENTATION (Region of Interest)
# ==============================================================================
def extract_roi(image: np.ndarray, padding: int = 10) -> np.ndarray:
    """
    Mendeteksi dan memotong area tulisan (ROI) dari citra biner.

    Menggunakan deteksi kontur untuk menemukan bounding box terkecil
    yang mencakup semua area tulisan, lalu memotong area tersebut
    dengan padding tambahan.

    Jika tidak ada kontur terdeteksi (gambar kosong/sangat noise),
    fungsi mengembalikan citra asli tanpa perubahan.

    Args:
        image (np.ndarray): Citra biner setelah noise removal
        padding (int): Piksel padding di sekitar bounding box ROI

    Returns:
        np.ndarray: Citra yang sudah dipotong pada area tulisan
    """
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image   # Fallback: kembalikan citra asli

    # Gabungkan semua kontur untuk mendapatkan bounding box keseluruhan tulisan
    all_points = np.concatenate(contours, axis=0)
    x, y, w, h = cv2.boundingRect(all_points)

    # Tambahkan padding dan pastikan tidak keluar batas gambar
    h_img, w_img = image.shape
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)

    roi = image[y1:y2, x1:x2]

    # Resize ROI kembali ke IMAGE_SIZE agar konsisten dengan preprocessing lain
    roi_resized = cv2.resize(roi, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return roi_resized


# ==============================================================================
# PIPELINE UTAMA
# ==============================================================================
def preprocess_image(image_path: str, save_path: str = None) -> np.ndarray:
    """
    Menjalankan seluruh pipeline preprocessing secara berurutan:
        grayscale → resize → gaussian blur → otsu threshold → noise removal → ROI

    Args:
        image_path (str): Path file citra input (JPG/PNG)
        save_path (str): Jika diberikan, simpan hasil preprocessing ke path ini

    Returns:
        np.ndarray: Citra hasil preprocessing siap untuk ekstraksi HOG

    Raises:
        FileNotFoundError: Jika file gambar tidak ditemukan
        ValueError: Jika file tidak dapat dibaca sebagai gambar
    """
    # --- Load citra ---
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File gambar tidak ditemukan: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Tidak dapat membaca file gambar: {image_path}")

    # --- Tahap 1: Grayscale ---
    gray = convert_to_grayscale(image)

    # --- Tahap 2: Resize ---
    resized = resize_image(gray)

    # --- Tahap 3: Gaussian Blur ---
    blurred = apply_gaussian_blur(resized)

    # --- Tahap 4: Otsu's Thresholding ---
    binary = apply_otsu_threshold(blurred)

    # --- Tahap 5: Noise Removal ---
    denoised = remove_noise(binary)

    # --- Tahap 6: ROI Segmentation ---
    processed = extract_roi(denoised)

    # --- Simpan hasil preprocessing (opsional) ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, processed)

    return processed


def preprocess_from_array(image_array: np.ndarray) -> np.ndarray:
    """
    Versi preprocess_image yang menerima numpy array langsung
    (untuk gambar yang sudah di-load di memory, misal dari upload Flask).

    Args:
        image_array (np.ndarray): Citra dalam format numpy array (BGR)

    Returns:
        np.ndarray: Citra hasil preprocessing
    """
    gray     = convert_to_grayscale(image_array)
    resized  = resize_image(gray)
    blurred  = apply_gaussian_blur(resized)
    binary   = apply_otsu_threshold(blurred)
    denoised = remove_noise(binary)
    processed = extract_roi(denoised)
    return processed
