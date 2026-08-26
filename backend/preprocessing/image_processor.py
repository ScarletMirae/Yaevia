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
# TAHAP 2: ASPECT-RATIO PRESERVING RESIZE (Letterboxing)
# ==============================================================================
def resize_with_aspect_ratio(image: np.ndarray, target_size: tuple = IMAGE_SIZE) -> np.ndarray:
    """
    Mengubah ukuran citra ke target_size dengan mempertahankan rasio aspek asli (aspect ratio).
    Citra diletakkan di tengah kanvas kosong (letterboxing / padding).

    Alasan:
        Peregangan non-proporsional (misal menarik teks lebar menjadi kotak)
        mengubah sudut kemiringan goresan (slant) tulisan tangan, sehingga
        histogram orientasi gradien (HOG) menjadi sangat berbeda antar foto.
        Dengan letterboxing, proporsi dan sudut goresan tulisan tetap terjaga 100%.

    Args:
        image (np.ndarray): Citra input (grayscale atau biner)
        target_size (tuple): Target ukuran (width, height), default dari config.py

    Returns:
        np.ndarray: Citra berukuran tepat target_size dengan padding di sekelilingnya
    """
    h_orig, w_orig = image.shape[:2]
    if h_orig == 0 or w_orig == 0:
        return np.zeros(target_size, dtype=np.uint8)

    tw, th = target_size
    scale = min(tw / w_orig, th / h_orig)
    new_w = max(1, int(w_orig * scale))
    new_h = max(1, int(h_orig * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Buat kanvas hitam berukuran target_size dan tempatkan citra di tengah
    canvas = np.zeros((th, tw), dtype=image.dtype)
    dx = (tw - new_w) // 2
    dy = (th - new_h) // 2
    canvas[dy:dy + new_h, dx:dx + new_w] = resized

    return canvas


def resize_image(image: np.ndarray, size: tuple = IMAGE_SIZE) -> np.ndarray:
    """
    Wrapper fungsi resize untuk backward-compatibility.
    Menggunakan resize_with_aspect_ratio untuk mempertahankan rasio aspek.
    """
    return resize_with_aspect_ratio(image, size)


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
# TAHAP 6: ROI SEGMENTATION (Region of Interest) PADA RESOLUSI TINGGI
# ==============================================================================
def extract_roi(image: np.ndarray, padding_ratio: float = 0.05) -> np.ndarray:
    """
    Mendeteksi dan memotong area tulisan (ROI) dari citra biner beresolusi tinggi.

    Menggunakan deteksi kontur untuk menemukan bounding box terkecil
    yang mencakup semua area tulisan, lalu memotong area tersebut
    dengan padding proporsional.

    Args:
        image (np.ndarray): Citra biner setelah noise removal
        padding_ratio (float): Rasio padding relatif terhadap dimensi tulisan

    Returns:
        np.ndarray: Crop ROI pada resolusi asli (belum di-downscale)
    """
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image   # Fallback: kembalikan citra asli

    # Gabungkan semua kontur untuk mendapatkan bounding box keseluruhan tulisan
    all_points = np.concatenate(contours, axis=0)
    x, y, w, h = cv2.boundingRect(all_points)

    # Tambahkan padding proporsional terhadap ukuran tulisan
    h_img, w_img = image.shape[:2]
    padding = max(4, int(min(w, h) * padding_ratio))
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)

    roi = image[y1:y2, x1:x2]
    return roi


# ==============================================================================
# PIPELINE UTAMA
# ==============================================================================
def preprocess_image(image_path: str, save_path: str = None) -> np.ndarray:
    """
    Menjalankan seluruh pipeline preprocessing secara berurutan:
        1. Grayscale Conversion (resolusi asli)
        2. Gaussian Blur (smoothing awal)
        3. Otsu's Thresholding (binarisasi adaptif)
        4. Noise Removal (median blur + morphological)
        5. ROI Segmentation (pemotongan area tulisan pada resolusi asli)
        6. Aspect-Ratio Preserving Resize (normalisasi ke 128x128 via letterboxing)

    Keunggulan urutan ini:
        - Binarisasi Otsu dan ROI dilakukan pada resolusi asli citra,
          mencegah hilangnya goresan halus.
        - Normalisasi ukuran menggunakan letterboxing mempertahankan
          kemiringan dan proporsi tulisan tangan.

    Args:
        image_path (str): Path file citra input (JPG/PNG)
        save_path (str): Jika diberikan, simpan hasil preprocessing ke path ini

    Returns:
        np.ndarray: Citra hasil preprocessing (128x128 biner) siap untuk ekstraksi HOG

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

    return preprocess_from_array(image, save_path=save_path)


def preprocess_from_array(image_array: np.ndarray, save_path: str = None) -> np.ndarray:
    """
    Versi preprocess_image yang menerima numpy array langsung
    (untuk gambar yang sudah di-load di memory, misal dari upload Flask).

    Args:
        image_array (np.ndarray): Citra dalam format numpy array (BGR)
        save_path (str): Path penyimpanan opsional

    Returns:
        np.ndarray: Citra hasil preprocessing (128x128 biner)
    """
    # --- Tahap 1: Grayscale ---
    gray = convert_to_grayscale(image_array)

    # --- Tahap 2: Gaussian Blur ---
    blurred = apply_gaussian_blur(gray)

    # --- Tahap 3: Otsu's Thresholding ---
    binary = apply_otsu_threshold(blurred)

    # --- Tahap 4: Noise Removal ---
    denoised = remove_noise(binary)

    # --- Tahap 5: ROI Segmentation (pada resolusi asli) ---
    roi = extract_roi(denoised)

    # --- Tahap 6: Aspect-Ratio Preserving Resize (Letterboxing ke IMAGE_SIZE) ---
    processed = resize_with_aspect_ratio(roi, IMAGE_SIZE)

    # --- Simpan hasil preprocessing (opsional) ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, processed)

    return processed

