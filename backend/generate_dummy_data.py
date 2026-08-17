"""
generate_dummy_data.py — Generator Data Sintetis Tulisan Tangan
================================================================
Skrip ini menghasilkan dataset dummy untuk 30 mahasiswa (sesuai target skripsi),
masing-masing 5 sampel per mahasiswa → total 150 gambar.

Data sintetis dibuat dengan:
    - Variasi teks (nama mahasiswa + kode unik)
    - Variasi font, ukuran, kemiringan
    - Variasi noise (gaussian noise + blur ringan)
    - Simulasi tekstur kertas (background tidak seragam)

Tujuan:
    Memungkinkan demonstrasi dan pengujian sistem secara penuh
    sebelum dataset asli tulisan tangan 30 responden tersedia.

Cara menjalankan:
    cd D:/handwriting-verification/backend
    python generate_dummy_data.py

Output:
    - Gambar disimpan ke: dataset/raw/dummy_{nama}_{i}.png
    - Data terdaftar di database SQLite
"""

import os
import sys
import io
import uuid
import random
import numpy as np
import cv2
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database import init_db, get_connection
from config import DATASET_RAW_DIR, DATASET_PROCESSED_DIR, MATA_KULIAH_OPTIONS
from preprocessing.image_processor import preprocess_image

# ==============================================================================
# DAFTAR 30 MAHASISWA DUMMY
# ==============================================================================
DUMMY_STUDENTS = [
    ("Ahmad Zulkifli",       "20210001"),
    ("Budi Santoso",         "20210002"),
    ("Citra Dewi Rahayu",    "20210003"),
    ("Dian Permatasari",     "20210004"),
    ("Eko Prasetyo",         "20210005"),
    ("Fatimah Azzahra",      "20210006"),
    ("Galih Wicaksono",      "20210007"),
    ("Hana Septiani",        "20210008"),
    ("Irfan Maulana",        "20210009"),
    ("Jasmine Putri",        "20210010"),
    ("Kevin Adriansyah",     "20210011"),
    ("Lita Agustina",        "20210012"),
    ("Muhammad Rizky",       "20210013"),
    ("Nadia Salsabila",      "20210014"),
    ("Oki Firmansyah",       "20210015"),
    ("Putri Handayani",      "20210016"),
    ("Qodir Hakim",          "20210017"),
    ("Rini Wulandari",       "20210018"),
    ("Siti Nurhaliza",       "20210019"),
    ("Taufik Hidayat",       "20210020"),
    ("Ulfah Ramadhani",      "20210021"),
    ("Vino Kusuma",          "20210022"),
    ("Wahyu Saputra",        "20210023"),
    ("Xenia Maharani",       "20210024"),
    ("Yoga Pratama",         "20210025"),
    ("Zahra Aulia",          "20210026"),
    ("Andika Ramadhan",      "20210027"),
    ("Bella Oktaviani",      "20210028"),
    ("Cahyo Nugroho",        "20210029"),
    ("Desi Fitriani",        "20210030"),
]

SAMPLES_PER_STUDENT = 5   # Jumlah sampel gambar per mahasiswa


def add_noise(image: np.ndarray, intensity: float = 0.1) -> np.ndarray:
    """Menambahkan Gaussian noise ke gambar untuk variasi."""
    noise = np.random.normal(0, intensity * 255, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def add_texture(image: np.ndarray) -> np.ndarray:
    """Menambahkan tekstur kertas tipis (garis-garis samar)."""
    h, w = image.shape
    for y in range(0, h, 20):
        thickness = random.randint(1, 2)
        alpha = random.uniform(0.85, 0.97)
        image[y:y+thickness, :] = (image[y:y+thickness, :] * alpha).astype(np.uint8)
    return image


def generate_handwriting_image(
    name: str,
    nim: str,
    sample_idx: int,
    width: int = 256,
    height: int = 128,
) -> np.ndarray:
    """
    Menghasilkan gambar simulasi tulisan tangan untuk satu mahasiswa.

    Setiap mahasiswa memiliki karakteristik unik:
        - Teks: inisial + NIM + nomor sampel
        - Ketebalan garis (font thickness) berbeda-beda per mahasiswa
        - Kemiringan teks (italic effect)
        - Tingkat noise
    """
    # Background kertas (sedikit cream/warm, bukan putih murni)
    bg_value = random.randint(230, 250)
    canvas = np.full((height, width), bg_value, dtype=np.uint8)

    # Tambah tekstur kertas
    canvas = add_texture(canvas)

    # Tentukan karakteristik "gaya tulis" berdasarkan NIM (deterministik per mahasiswa)
    student_seed = int(nim) % 1000
    rng = np.random.RandomState(student_seed + sample_idx)

    # Parameter gaya tulisan (bervariasi per mahasiswa)
    font_scale = rng.uniform(0.4, 0.8)
    thickness  = rng.randint(1, 3)

    # Inisial nama + NIM sebagai representasi tulisan
    initials = "".join([w[0].upper() for w in name.split()[:3]])
    text_line1 = f"{initials}-{nim[-4:]}"
    text_line2 = f"S{sample_idx+1:02d}"

    # Posisi acak sedikit untuk variasi
    x_off = rng.randint(5, 30)
    y_off = rng.randint(30, 60)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Tulis teks pertama
    cv2.putText(canvas, text_line1, (x_off, y_off),
                font, font_scale, 0, thickness, cv2.LINE_AA)

    # Tulis teks kedua (lebih kecil)
    cv2.putText(canvas, text_line2, (x_off + 10, y_off + 35),
                font, font_scale * 0.7, 30, max(1, thickness - 1), cv2.LINE_AA)

    # Simulasi goresan tangan (beberapa garis pendek acak)
    n_strokes = rng.randint(2, 6)
    for _ in range(n_strokes):
        x1 = rng.randint(5, width - 30)
        y1 = rng.randint(10, height - 20)
        x2 = x1 + rng.randint(10, 60)
        y2 = y1 + rng.randint(-10, 10)
        stroke_thickness = rng.randint(1, 2)
        cv2.line(canvas, (x1, y1), (x2, y2), 10, stroke_thickness, cv2.LINE_AA)

    # Tambah noise ringan
    noise_level = rng.uniform(0.03, 0.12)
    canvas = add_noise(canvas, noise_level)

    # Blur ringan untuk efek "tidak tajam seperti printer"
    if rng.random() > 0.4:
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    return canvas


def generate_and_register_dataset():
    """
    Menghasilkan gambar dummy untuk semua mahasiswa dan mendaftarkannya ke database.
    """
    init_db()
    os.makedirs(DATASET_RAW_DIR, exist_ok=True)
    os.makedirs(DATASET_PROCESSED_DIR, exist_ok=True)

    total_generated = 0
    total_failed    = 0
    conn = get_connection()

    print("=" * 60)
    print("  GENERATE DUMMY DATASET")
    print(f"  {len(DUMMY_STUDENTS)} mahasiswa × {SAMPLES_PER_STUDENT} sampel")
    print("=" * 60)

    for student_name, nim in DUMMY_STUDENTS:
        for i in range(SAMPLES_PER_STUDENT):
            mata_kuliah = random.choice(MATA_KULIAH_OPTIONS)

            # Generate gambar
            img = generate_handwriting_image(student_name, nim, i)

            # Simpan file
            unique_id = uuid.uuid4().hex[:8]
            filename  = f"dummy_{nim}_{i}_{unique_id}.png"
            raw_path  = os.path.join(DATASET_RAW_DIR, filename)
            cv2.imwrite(raw_path, img)

            # Preprocessing & simpan
            processed_path = None
            is_processed   = 0
            try:
                proc_filename  = f"proc_{filename}"
                processed_path = os.path.join(DATASET_PROCESSED_DIR, proc_filename)
                preprocess_image(raw_path, save_path=processed_path)
                is_processed = 1
            except Exception as e:
                print(f"  [WARN] Preprocessing gagal [{student_name}-{i}]: {e}")

            # Daftar ke database
            timestamp = (datetime.now() - timedelta(days=random.randint(0, 60))).isoformat()
            try:
                conn.execute("""
                    INSERT INTO dataset (
                        student_name, student_id, mata_kuliah,
                        original_filename, saved_filename, file_path,
                        processed_path, is_processed, upload_timestamp, notes
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (
                    student_name, nim, mata_kuliah,
                    filename, filename, raw_path,
                    processed_path, is_processed, timestamp,
                    "Data sintetis — generated dummy",
                ))
                total_generated += 1
            except Exception as e:
                print(f"  [ERR] Gagal simpan ke DB [{student_name}-{i}]: {e}")
                total_failed += 1

        print(f"  [OK] {student_name:<30} ({SAMPLES_PER_STUDENT} sampel)")

    conn.commit()
    conn.close()

    print("=" * 60)
    print(f"  Total berhasil : {total_generated}")
    print(f"  Total gagal    : {total_failed}")
    print("=" * 60)
    print("  Dataset dummy siap! Jalankan training dari UI atau API.")
    print("=" * 60)


if __name__ == "__main__":
    generate_and_register_dataset()
