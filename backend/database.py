"""
database.py — Inisialisasi & Manajemen Database SQLite
========================================================
Skema yang digunakan:
  1. dataset       — metadata citra tulisan tangan yang diupload
  2. verifications — riwayat hasil verifikasi/klasifikasi
  3. model_meta    — metadata model terlatih (diperluas untuk skripsi)

BAB IV — Implementasi:
    SQLite dipilih karena tidak memerlukan server eksternal,
    sesuai untuk deployment lokal penelitian skripsi ini.
"""

import sqlite3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DB_PATH


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """
    Membuat semua tabel yang diperlukan jika belum ada.
    Juga menjalankan migrasi kolom baru pada tabel yang sudah ada.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ----------------------------------------------------------------
    # Tabel 1: dataset
    # ----------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dataset (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name      TEXT NOT NULL,
            student_id        TEXT,
            mata_kuliah       TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            saved_filename    TEXT NOT NULL UNIQUE,
            file_path         TEXT NOT NULL,
            processed_path    TEXT,
            is_processed      INTEGER DEFAULT 0,
            upload_timestamp  TEXT NOT NULL,
            notes             TEXT
        )
    """)

    # ----------------------------------------------------------------
    # Tabel 2: verifications (diperluas)
    # ----------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS verifications (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            query_filename        TEXT NOT NULL,
            query_path            TEXT NOT NULL DEFAULT '',
            predicted_name        TEXT,
            predicted_student_id  TEXT,
            similarity_percent    REAL,
            euclidean_distance    REAL,
            verification_status   TEXT,
            similarity_status     TEXT,
            top_matches_json      TEXT,
            model_version         TEXT,
            feature_vector_length INTEGER,
            knn_k                 INTEGER,
            analysis_time         REAL,
            verification_timestamp TEXT NOT NULL
        )
    """)

    # ----------------------------------------------------------------
    # Tabel 3: model_meta (diperluas untuk metadata skripsi)
    # ----------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_meta (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            model_filename        TEXT NOT NULL,
            model_path            TEXT NOT NULL,
            train_accuracy        REAL,
            test_accuracy         REAL,
            precision_score       REAL,
            recall_score          REAL,
            f1_score              REAL,
            n_train_samples       INTEGER,
            n_test_samples        INTEGER,
            n_classes             INTEGER,
            n_total_dataset       INTEGER,
            knn_k                 INTEGER,
            knn_metric            TEXT,
            knn_weights           TEXT,
            hog_orientations      INTEGER,
            hog_pixels_per_cell   TEXT,
            hog_cells_per_block   TEXT,
            hog_block_norm        TEXT,
            feature_vector_size   INTEGER,
            test_size             REAL,
            training_time_seconds REAL,
            metadata_json_path    TEXT,
            train_timestamp       TEXT NOT NULL,
            is_active             INTEGER DEFAULT 1
        )
    """)

    conn.commit()

    # ----------------------------------------------------------------
    # Migrasi: tambahkan kolom baru jika belum ada (safe migration)
    # ----------------------------------------------------------------
    _migrate_columns(cursor, conn)

    conn.close()
    print("[DB] Database berhasil diinisialisasi:", DB_PATH)


def _migrate_columns(cursor, conn):
    """Menambahkan kolom baru pada tabel existing (safe, idempotent)."""
    # Kolom baru untuk verifications
    verif_new_cols = [
        ("euclidean_distance",    "REAL"),
        ("similarity_status",     "TEXT"),
        ("top_matches_json",      "TEXT"),
        ("feature_vector_length", "INTEGER"),
        ("knn_k",                 "INTEGER"),
        ("analysis_time",         "REAL"),
        ("query_path",            "TEXT NOT NULL DEFAULT ''"),
    ]

    # Kolom baru untuk model_meta
    meta_new_cols = [
        ("precision_score",       "REAL"),
        ("recall_score",          "REAL"),
        ("f1_score",              "REAL"),
        ("n_total_dataset",       "INTEGER"),
        ("knn_metric",            "TEXT"),
        ("knn_weights",           "TEXT"),
        ("hog_block_norm",        "TEXT"),
        ("test_size",             "REAL"),
        ("training_time_seconds", "REAL"),
        ("metadata_json_path",    "TEXT"),
    ]

    def get_columns(table):
        rows = cursor.execute(f"PRAGMA table_info({table})").fetchall()
        return {r[1] for r in rows}

    verif_cols = get_columns("verifications")
    for col_name, col_type in verif_new_cols:
        if col_name not in verif_cols:
            try:
                cursor.execute(f"ALTER TABLE verifications ADD COLUMN {col_name} {col_type}")
                print(f"[DB] Migrasi: tambah kolom verifications.{col_name}")
            except Exception as e:
                print(f"[DB] Migrasi skip {col_name}: {e}")

    meta_cols = get_columns("model_meta")
    for col_name, col_type in meta_new_cols:
        if col_name not in meta_cols:
            try:
                cursor.execute(f"ALTER TABLE model_meta ADD COLUMN {col_name} {col_type}")
                print(f"[DB] Migrasi: tambah kolom model_meta.{col_name}")
            except Exception as e:
                print(f"[DB] Migrasi skip {col_name}: {e}")

    conn.commit()


def get_active_model_meta():
    """Mengambil metadata model yang sedang aktif."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM model_meta WHERE is_active=1 ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def deactivate_all_models():
    """Menonaktifkan semua model sebelum menyimpan model baru."""
    conn = get_connection()
    conn.execute("UPDATE model_meta SET is_active=0")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
