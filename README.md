# Sistem Verifikasi Keaslian Tulisan Tangan (HOG + KNN)
# ======================================================
# Skripsi Program Studi Pendidikan Teknik Informatika

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
cd D:\handwriting-verification\backend
python -m pip install -r requirements.txt
```

### 2. Jalankan Server Flask
```bash
python app.py
```
Server akan berjalan di: `http://localhost:5000`

### 3. Buka Frontend
Buka file `D:\handwriting-verification\frontend\index.html` di browser.

---

## 📁 Struktur Folder
```
handwriting-verification/
├── backend/
│   ├── app.py                    ← Entry point Flask
│   ├── config.py                 ← Parameter HOG & KNN (tuning di sini)
│   ├── database.py               ← SQLite setup
│   ├── requirements.txt
│   ├── generate_dummy_data.py    ← Generator 150 gambar dummy
│   ├── preprocessing/
│   │   └── image_processor.py   ← Grayscale→Otsu→Noise removal→ROI
│   ├── features/
│   │   └── hog_extractor.py     ← Ekstraksi fitur HOG
│   ├── model/
│   │   ├── trainer.py           ← Training pipeline KNN
│   │   └── classifier.py        ← Prediksi + similarity score
│   ├── api/
│   │   ├── dataset_routes.py    ← /api/dataset/*
│   │   ├── train_routes.py      ← /api/train/*
│   │   └── verify_routes.py     ← /api/verify/*
│   └── tests/
│       ├── test_blackbox.py     ← 13 skenario black box test
│       └── evaluate_model.py    ← Akurasi, confusion matrix, CSV
│
└── frontend/
    ├── index.html               ← Dashboard
    ├── upload.html              ← Upload & Training
    ├── verify.html              ← Verifikasi
    ├── history.html             ← Riwayat
    ├── css/style.css            ← Tema Soft Pink & Gold
    └── js/                     ← Logic per halaman
```

---

## 🔧 Workflow Penggunaan

### Untuk Demo/Skripsi (dengan data dummy):
1. `python generate_dummy_data.py` — generate 150 gambar sintetis
2. Buka `upload.html` → klik **Mulai Training**
3. Buka `verify.html` → upload gambar → klik **Verifikasi**

### Untuk Dataset Asli (30 responden):
1. Buka `upload.html`
2. Upload setiap citra tulisan tangan dengan nama + mata kuliah
3. Klik **Mulai Training** setelah semua data terupload
4. Gunakan `verify.html` untuk verifikasi

---

## ⚙️ Tuning Parameter (Eksperimen Skripsi)

Edit `backend/config.py`:

```python
# KNN
KNN_N_NEIGHBORS = 5        # Coba: 3, 5, 7, 9, 11
KNN_METRIC      = "euclidean"  # Coba: "manhattan", "minkowski"

# HOG
HOG_ORIENTATIONS    = 9    # Coba: 6, 9, 12
HOG_PIXELS_PER_CELL = (8,8)  # Coba: (4,4), (8,8), (16,16)
HOG_CELLS_PER_BLOCK = (2,2)  # Coba: (2,2), (3,3)
```

---

## 🧪 Menjalankan Tests

```bash
cd D:\handwriting-verification\backend

# Black Box Testing
python -m pytest tests/test_blackbox.py -v

# Evaluasi Model (akurasi, confusion matrix, CSV)
python tests/evaluate_model.py
```

---

## 📊 API Endpoints

| Method | URL | Fungsi |
|--------|-----|--------|
| GET | `/api/health` | Status server |
| GET | `/api/config` | Parameter HOG & KNN |
| POST | `/api/dataset/upload` | Upload citra |
| GET | `/api/dataset/list` | Daftar dataset |
| GET | `/api/dataset/stats` | Statistik dataset |
| DELETE | `/api/dataset/<id>` | Hapus satu data |
| POST | `/api/train` | Mulai training |
| GET | `/api/train/status` | Status model |
| POST | `/api/verify` | Verifikasi gambar baru |
| GET | `/api/verify/history` | Riwayat verifikasi |

---

## 📚 Referensi Ilmiah

- Dalal, N., & Triggs, B. (2005). *Histograms of Oriented Gradients for Human Detection*. IEEE CVPR.
- Otsu, N. (1979). *A threshold selection method from gray-level histograms*. IEEE Trans. Systems, Man, Cybernetics.
- Cover, T., & Hart, P. (1967). *Nearest neighbor pattern classification*. IEEE Trans. Information Theory.
- Gonzalez, R.C. & Woods, R.E. (2018). *Digital Image Processing*, 4th Ed.
