# 🌸 Yaevia — Sistem Verifikasi Keaslian Tulisan Tangan (HOG + KNN)
========================================================================
Skripsi Program Studi Pendidikan Teknik Informatika

Sistem otomatis berbasis pengolahan citra digital untuk verifikasi keaslian tulisan tangan mahasiswa pada lembar praktikum menggunakan metode **Histogram of Oriented Gradients (HOG)** untuk ekstraksi fitur dan **K-Nearest Neighbor (KNN)** berbasis **Euclidean Distance** untuk klasifikasi dan perhitungan skor kemiripan.

---

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
cd backend
python -m pip install -r requirements.txt
```

### 2. Jalankan Server Flask
```bash
python app.py
```
Server akan berjalan di: `http://localhost:5000`

### 3. Akses Antarmuka Web (Frontend)
Buka browser dan akses:
👉 **`http://localhost:5000/`**

---

## 📁 Struktur Folder
```
Yaevia/
├── backend/
│   ├── app.py                    ← Entry point Flask & file server
│   ├── config.py                 ← Hyperparameter HOG, KNN, & threshold similarity
│   ├── database.py               ← SQLite schema & auto-migration
│   ├── requirements.txt          ← Daftar library Python
│   ├── generate_dummy_data.py    ← Generator data dummy untuk simulasi
│   ├── preprocessing/
│   │   └── image_processor.py   ← Grayscale → Otsu Thresholding → Noise Removal → ROI
│   ├── features/
│   │   └── hog_extractor.py     ← Ekstraksi fitur HOG
│   ├── model/
│   │   ├── trainer.py           ← 9-step training pipeline (80:20 split, Precision/Recall/F1)
│   │   └── classifier.py        ← Klasifikasi & Euclidean Distance similarity
│   ├── api/
│   │   ├── dataset_routes.py    ← /api/dataset/* (upload, summary, list, stats)
│   │   ├── train_routes.py      ← /api/train/* & /api/model/info
│   │   ├── verify_routes.py     ← /api/verify/* & /api/verify/history
│   │   └── evaluate_routes.py   ← /api/evaluate (Confusion Matrix, chart dataset)
│   └── tests/
│       ├── test_blackbox.py     ← Pengujian Black Box sistem
│       └── evaluate_model.py    ← Evaluasi metrik & confusion matrix generator
│
└── frontend/
    ├── index.html               ← Dashboard utama
    ├── upload.html              ← Upload dataset & training model
    ├── verify.html              ← Verifikasi tulisan tangan
    ├── evaluate.html            ← Dashboard evaluasi, akurasi, & confusion matrix
    ├── history.html             ← Riwayat verifikasi & export CSV
    ├── css/style.css            ← Desain UI Yae Miko Theme (Pink, Gold, Soft Cream)
    ├── js/                      ← Logika interaktif per halaman
    └── images/icons/            ← Asset icon & loading PNG Yaevia
```

---

## 🔧 Workflow Penggunaan

### Alur Kerja Sistem:
1. **Upload Dataset** (`upload.html`): Unggah minimal 2 sampel tulisan tangan untuk setiap mahasiswa yang akan didaftarkan.
2. **Training Model** (`upload.html`): Latih model KNN dengan ekstraksi fitur HOG (otomatis membagi data 80% train, 20% test).
3. **Verifikasi** (`verify.html`): Unggah foto/scan tulisan tangan uji untuk mengidentifikasi mahasiswa dan mendapatkan skor kemiripan berbasis jarak Euclidean (*Euclidean Distance*).
4. **Evaluasi** (`evaluate.html`): Lihat metrik performa (*Accuracy*, *Precision*, *Recall*, *F1-Score*) dan *Confusion Matrix*.
5. **Riwayat** (`history.html`): Pantau seluruh histori verifikasi dan ekspor laporan ke format CSV.

---

## 📐 Formula Kemiripan (*Similarity Score*) & Keputusan Identifikasi

Sistem menghitung kemiripan geometris menggunakan ruang fitur HOG ternormalisasi L2-Hys ($N_{\text{blocks}} = 225, d_{\text{max}}^2 = 450.0$):

$$\text{Similarity (\%)} = \max\left(0.0, \min\left(100.0, \left(1 - \frac{d^2}{450.0}\right) \times 100\right)\right)$$

### 1. Kategori Status Kemiripan Geometris (*Similarity Status*):
| Skor Kemiripan | Batas Jarak ($d$) | Status Kemiripan |
|---|---|---|
| $\ge 65\%$ | $d \le 12.55$ | 🟢 **SANGAT MIRIP** |
| $\ge 50\%$ | $d \le 15.00$ | 🟡 **MIRIP** |
| $\ge 40\%$ | $d \le 16.43$ | 🟠 **KURANG MIRIP** |
| $< 40\%$ | $d > 16.43$ | 🔴 **TIDAK MIRIP** |

### 2. Status Keputusan Verifikasi (*Verification Status*):
| Status Verifikasi | Kriteria Evaluasi | Keterangan |
|---|---|---|
| 🟢 **TERIDENTIFIKASI** | $\text{Similarity} \ge 50\%$ dan $\text{Vote Share} \ge 35\%$ | Identitas penulis terverifikasi kuat |
| 🟡 **TIDAK PASTI** | $40\% \le \text{Similarity} < 50\%$ atau voting terbagi | Memerlukan konfirmasi manual |
| 🔴 **TIDAK TERIDENTIFIKASI** | $\text{Similarity} < 40\%$ | Penulis tidak cocok / berbeda |

---

## ⚙️ Tuning Parameter (Eksperimen Skripsi)

Semua hyperparameter dapat dikonfigurasi langsung pada `backend/config.py`:

```python
# KNN Hyperparameters
KNN_N_NEIGHBORS = 5            # Nilai K (baseline: 5; perbandingan: 3, 5, 7, 9)
KNN_METRIC      = "euclidean"  # Jarak Euclidean
KNN_WEIGHTS     = "distance"   # Weighted voting (1/d)

# HOG Hyperparameters (Dalal & Triggs, 2005)
HOG_ORIENTATIONS    = 9        # Jumlah orientasi gradient
HOG_PIXELS_PER_CELL = (8, 8)   # Ukuran cell dalam pixel
HOG_CELLS_PER_BLOCK = (2, 2)   # Ukuran blok dalam cell (225 blok = 8100 dimensi)
HOG_BLOCK_NORM      = "L2-Hys"

# Train/Test Split
TEST_SIZE           = 0.2      # 80:20 Train-Test Split (stratified)
RANDOM_STATE        = 42
```

---

## 🧪 Menjalankan Pengujian

```bash
cd backend

# Black Box Testing
python -m pytest tests/test_blackbox.py -v

# Evaluasi Model Mandiri
python tests/evaluate_model.py
```

---

## 📊 Ringkasan REST API Endpoints

| Method | Endpoint | Deskripsi |
|---|---|---|
| `GET` | `/api/health` | Status server & versi |
| `GET` | `/api/config` | Konfigurasi aktif HOG & KNN |
| `POST` | `/api/dataset/upload` | Upload citra tulisan tangan & label |
| `GET` | `/api/dataset/summary` | Ringkasan dataset & per-mahasiswa |
| `GET` | `/api/dataset/stats` | Statistik total dataset & mahasiswa |
| `DELETE` | `/api/dataset/<id>` | Hapus satu data dataset |
| `DELETE` | `/api/dataset/all` | Reset seluruh dataset |
| `GET` | `/api/train/validate` | Validasi kelayakan dataset sebelum training |
| `POST` | `/api/train` | Proses ekstraksi HOG & training model KNN |
| `GET` | `/api/model/info` | Metadata lengkap model aktif |
| `POST` | `/api/verify` | Ekstraksi HOG & verifikasi citra input |
| `GET` | `/api/verify/history` | Riwayat log verifikasi dengan pagination |
| `GET` | `/api/evaluate` | Data metrik performa & Confusion Matrix |
| `GET` | `/api/evaluate/compare-k` | Evaluasi perbandingan nilai K (K=3, 5, 7, 9) |

---

## 📚 Referensi Ilmiah

- **Dalal, N., & Triggs, B. (2005).** *Histograms of Oriented Gradients for Human Detection*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
- **Otsu, N. (1979).** *A Threshold Selection Method from Gray-Level Histograms*. IEEE Transactions on Systems, Man, and Cybernetics.
- **Cover, T., & Hart, P. (1967).** *Nearest Neighbor Pattern Classification*. IEEE Transactions on Information Theory.
- **Gonzalez, R. C., & Woods, R. E. (2018).** *Digital Image Processing*, 4th Edition. Pearson.

