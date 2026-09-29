"""
tests/experiment_g_knn_final.py — Experiment G: Final KNN Sensitivity & Model Freezing
========================================================================================
Evaluates K in [1, 3, 5, 7, 9] on the locked 256x256 HOG (8x8 cells, 34,596-dim) representation
across all 360 images (18 students x 20 samples) under strict LOOCV.

Feature Extraction is performed ONCE to produce the 360x34596 feature matrix and 360x360
pairwise Euclidean distance matrix, ensuring perfect consistency across all K values.

Outputs saved in: backend/tests/evaluation_results/knn_final_experiment/
"""

import os
import sys
import json
import csv
import time
import sqlite3
import numpy as np
import cv2
from collections import defaultdict
from skimage.feature import hog

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import (
    DB_PATH, DATASET_RAW_DIR,
    HOG_ORIENTATIONS, HOG_CELLS_PER_BLOCK, HOG_BLOCK_NORM,
)
from preprocessing.image_processor import (
    normalize_orientation, convert_to_grayscale,
    apply_gaussian_blur, apply_otsu_threshold, remove_noise,
    extract_roi, resize_with_aspect_ratio,
)

SIZE_256 = (256, 256)
PPC = (8, 8)
K_VALUES = [1, 3, 5, 7, 9]
EPS = 1e-7

OUT_DIR = os.path.join(BASE_DIR, "tests", "evaluation_results", "knn_final_experiment")
os.makedirs(OUT_DIR, exist_ok=True)


def extract_hog_256(image: np.ndarray):
    img_f = image.astype(np.float64) / 255.0 if image.dtype != np.float64 else image
    feat = hog(
        img_f,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=PPC,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        visualize=False,
        feature_vector=True,
    )
    return feat


def pairwise_dist(X):
    dot = np.dot(X, X.T)
    sq = np.diag(dot)
    dsq = np.maximum(0.0, sq[:, None] + sq[None, :] - 2 * dot)
    d = np.sqrt(dsq)
    np.fill_diagonal(d, 0.0)
    return d


def evaluate_loocv_for_k(dist_matrix, y, k):
    n = len(y)
    y_a = np.array(y)
    unique_classes = sorted(set(y))
    results = []

    for i in range(n):
        true_label = y_a[i]
        # Strictly exclude query sample itself (self-exclusion)
        row = np.delete(dist_matrix[i], i)
        lbls = np.delete(y_a, i)

        sorted_idx = np.argsort(row)
        topk_idx = sorted_idx[:k]
        topk_dists = row[topk_idx]
        topk_labels = lbls[topk_idx]

        if k == 1:
            pred_label = topk_labels[0]
            pred_share = 100.0
            second_share = 0.0
            margin = 100.0
            true_nbr = 1 if pred_label == true_label else 0
        else:
            wk = 1.0 / np.maximum(topk_dists, EPS)
            total_w = wk.sum()

            vote_w = defaultdict(float)
            vote_c = defaultdict(int)
            for d_val, lbl, wv in zip(topk_dists, topk_labels, wk):
                vote_w[lbl] += wv
                vote_c[lbl] += 1

            sorted_votes = sorted(vote_w.items(), key=lambda x: -x[1])
            pred_label = sorted_votes[0][0]
            pred_share = sorted_votes[0][1] / total_w * 100.0
            second_share = sorted_votes[1][1] / total_w * 100.0 if len(sorted_votes) > 1 else 0.0
            margin = pred_share - second_share
            true_nbr = vote_c.get(true_label, 0)

        results.append({
            "sample_idx": i,
            "true_label": true_label,
            "pred_label": pred_label,
            "is_correct": bool(pred_label == true_label),
            "pred_vote_share": round(float(pred_share), 2),
            "margin": round(float(margin), 2),
            "true_neighbors": true_nbr,
            "nearest_dist": round(float(topk_dists[0]), 4),
        })

    return results


def calc_classification_metrics(loocv_results, unique_classes):
    """
    Computes overall accuracy, macro precision, macro recall, and macro F1 score.
    """
    cls2i = {c: i for i, c in enumerate(unique_classes)}
    n_classes = len(unique_classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for r in loocv_results:
        cm[cls2i[r["true_label"]], cls2i[r["pred_label"]]] += 1

    total_correct = int(np.trace(cm))
    total_samples = len(loocv_results)
    overall_acc = (total_correct / total_samples) * 100.0

    precisions = []
    recalls = []
    f1s = []

    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_precision = float(np.mean(precisions)) * 100.0
    macro_recall = float(np.mean(recalls)) * 100.0
    macro_f1 = float(np.mean(f1s)) * 100.0

    return {
        "overall_accuracy": round(overall_acc, 2),
        "macro_precision": round(macro_precision, 2),
        "macro_recall": round(macro_recall, 2),
        "macro_f1": round(macro_f1, 2),
        "total_correct": total_correct,
        "total_errors": total_samples - total_correct,
        "confusion_matrix": cm,
    }


def run_experiment_g():
    print("=" * 80)
    print("EXPERIMENT G: FINAL KNN SENSITIVITY (K in [1, 3, 5, 7, 9])")
    print("=" * 80)

    # 1. Load Dataset
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, student_name, original_filename, saved_filename, file_path "
        "FROM dataset ORDER BY student_name ASC, original_filename ASC, saved_filename ASC"
    ).fetchall()
    conn.close()

    students = sorted(list(set(r["student_name"] for r in rows)))
    n_records = len(rows)
    print(f"Loaded {n_records} records across {len(students)} students.")

    # 2. Extract Features ONCE at 256x256
    print("\nExtracting 256x256 HOG features (single pass)...")
    t0 = time.perf_counter()
    feat_list = []
    labels = []
    samples_meta = []

    for idx, r in enumerate(rows):
        student = r["student_name"]
        raw_name = r["saved_filename"] or os.path.basename(r["file_path"])
        raw_path = os.path.join(DATASET_RAW_DIR, raw_name)

        if not os.path.exists(raw_path):
            continue
        img_bgr = cv2.imread(raw_path)
        if img_bgr is None:
            continue

        oriented = normalize_orientation(img_bgr)
        gray = convert_to_grayscale(oriented)
        blurred = apply_gaussian_blur(gray)
        binary = apply_otsu_threshold(blurred)
        denoised = remove_noise(binary)
        roi = extract_roi(denoised)
        img256 = resize_with_aspect_ratio(roi, SIZE_256)

        feat = extract_hog_256(img256)
        feat_list.append(feat)
        labels.append(student)

        sample_num = len([s for s in samples_meta if s["student_name"] == student]) + 1
        samples_meta.append({
            "sample_idx": len(samples_meta),
            "id": r["id"],
            "student_name": student,
            "filename": r["original_filename"],
            "sample_num": sample_num,
        })

    t_extract = time.perf_counter() - t0
    X = np.array(feat_list)
    n = len(labels)
    print(f"Extraction complete: {X.shape} in {t_extract:.2f}s ({t_extract/n*1000:.2f}ms/image).")

    # 3. Compute Distance Matrix ONCE
    print("\nComputing pairwise Euclidean distance matrix...")
    t_dm0 = time.perf_counter()
    dist_matrix = pairwise_dist(X)
    t_dm = time.perf_counter() - t_dm0
    print(f"Distance matrix computed in {t_dm:.4f}s.")

    # 4. Evaluate LOOCV for K in [1, 3, 5, 7, 9]
    k_eval_results = {}
    k_metrics = {}
    k_times = {}

    p1_indices = [i for i, s in enumerate(samples_meta) if s["sample_num"] == 1]
    p16_indices = [i for i, s in enumerate(samples_meta) if s["sample_num"] == 16]
    prest_indices = [i for i, s in enumerate(samples_meta) if s["sample_num"] > 1]

    for k in K_VALUES:
        print(f"Evaluating K={k}...")
        tk0 = time.perf_counter()
        res = evaluate_loocv_for_k(dist_matrix, labels, k)
        tk_eval = time.perf_counter() - tk0

        metrics = calc_classification_metrics(res, students)
        k_eval_results[k] = res
        k_metrics[k] = metrics
        k_times[k] = tk_eval

    # 5. Position Accuracy by K (1..20)
    pos_rows = []
    for p in range(1, 21):
        s_idxs = [i for i, s in enumerate(samples_meta) if s["sample_num"] == p]
        row_dict = {"position": f"#{p:02d}", "pos_int": p}
        for k in K_VALUES:
            corr_k = sum(1 for i in s_idxs if k_eval_results[k][i]["is_correct"])
            row_dict[f"acc_k{k}"] = round(corr_k / len(s_idxs) * 100.0, 2)
        pos_rows.append(row_dict)

    # 6. Per-Student Accuracy by K
    student_rows = []
    for st in students:
        s_idxs = [i for i, s in enumerate(samples_meta) if s["student_name"] == st]
        st_dict = {"student": st}
        accs_st = {}
        for k in K_VALUES:
            corr_k = sum(1 for i in s_idxs if k_eval_results[k][i]["is_correct"])
            acc_val = round(corr_k / len(s_idxs) * 100.0, 2)
            st_dict[f"acc_k{k}"] = acc_val
            accs_st[k] = acc_val

        # Find best K for student
        max_a = max(accs_st.values())
        best_k_list = [f"K{k}" for k, v in accs_st.items() if v == max_a]
        st_dict["best_k_diagnostic"] = "/".join(best_k_list)
        student_rows.append(st_dict)

    # 7. Summary Table for K Values
    k_summary_rows = []
    for k in K_VALUES:
        m = k_metrics[k]
        res = k_eval_results[k]

        # Position subsets
        p1_corr = sum(1 for i in p1_indices if res[i]["is_correct"])
        p16_corr = sum(1 for i in p16_indices if res[i]["is_correct"])
        prest_corr = sum(1 for i in prest_indices if res[i]["is_correct"])

        p1_acc = round(p1_corr / len(p1_indices) * 100.0, 2)
        p16_acc = round(p16_corr / len(p16_indices) * 100.0, 2)
        prest_acc = round(prest_corr / len(prest_indices) * 100.0, 2)

        # Class stability stats
        st_accs = [r[f"acc_k{k}"] for r in student_rows]
        min_cls_acc = min(st_accs)
        max_cls_acc = max(st_accs)
        std_cls_acc = round(float(np.std(st_accs)), 2)
        count_ge50 = sum(1 for a in st_accs if a >= 50.0)
        count_ge70 = sum(1 for a in st_accs if a >= 70.0)
        count_lt40 = sum(1 for a in st_accs if a < 40.0)

        mean_vote = round(float(np.mean([r["pred_vote_share"] for r in res])), 2)
        mean_margin = round(float(np.mean([r["margin"] for r in res])), 2)

        k_summary_rows.append({
            "K": k,
            "overall_accuracy": m["overall_accuracy"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "macro_f1": m["macro_f1"],
            "total_correct": m["total_correct"],
            "total_errors": m["total_errors"],
            "pos1_acc": p1_acc,
            "pos16_acc": p16_acc,
            "pos2_20_acc": prest_acc,
            "min_class_acc": min_cls_acc,
            "max_class_acc": max_cls_acc,
            "std_class_acc": std_cls_acc,
            "students_ge50": count_ge50,
            "students_ge70": count_ge70,
            "students_lt40": count_lt40,
            "mean_vote": mean_vote,
            "mean_margin": mean_margin,
            "eval_time_s": round(k_times[k], 4),
        })

    # 8. Pairwise Confusion Analysis across K transitions (K1->K3, K3->K5, K5->K7, K7->K9)
    conf_analysis_rows = []
    for i in range(len(students)):
        for j in range(i + 1, len(students)):
            sa, sb = students[i], students[j]
            t_k1 = k_metrics[1]["confusion_matrix"][i, j] + k_metrics[1]["confusion_matrix"][j, i]
            t_k3 = k_metrics[3]["confusion_matrix"][i, j] + k_metrics[3]["confusion_matrix"][j, i]
            t_k5 = k_metrics[5]["confusion_matrix"][i, j] + k_metrics[5]["confusion_matrix"][j, i]
            t_k7 = k_metrics[7]["confusion_matrix"][i, j] + k_metrics[7]["confusion_matrix"][j, i]
            t_k9 = k_metrics[9]["confusion_matrix"][i, j] + k_metrics[9]["confusion_matrix"][j, i]

            if any([t_k1 > 0, t_k3 > 0, t_k5 > 0, t_k7 > 0, t_k9 > 0]):
                conf_analysis_rows.append({
                    "class_a": sa,
                    "class_b": sb,
                    "total_k1": int(t_k1),
                    "total_k3": int(t_k3),
                    "total_k5": int(t_k5),
                    "total_k7": int(t_k7),
                    "total_k9": int(t_k9),
                    "delta_k5_vs_k1": int(t_k5 - t_k1),
                    "delta_k5_vs_k3": int(t_k5 - t_k3),
                })

    conf_analysis_rows.sort(key=lambda x: -x["total_k5"])

    # 9. Known Diagnostic Cases Summary
    def get_case(name):
        return [r for r in student_rows if r["student"] == name][0]

    case_dinnar = get_case("Febrian Dinnar Purnama")
    case_fahim = get_case("Fahim J Mujaddid")
    case_brama = get_case("Bramasetya Raka Purnama")
    case_dimas = get_case("Dimas Wahyu Prasetyo")
    case_rakha = get_case("Rakha Burhannudin Majid")

    # 10. Determine Final K Selection
    # Compare K=3 vs K=5:
    # If K=5 has highest accuracy & macro-F1, select K=5. If K=3 is equal, compare class stability.
    best_k_row = max(k_summary_rows, key=lambda x: (x["overall_accuracy"], x["macro_f1"], -x["std_class_acc"]))
    selected_k = best_k_row["K"]

    # 11. Save CSVs and JSONs
    print("\nSaving Experiment G outputs...")

    # k_summary.csv
    with open(os.path.join(OUT_DIR, "k_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(k_summary_rows[0].keys()))
        w.writeheader(); w.writerows(k_summary_rows)

    # per_student_by_k.csv
    with open(os.path.join(OUT_DIR, "per_student_by_k.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(student_rows[0].keys()))
        w.writeheader(); w.writerows(student_rows)

    # position_by_k.csv
    with open(os.path.join(OUT_DIR, "position_by_k.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pos_rows[0].keys()))
        w.writeheader(); w.writerows(pos_rows)

    # confusion_k*.csv
    for k in K_VALUES:
        with open(os.path.join(OUT_DIR, f"confusion_k{k}.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["actual \\ pred"] + students)
            for i, st in enumerate(students):
                writer.writerow([st] + list(k_metrics[k]["confusion_matrix"][i]))

    # confusion_analysis.csv
    with open(os.path.join(OUT_DIR, "confusion_analysis.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(conf_analysis_rows[0].keys()))
        w.writeheader(); w.writerows(conf_analysis_rows)

    # final_configuration.json
    final_config = {
        "model_architecture": "HOG + K-Nearest Neighbor (KNN)",
        "image_preprocessing": {
            "orientation_normalization": "Auto-rotate landscape to portrait (90 deg clockwise)",
            "grayscale_conversion": "cv2.COLOR_BGR2GRAY",
            "gaussian_blur_kernel": [5, 5],
            "binarization_method": "Otsu Adaptive Thresholding (cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)",
            "noise_removal": "Median blur (k=3) + Morphological Close & Open (3x3 rect)",
            "roi_segmentation": "Baseline extract_roi (bounding box of all external contours + proportional padding)",
            "resize_method": "Aspect-Ratio Preserving Resize (Letterboxing to target canvas)",
            "target_image_size": [256, 256],
        },
        "hog_feature_extractor": {
            "orientations": 9,
            "pixels_per_cell": [8, 8],
            "cells_per_block": [2, 2],
            "block_norm": "L2-Hys",
            "feature_vector_dimension": 34596,
        },
        "knn_classifier": {
            "n_neighbors": selected_k,
            "metric": "euclidean",
            "weights": "distance" if selected_k > 1 else "uniform",
            "algorithm": "auto",
        },
        "benchmark_results_on_development_dataset": {
            "dataset_size": "360 samples across 18 students (20 samples per student)",
            "evaluation_methodology": "Strict Leave-One-Out Cross-Validation (LOOCV)",
            "loocv_accuracy": f"{best_k_row['overall_accuracy']}% ({best_k_row['total_correct']}/360)",
            "macro_precision": f"{best_k_row['macro_precision']}%",
            "macro_recall": f"{best_k_row['macro_recall']}%",
            "macro_f1": f"{best_k_row['macro_f1']}%",
            "sample_position_1_accuracy": f"{best_k_row['pos1_acc']}%",
            "sample_position_16_accuracy": f"{best_k_row['pos16_acc']}%",
            "sample_positions_2_to_20_accuracy": f"{best_k_row['pos2_20_acc']}%",
            "students_with_accuracy_ge_50_pct": f"{best_k_row['students_ge50']}/18",
            "students_with_accuracy_ge_70_pct": f"{best_k_row['students_ge70']}/18",
            "students_with_accuracy_lt_40_pct": f"{best_k_row['students_lt40']}/18",
        },
    }

    with open(os.path.join(OUT_DIR, "final_configuration.json"), "w", encoding="utf-8") as f:
        json.dump(final_config, f, indent=2)

    # experiment_g_summary.json
    master_summary = {
        "experiment": "G — Final KNN Sensitivity & Model Freezing",
        "selected_k": selected_k,
        "k_summary": k_summary_rows,
        "diagnostic_cases": {
            "dinnar": case_dinnar,
            "fahim": case_fahim,
            "brama": case_brama,
            "dimas": case_dimas,
            "rakha": case_rakha,
        },
        "final_configuration": final_config,
    }

    with open(os.path.join(OUT_DIR, "experiment_g_summary.json"), "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    # 12. Generate external_holdout_protocol.md
    holdout_protocol_content = f"""# Protokol Pengujian External Holdout Dataset (Frozen Verification Protocol)

> **Status: PROTOKOL TERKUNCI (FROZEN) — SIAP UNTUK EVALUASI FINAL.**
> Protokol ini mendefinisikan aturan ketat pengujian data foto tulisan tangan baru yang sengaja disimpan di luar (*holdout/unseen*) 20 sampel per mahasiswa.

---

## 1. Konfigurasi Sistem Terkunci (Zero-Leakage Freeze)

Seluruh komponen pipeline berikut **DIKUNCI SECARA PERMANEN** dan TIDAK BOLEH diubah saat maupun setelah pengujian external data:

| Komponen | Konfigurasi Terkunci |
|---|---|
| **Resolusi Citra Input** | **256 x 256 piksel** (Aspect-Ratio Preserving Letterbox) |
| **Pipeline Preprocessing** | Auto-Orientation (Portrait) -> Grayscale -> Gaussian Blur (5x5) -> Otsu Inverted Binarization -> Noise Removal (Median 3 + Morph Close/Open 3x3) -> Baseline extract_roi() |
| **HOG Descriptor** | Orientations = 9, Pixels per Cell = (8, 8), Cells per Block = (2, 2), Normalisasi Blok = L2-Hys |
| **Dimensi Fitur HOG** | **34,596 fitur** (float64) |
| **K-Nearest Neighbor** | **K = {selected_k}**, Metrik Jarak = **Euclidean**, Pembobotan = **weights="distance"** |
| **Koleksi Data Latih** | **360 sampel data latih** (18 mahasiswa x 20 sampel) |
| **Formula Similarity** | Similarity (%) = max(0, min(100, (1 - (d^2 / 450)) * 100)) |

---

## 2. Aturan Pencegahan Kebocoran Data (Data Leakage Rules)

1. **Citra external holdout TIDAK BOLEH ditambahkan ke database training.**
2. **Citra external holdout TIDAK BOLEH digunakan untuk memilih atau mengubah parameter HOG/KNN.**
3. **Pipeline preprocessing TIDAK BOLEH dimodifikasi setelah melihat hasil klasifikasi holdout.**
4. Evaluasi external holdout harus murni merupakan pengujian *single-pass inferensi* (*blind testing*).

---

## 3. Format Pencatatan Data Uji per Sampel (Audit Log Record)

Untuk setiap citra uji external yang diverifikasi, catat variabel berikut ke dalam berkas `external_holdout_results.csv`:

| Field Name | Tipe Data | Deskripsi |
|---|---|---|
| `sample_id` | Integer / String | ID unik file uji |
| `original_filename` | String | Nama file foto asli |
| `ground_truth_name` | String | Nama mahasiswa sebenarnya (*ground truth label*) |
| `ground_truth_nim` | String | NIM mahasiswa sebenarnya |
| `predicted_name` | String | Kelas hasil prediksi K={selected_k} KNN |
| `is_correct` | Boolean | True jika predicted_name == ground_truth_name, False jika salah |
| `nearest_distance` | Float | Jarak Euclidean ke tetangga terdekat peringkat #1 |
| `similarity_percent` | Float | Persentase kemiripan geometris hasil formula |
| `vote_share_percent` | Float | Persentase bobot suara kandidat pemenang |
| `top5_candidates` | String/JSON | Daftar 5 tetangga terdekat beserta jarak & similarity |
| `top5_contains_truth` | Boolean | True jika ground truth ada di dalam Top-5 candidates |
| `inference_time_ms` | Float | Total waktu preprocessing + HOG + KNN (milidetik) |

---

## 4. Metrik Evaluasi Akhir yang Wajib Dilaporkan

1. **Overall Accuracy (%)**: (Jumlah Benar / Total Sampel Holdout) * 100%
2. **Top-5 Accuracy (%)**: Persentase sampel di mana identitas asli berada di dalam Top-5 kandidat teratas.
3. **Per-Student Accuracy (%)**: Akurasi per masing-masing mahasiswa pada data holdout.
4. **Macro Precision, Recall, dan F1-Score (%)**.
5. **Full Confusion Matrix 18 x 18**.
6. **Rata-rata Similarity untuk Prediksi Benar vs Prediksi Salah**: Mengukur separabilitas keyakinan model.
"""

    with open(os.path.join(OUT_DIR, "external_holdout_protocol.md"), "w", encoding="utf-8") as f:
        f.write(holdout_protocol_content)

    # 13. Console Output
    print("\n" + "=" * 80)
    print("EXPERIMENT G — FINAL KNN DECISION")
    print("=" * 80)
    print(f"{'K':<5} {'Accuracy':<12} {'Macro-F1':<12} {'Correct/360':<14} {'Pos1 Acc':<10} {'Class Std':<10}")
    print("-" * 65)
    for r in k_summary_rows:
        sel_mark = " (SELECTED)" if r["K"] == selected_k else ""
        print(f"{r['K']:<5} {r['overall_accuracy']:.2f}%{'':<5} {r['macro_f1']:.2f}%{'':<5} {r['total_correct']}/360{'':<6} {r['pos1_acc']:.1f}%{'':<4} {r['std_class_acc']:.2f}%{sel_mark}")

    print("\n" + "=" * 80)
    print(f"SELECTED FINAL K     : K = {selected_k}")
    print(f"FINAL CONFIGURATION  : 256x256, HOG orientations=9, PPC=(8,8), CPB=(2,2), KNN K={selected_k}, Euclidean, distance-weighted")
    print(f"FINAL LOOCV ACCURACY : {best_k_row['overall_accuracy']}% ({best_k_row['total_correct']}/360)")
    print(f"MACRO PRECISION      : {best_k_row['macro_precision']}%")
    print(f"MACRO RECALL         : {best_k_row['macro_recall']}%")
    print(f"MACRO F1             : {best_k_row['macro_f1']}%")
    print("=" * 80)

    print("\nPer-Class Stability Summary:")
    print(f"  Min Class Acc: {best_k_row['min_class_acc']}% | Max Class Acc: {best_k_row['max_class_acc']}% | Std Dev: {best_k_row['std_class_acc']}%")
    print(f"  Students >= 50% Acc: {best_k_row['students_ge50']}/18 ({best_k_row['students_ge50']/18*100:.1f}%)")
    print(f"  Students >= 70% Acc: {best_k_row['students_ge70']}/18 ({best_k_row['students_ge70']/18*100:.1f}%)")
    print(f"  Students < 40% Acc : {best_k_row['students_lt40']}/18 ({best_k_row['students_lt40']/18*100:.1f}%)")

    print("\nDiagnostic Cases (K=1 -> K=3 -> K=5 -> K=7 -> K=9):")
    print(f"  Dinnar : K1={case_dinnar['acc_k1']:.1f}% | K3={case_dinnar['acc_k3']:.1f}% | K5={case_dinnar['acc_k5']:.1f}% | K7={case_dinnar['acc_k7']:.1f}% | K9={case_dinnar['acc_k9']:.1f}%")
    print(f"  Fahim  : K1={case_fahim['acc_k1']:.1f}% | K3={case_fahim['acc_k3']:.1f}% | K5={case_fahim['acc_k5']:.1f}% | K7={case_fahim['acc_k7']:.1f}% | K9={case_fahim['acc_k9']:.1f}%")
    print(f"  Brama  : K1={case_brama['acc_k1']:.1f}% | K3={case_brama['acc_k3']:.1f}% | K5={case_brama['acc_k5']:.1f}% | K7={case_brama['acc_k7']:.1f}% | K9={case_brama['acc_k9']:.1f}%")
    print(f"  Dimas  : K1={case_dimas['acc_k1']:.1f}% | K3={case_dimas['acc_k3']:.1f}% | K5={case_dimas['acc_k5']:.1f}% | K7={case_dimas['acc_k7']:.1f}% | K9={case_dimas['acc_k9']:.1f}%")
    print(f"  Rakha  : K1={case_rakha['acc_k1']:.1f}% | K3={case_rakha['acc_k3']:.1f}% | K5={case_rakha['acc_k5']:.1f}% | K7={case_rakha['acc_k7']:.1f}% | K9={case_rakha['acc_k9']:.1f}%")

    print("=" * 80)
    print("EXPERIMENT G EVALUATION FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment_g()
