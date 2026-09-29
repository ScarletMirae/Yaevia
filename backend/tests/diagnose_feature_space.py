"""
tests/diagnose_feature_space.py — Diagnostic & Audit Tooling for Yaevia HOG-KNN System
======================================================================================
Audit Baseline Commit: 62e7e56bd1005679d44e4ed59eef873369f3dade

Tujuan:
1. Audit pipeline preprocessing: orientation -> gray -> blur -> Otsu -> morphology -> ROI -> resize 128x128.
2. Analisis matematis ruang fitur HOG (intra-class vs inter-class distance).
3. Investigasi khusus confusion antara Fahim J Mujaddid vs Fathurrahman Nugroho.
4. Investigasi anomali halaman pertama vs halaman berikutnya (pengaruh layout/screenshot/header).
5. Analisis margin keputusan KNN (proporsi 1/5, 2/5 tetangga).
6. Evaluasi hipotesis kegagalan extract_roi() pada citra dengan elemen non-tulisan.

Output disimpan ke: backend/tests/evaluation_results/ (di-ignore oleh git).
"""

import os
import sys
import json
import sqlite3
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from skimage.feature import hog

# Ensure backend root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import (
    DB_PATH, DATASET_RAW_DIR, DATASET_PROCESSED_DIR,
    IMAGE_SIZE, GAUSSIAN_BLUR_KERNEL, MEDIAN_BLUR_KERNEL, MORPH_KERNEL_SIZE,
    HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK, HOG_BLOCK_NORM,
    KNN_N_NEIGHBORS, KNN_METRIC, KNN_WEIGHTS,
    SIMILARITY_THRESHOLDS, VERIFICATION_CONFIDENCE_THRESHOLDS,
)
RAW_DIR = DATASET_RAW_DIR
PROCESSED_DIR = DATASET_PROCESSED_DIR
from preprocessing.image_processor import (
    normalize_orientation, convert_to_grayscale, apply_gaussian_blur,
    apply_otsu_threshold, remove_noise, extract_roi, resize_with_aspect_ratio
)
from model.classifier import euclidean_to_similarity, get_similarity_status, get_verification_status

DIAG_OUT_DIR = os.path.join(BASE_DIR, "tests", "evaluation_results", "audit_diagnosis")
VIS_OUT_DIR = os.path.join(DIAG_OUT_DIR, "visualizations")
os.makedirs(VIS_OUT_DIR, exist_ok=True)


def run_full_diagnosis():
    print("=" * 80)
    print("STARTING YAEVIA FEATURE-SPACE & PREPROCESSING DIAGNOSTIC AUDIT")
    print("=" * 80)

    # 1. Load dataset records from DB
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    rows = cur.execute(
        "SELECT id, student_name, student_id, original_filename, saved_filename, file_path "
        "FROM dataset ORDER BY student_name ASC, original_filename ASC, saved_filename ASC"
    ).fetchall()
    conn.close()
    
    print(f"Loaded {len(rows)} records across {len(set(r['student_name'] for r in rows))} students.")
    
    # 2. Process all images through pipeline step-by-step, recording metrics
    samples_data = []
    features_list = []
    labels_list = []
    
    print("\nExecuting step-by-step pipeline audit for all samples...")
    for idx, r in enumerate(rows):
        sample_id = r["id"]
        student = r["student_name"]
        raw_name = r["saved_filename"] or os.path.basename(r["file_path"])
        raw_path = os.path.join(RAW_DIR, raw_name)
        
        if not os.path.exists(raw_path):
            print(f"Warning: file not found {raw_path}")
            continue
            
        img_bgr = cv2.imread(raw_path)
        if img_bgr is None:
            print(f"Warning: could not read {raw_path}")
            continue
            
        # Step 0: Orientation
        img_oriented = normalize_orientation(img_bgr)
        orig_h, orig_w = img_oriented.shape[:2]
        
        # Step 1: Grayscale
        gray = convert_to_grayscale(img_oriented)
        
        # Step 2: Blur
        blurred = apply_gaussian_blur(gray)
        
        # Step 3: Otsu
        binary = apply_otsu_threshold(blurred)
        
        # Step 4: Morphology noise removal
        denoised = remove_noise(binary)
        
        # Step 5: ROI extraction & Contour Analysis
        contours, _ = cv2.findContours(denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)
        
        if contours:
            all_pts = np.concatenate(contours, axis=0)
            rx, ry, rw, rh = cv2.boundingRect(all_pts)
            
            # Sorted contour areas
            c_areas = sorted([cv2.contourArea(c) for c in contours], reverse=True)
            max_c_area = c_areas[0] if c_areas else 0
            sum_c_area = sum(c_areas)
        else:
            rx, ry, rw, rh = 0, 0, orig_w, orig_h
            max_c_area = 0
            sum_c_area = 0
            
        roi = extract_roi(denoised)
        roi_h, roi_w = roi.shape[:2]
        
        # Step 6: 128x128 resize
        final_128 = resize_with_aspect_ratio(roi, IMAGE_SIZE)
        
        # Step 7: HOG feature & visualization
        feat, hog_vis = hog(
            final_128,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            block_norm=HOG_BLOCK_NORM,
            visualize=True,
            feature_vector=True,
        )
        
        feat_norm = float(np.linalg.norm(feat))
        
        # Metrics
        orig_area = orig_w * orig_h
        roi_area = rw * rh
        roi_area_ratio = roi_area / orig_area if orig_area > 0 else 0
        
        fg_ratio_before = float(np.sum(denoised > 0)) / orig_area if orig_area > 0 else 0
        fg_ratio_after = float(np.sum(roi > 0)) / (roi_w * roi_h) if (roi_w * roi_h) > 0 else 0
        roi_aspect_ratio = rw / rh if rh > 0 else 0
        
        sample_info = {
            "idx": idx,
            "id": sample_id,
            "student_name": student,
            "original_filename": r["original_filename"],
            "saved_filename": raw_name,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "num_contours": num_contours,
            "max_contour_area": max_c_area,
            "sum_contour_area": sum_c_area,
            "roi_x": int(rx),
            "roi_y": int(ry),
            "roi_w": int(rw),
            "roi_h": int(rh),
            "roi_cropped_w": int(roi_w),
            "roi_cropped_h": int(roi_h),
            "roi_area_ratio": round(roi_area_ratio, 4),
            "fg_ratio_before": round(fg_ratio_before, 4),
            "fg_ratio_after": round(fg_ratio_after, 4),
            "roi_aspect_ratio": round(roi_aspect_ratio, 4),
            "feat_dim": len(feat),
            "feat_norm": round(feat_norm, 4),
        }
        
        samples_data.append(sample_info)
        features_list.append(feat)
        labels_list.append(student)
        
        # Generate fast multi-stage visual montage with cv2
        student_dir = os.path.join(VIS_OUT_DIR, student.replace(" ", "_").replace("'", ""))
        os.makedirs(student_dir, exist_ok=True)
        
        vis_save_path = os.path.join(student_dir, f"sample_{idx % 20 + 1:02d}_id{sample_id}.png")
        if not os.path.exists(vis_save_path):
            # 1. Original with red bounding box (thumbnail h=256)
            vis_orig_box = img_oriented.copy()
            cv2.rectangle(vis_orig_box, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), max(2, int(min(orig_w, orig_h)*0.005)))
            scale_orig = 256.0 / orig_h
            thumb_orig = cv2.resize(vis_orig_box, (int(orig_w * scale_orig), 256))
            
            # 2. Binary with green bounding box (thumbnail h=256)
            vis_bin_box = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(vis_bin_box, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), max(2, int(min(orig_w, orig_h)*0.005)))
            thumb_bin = cv2.resize(vis_bin_box, (int(orig_w * scale_orig), 256))
            
            # 3. ROI crop (resized to h=256 keeping aspect ratio)
            roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            scale_roi = 256.0 / roi_h if roi_h > 0 else 1.0
            thumb_roi = cv2.resize(roi_bgr, (max(10, int(roi_w * scale_roi)), 256))
            
            # 4. Final 128x128 Letterbox (scaled to 256x256 display)
            thumb_final = cv2.resize(cv2.cvtColor(final_128, cv2.COLOR_GRAY2BGR), (256, 256), interpolation=cv2.INTER_NEAREST)
            
            # 5. HOG visualization (scaled to 256x256 display)
            hog_vis_uint8 = np.uint8(np.clip(hog_vis * 255, 0, 255))
            hog_vis_color = cv2.applyColorMap(hog_vis_uint8, cv2.COLORMAP_INFERNO)
            thumb_hog = cv2.resize(hog_vis_color, (256, 256), interpolation=cv2.INTER_NEAREST)
            
            # Add labels to thumbnails
            def add_label(img, text, subtext=""):
                h, w = img.shape[:2]
                header = np.zeros((40, w, 3), dtype=np.uint8) + 30
                cv2.putText(header, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                if subtext:
                    cv2.putText(header, subtext, (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
                return np.vstack([header, img])
                
            panel1 = add_label(thumb_orig, "1. Original + BBox", f"{orig_w}x{orig_h} (cov: {roi_area_ratio*100:.0f}%)")
            panel2 = add_label(thumb_bin, "2. Otsu + Denoise", f"contours: {num_contours}")
            panel3 = add_label(thumb_roi, "3. ROI Crop", f"{roi_w}x{roi_h} (ar: {roi_aspect_ratio:.2f})")
            panel4 = add_label(thumb_final, "4. 128x128 Letterbox", "Input to HOG")
            panel5 = add_label(thumb_hog, "5. HOG Gradients", f"8100-dim (norm: {feat_norm:.1f})")
            
            montage = np.hstack([panel1, panel2, panel3, panel4, panel5])
            cv2.imwrite(vis_save_path, montage)

    X = np.array(features_list)
    y = np.array(labels_list)
    n_samples = len(X)
    
    print(f"\nCompleted feature extraction for all {n_samples} samples. Shape: {X.shape}")
    
    # 3. Compute Pairwise Euclidean Distance Matrix
    print("\nComputing pairwise Euclidean distance matrix...")
    dot_prod = np.dot(X, X.T)
    sq_norms = np.diag(dot_prod)
    dist_sq = np.maximum(0.0, sq_norms[:, None] + sq_norms[None, :] - 2 * dot_prod)
    dist_matrix = np.sqrt(dist_sq)
    np.fill_diagonal(dist_matrix, 0.0)
    
    # 4. Intra-class vs Inter-class Distance Analysis
    print("\n" + "=" * 80)
    print("INTRA-CLASS VS INTER-CLASS DISTANCE ANALYSIS")
    print("=" * 80)
    
    unique_classes = sorted(list(set(y)))
    class_indices = {c: np.where(y == c)[0] for c in unique_classes}
    
    intra_distances_all = []
    inter_distances_all = []
    
    class_stats = {}
    
    for c in unique_classes:
        c_idxs = class_indices[c]
        other_idxs = np.where(y != c)[0]
        
        # Intra-class distances
        intra_sub = dist_matrix[np.ix_(c_idxs, c_idxs)]
        i_upper = np.triu_indices(len(c_idxs), k=1)
        intra_dists = intra_sub[i_upper]
        intra_distances_all.extend(intra_dists)
        
        # Inter-class distances from class c to all other classes
        inter_sub = dist_matrix[np.ix_(c_idxs, other_idxs)]
        inter_dists = inter_sub.flatten()
        inter_distances_all.extend(inter_dists)
        
        class_stats[c] = {
            "intra_mean": float(np.mean(intra_dists)),
            "intra_std": float(np.std(intra_dists)),
            "intra_min": float(np.min(intra_dists)),
            "intra_median": float(np.median(intra_dists)),
            "intra_max": float(np.max(intra_dists)),
            "inter_mean": float(np.mean(inter_dists)),
            "inter_min": float(np.min(inter_dists)),
            "inter_median": float(np.median(inter_dists)),
            "separability_ratio": float(np.mean(inter_dists) / np.mean(intra_dists)),
        }
        
    print(f"Overall Intra-class distance: Mean = {np.mean(intra_distances_all):.4f} +/- {np.std(intra_distances_all):.4f}, Min = {np.min(intra_distances_all):.4f}, Max = {np.max(intra_distances_all):.4f}")
    print(f"Overall Inter-class distance: Mean = {np.mean(inter_distances_all):.4f} +/- {np.std(inter_distances_all):.4f}, Min = {np.min(inter_distances_all):.4f}, Max = {np.max(inter_distances_all):.4f}")
    print(f"Overall Separability (Inter / Intra): {np.mean(inter_distances_all) / np.mean(intra_distances_all):.3f}")

    print("\nPER-CLASS DISTANCE SUMMARY:")
    print(f"{'Class Name':<30} | {'Intra Mean':<10} | {'Intra Max':<10} | {'Inter Mean':<10} | {'Inter Min':<10} | {'Ratio (Inter/Intra)'}")
    print("-" * 95)
    for c, s in class_stats.items():
        print(f"{c:<30} | {s['intra_mean']:<10.3f} | {s['intra_max']:<10.3f} | {s['inter_mean']:<10.3f} | {s['inter_min']:<10.3f} | {s['separability_ratio']:.3f}")

    # 5. Specialized Focus: FAHIM J MUJADDID vs FATHURRAHMAN NUGROHO
    print("\n" + "=" * 80)
    print("SPECIALIZED AUDIT: FAHIM J MUJADDID VS FATHURRAHMAN NUGROHO")
    print("=" * 80)
    
    fahim_name = "Fahim J Mujaddid"
    fathur_name = "Fathurrahman Nugroho"
    
    if fahim_name in class_indices and fathur_name in class_indices:
        fahim_idxs = class_indices[fahim_name]
        fathur_idxs = class_indices[fathur_name]
        
        fahim_intra = dist_matrix[np.ix_(fahim_idxs, fahim_idxs)][np.triu_indices(len(fahim_idxs), k=1)]
        fathur_intra = dist_matrix[np.ix_(fathur_idxs, fathur_idxs)][np.triu_indices(len(fathur_idxs), k=1)]
        fahim_fathur_cross = dist_matrix[np.ix_(fahim_idxs, fathur_idxs)].flatten()
        
        other_non_fathur = [i for i in range(n_samples) if y[i] != fahim_name and y[i] != fathur_name]
        fahim_to_others = dist_matrix[np.ix_(fahim_idxs, other_non_fathur)].flatten()
        fathur_to_others = dist_matrix[np.ix_(fathur_idxs, other_non_fathur)].flatten()
        
        print(f"Fahim Intra-class Distance     : Mean = {np.mean(fahim_intra):.4f} +/- {np.std(fahim_intra):.4f} (Min={np.min(fahim_intra):.4f}, Max={np.max(fahim_intra):.4f})")
        print(f"Fathur Intra-class Distance    : Mean = {np.mean(fathur_intra):.4f} +/- {np.std(fathur_intra):.4f} (Min={np.min(fathur_intra):.4f}, Max={np.max(fathur_intra):.4f})")
        print(f"Fahim-Fathur Cross Distance    : Mean = {np.mean(fahim_fathur_cross):.4f} +/- {np.std(fahim_fathur_cross):.4f} (Min={np.min(fahim_fathur_cross):.4f}, Max={np.max(fahim_fathur_cross):.4f})")
        print(f"Fahim to All Other Students    : Mean = {np.mean(fahim_to_others):.4f} +/- {np.std(fahim_to_others):.4f} (Min={np.min(fahim_to_others):.4f})")
        print(f"Fathur to All Other Students   : Mean = {np.mean(fathur_to_others):.4f} +/- {np.std(fathur_to_others):.4f} (Min={np.min(fathur_to_others):.4f})")
        
        cross_below_fahim_intra_mean = np.sum(fahim_fathur_cross < np.mean(fahim_intra))
        cross_below_fathur_intra_mean = np.sum(fahim_fathur_cross < np.mean(fathur_intra))
        print(f"\nCross pairs closer than Fahim intra-mean: {cross_below_fahim_intra_mean} / {len(fahim_fathur_cross)} ({cross_below_fahim_intra_mean/len(fahim_fathur_cross)*100:.1f}%)")
        print(f"Cross pairs closer than Fathur intra-mean: {cross_below_fathur_intra_mean} / {len(fahim_fathur_cross)} ({cross_below_fathur_intra_mean/len(fahim_fathur_cross)*100:.1f}%)")

    # 6. Full Leave-One-Out KNN Simulation (K=5, distance-weighted)
    print("\n" + "=" * 80)
    print("LEAVE-ONE-OUT KNN (K=5, WEIGHTS='DISTANCE') CLASSIFICATION AUDIT")
    print("=" * 80)
    
    eps = 1e-7
    k = KNN_N_NEIGHBORS
    
    loocv_results = []
    misclassified_samples = []
    razor_thin_samples = []
    first_page_anomalies = []
    
    for i in range(n_samples):
        true_label = y[i]
        
        dists_to_others = np.delete(dist_matrix[i], i)
        labels_of_others = np.delete(y, i)
        
        sorted_indices = np.argsort(dists_to_others)
        top_k_indices = sorted_indices[:k]
        top_k_dists = dists_to_others[top_k_indices]
        top_k_labels = labels_of_others[top_k_indices]
        
        weights = 1.0 / np.maximum(top_k_dists, eps)
        total_w = np.sum(weights)
        
        class_votes = defaultdict(float)
        class_neighbor_counts = defaultdict(int)
        class_min_dists = {}
        
        for d_val, lbl, w_val in zip(top_k_dists, top_k_labels, weights):
            class_votes[lbl] += w_val
            class_neighbor_counts[lbl] += 1
            
        for lbl in unique_classes:
            lbl_mask = (labels_of_others == lbl)
            class_min_dists[lbl] = float(np.min(dists_to_others[lbl_mask]))
            
        sorted_classes = sorted(class_votes.items(), key=lambda x: -x[1])
        pred_label = sorted_classes[0][0]
        pred_vote_share = (sorted_classes[0][1] / total_w) * 100.0
        
        true_class_neighbors = class_neighbor_counts[true_label]
        true_class_vote_share = (class_votes[true_label] / total_w) * 100.0 if true_label in class_votes else 0.0
        
        min_dist_true = class_min_dists[true_label]
        min_dist_pred = class_min_dists[pred_label]
        
        is_correct = (pred_label == true_label)
        
        neighbors_detail = []
        for rank, (d_val, lbl, w_val) in enumerate(zip(top_k_dists, top_k_labels, weights), start=1):
            neighbors_detail.append({
                "rank": rank,
                "name": lbl,
                "distance": round(float(d_val), 4),
                "similarity": euclidean_to_similarity(d_val),
                "vote_contrib_pct": round(float((w_val / total_w) * 100.0), 2),
            })
            
        res_entry = {
            "sample_idx": i,
            "sample_id": samples_data[i]["id"],
            "student_name": true_label,
            "original_filename": samples_data[i]["original_filename"],
            "saved_filename": samples_data[i]["saved_filename"],
            "sample_num_in_class": (i % 20) + 1,
            "is_correct": is_correct,
            "pred_label": pred_label,
            "pred_vote_share": round(pred_vote_share, 2),
            "true_class_neighbors": true_class_neighbors,
            "true_class_vote_share": round(true_class_vote_share, 2),
            "min_dist_true": round(min_dist_true, 4),
            "min_dist_pred": round(min_dist_pred, 4),
            "roi_area_ratio": samples_data[i]["roi_area_ratio"],
            "roi_aspect_ratio": samples_data[i]["roi_aspect_ratio"],
            "fg_ratio_after": samples_data[i]["fg_ratio_after"],
            "neighbors_detail": neighbors_detail,
        }
        
        loocv_results.append(res_entry)
        
        if not is_correct:
            misclassified_samples.append(res_entry)
        elif true_class_neighbors <= 2:
            razor_thin_samples.append(res_entry)
            
        if (i % 20) == 0:
            first_page_anomalies.append(res_entry)

    overall_acc = sum(1 for r in loocv_results if r["is_correct"]) / n_samples * 100.0
    print(f"Overall Leave-One-Out Accuracy: {overall_acc:.2f}% ({sum(1 for r in loocv_results if r['is_correct'])} / {n_samples})")
    print(f"Total Misclassified Samples: {len(misclassified_samples)} ({len(misclassified_samples)/n_samples*100:.2f}%)")
    print(f"Total Razor-Thin Correct Predictions (<=2 neighbors): {len(razor_thin_samples)} ({len(razor_thin_samples)/n_samples*100:.2f}%)")

    # 7. Page 1 vs Page 2+ Anomaly Breakdown
    print("\n" + "=" * 80)
    print("PAGE 1 VS PAGE 2+ ANOMALY & ROI AUDIT")
    print("=" * 80)
    
    page1_results = [r for r in loocv_results if r["sample_num_in_class"] == 1]
    page_other_results = [r for r in loocv_results if r["sample_num_in_class"] > 1]
    
    page1_acc = sum(1 for r in page1_results if r["is_correct"]) / len(page1_results) * 100.0
    page_other_acc = sum(1 for r in page_other_results if r["is_correct"]) / len(page_other_results) * 100.0
    
    print(f"Page 1 Accuracy       : {page1_acc:.2f}% ({sum(1 for r in page1_results if r['is_correct'])} / {len(page1_results)})")
    print(f"Page 2-20 Accuracy    : {page_other_acc:.2f}% ({sum(1 for r in page_other_results if r['is_correct'])} / {len(page_other_results)})")
    
    p1_roi_areas = [r["roi_area_ratio"] for r in page1_results]
    p_other_roi_areas = [r["roi_area_ratio"] for r in page_other_results]
    p1_aspects = [r["roi_aspect_ratio"] for r in page1_results]
    p_other_aspects = [r["roi_aspect_ratio"] for r in page_other_results]
    
    print(f"Page 1 Mean ROI Area Ratio    : {np.mean(p1_roi_areas):.3f} (Aspect: {np.mean(p1_aspects):.3f})")
    print(f"Page 2+ Mean ROI Area Ratio   : {np.mean(p_other_roi_areas):.3f} (Aspect: {np.mean(p_other_aspects):.3f})")

    print("\nPAGE 1 EVALUATION DETAILS:")
    for r in page1_results:
        status_sym = "[PASS]" if r["is_correct"] else "[FAIL]"
        print(f" {status_sym} {r['student_name']:<28} | File: {r['original_filename']:<20} | Pred: {r['pred_label']:<28} | KNN: {r['true_class_neighbors']}/5 | ROI Area: {r['roi_area_ratio']:.2f}")

    # 8. List of Misclassified Samples
    print("\n" + "=" * 80)
    print("COMPLETE LIST OF MISCLASSIFIED SAMPLES")
    print("=" * 80)
    for m in misclassified_samples:
        print(f"Sample #{m['sample_num_in_class']:02d} (id={m['sample_id']}) {m['student_name']:<25} -> Pred: {m['pred_label']:<25} (Vote: {m['pred_vote_share']}%) | True Min Dist: {m['min_dist_true']} | Pred Min Dist: {m['min_dist_pred']}")
        for n in m["neighbors_detail"]:
            print(f"    Neighbor #{n['rank']}: {n['name']:<25} | d={n['distance']:.4f} | sim={n['similarity']:.1f}% | vote_contrib={n['vote_contrib_pct']}%")

    # 9. Outlier Detection per Student
    print("\n" + "=" * 80)
    print("OUTLIER IDENTIFICATION PER STUDENT")
    print("=" * 80)
    
    outliers = []
    for c in unique_classes:
        c_idxs = class_indices[c]
        c_dist_sub = dist_matrix[np.ix_(c_idxs, c_idxs)]
        
        mean_intra_per_sample = []
        for i_pos in range(len(c_idxs)):
            row_dists = np.delete(c_dist_sub[i_pos], i_pos)
            mean_intra_per_sample.append(np.mean(row_dists))
            
        c_mean = np.mean(mean_intra_per_sample)
        c_std = np.std(mean_intra_per_sample)
        
        for i_pos, (s_idx, m_val) in enumerate(zip(c_idxs, mean_intra_per_sample)):
            if m_val > (c_mean + 1.5 * c_std):
                outlier_info = {
                    "student_name": c,
                    "sample_num": (s_idx % 20) + 1,
                    "sample_id": samples_data[s_idx]["id"],
                    "filename": samples_data[s_idx]["original_filename"],
                    "mean_intra_dist": round(m_val, 4),
                    "class_mean_intra": round(c_mean, 4),
                    "std_above": round((m_val - c_mean) / c_std if c_std > 0 else 0, 2),
                    "roi_area_ratio": samples_data[s_idx]["roi_area_ratio"],
                    "roi_aspect_ratio": samples_data[s_idx]["roi_aspect_ratio"],
                }
                outliers.append(outlier_info)
                print(f"OUTLIER: {c:<25} Sample #{outlier_info['sample_num']:02d} ({outlier_info['filename']}) -> IntraDist: {m_val:.3f} (ClassMean: {c_mean:.3f}, +{outlier_info['std_above']} std) | ROI Area: {outlier_info['roi_area_ratio']:.2f}")

    # 10. Audit of extract_roi() Failure Cases
    print("\n" + "=" * 80)
    print("HYPOTHESIS TESTING: EXTRACT_ROI() & BOUNDING BOX UNION IMPACT")
    print("=" * 80)
    
    all_roi_areas = np.array([s["roi_area_ratio"] for s in samples_data])
    all_aspects = np.array([s["roi_aspect_ratio"] for s in samples_data])
    
    misclass_indices = [m["sample_idx"] for m in misclassified_samples]
    correct_indices = [r["sample_idx"] for r in loocv_results if r["is_correct"]]
    
    misclass_roi_areas = all_roi_areas[misclass_indices] if misclass_indices else np.array([])
    correct_roi_areas = all_roi_areas[correct_indices]
    
    print(f"Mean ROI Area Ratio - Correct Samples       : {np.mean(correct_roi_areas):.4f} +/- {np.std(correct_roi_areas):.4f}")
    if len(misclass_roi_areas) > 0:
        print(f"Mean ROI Area Ratio - Misclassified Samples : {np.mean(misclass_roi_areas):.4f} +/- {np.std(misclass_roi_areas):.4f}")
    
    full_page_rois = [s for s in samples_data if s["roi_area_ratio"] > 0.85]
    print(f"\nSamples with Full-Page ROI (>85% of image): {len(full_page_rois)} / {n_samples} ({len(full_page_rois)/n_samples*100:.1f}%)")
    
    # 11. Save Full Diagnostic JSON Report
    report = {
        "dataset_summary": {
            "total_samples": n_samples,
            "total_classes": len(unique_classes),
            "samples_per_class": {c: len(class_indices[c]) for c in unique_classes},
        },
        "distance_metrics": {
            "overall_intra_mean": float(np.mean(intra_distances_all)),
            "overall_intra_std": float(np.std(intra_distances_all)),
            "overall_intra_min": float(np.min(intra_distances_all)),
            "overall_intra_max": float(np.max(intra_distances_all)),
            "overall_inter_mean": float(np.mean(inter_distances_all)),
            "overall_inter_std": float(np.std(inter_distances_all)),
            "overall_inter_min": float(np.min(inter_distances_all)),
            "overall_inter_max": float(np.max(inter_distances_all)),
            "separability_ratio": float(np.mean(inter_distances_all) / np.mean(intra_distances_all)),
            "per_class": class_stats,
        },
        "fahim_vs_fathur": {
            "fahim_intra_mean": float(np.mean(fahim_intra)) if 'fahim_intra' in locals() else None,
            "fathur_intra_mean": float(np.mean(fathur_intra)) if 'fathur_intra' in locals() else None,
            "fahim_fathur_cross_mean": float(np.mean(fahim_fathur_cross)) if 'fahim_fathur_cross' in locals() else None,
            "fahim_to_others_mean": float(np.mean(fahim_to_others)) if 'fahim_to_others' in locals() else None,
            "fathur_to_others_mean": float(np.mean(fathur_to_others)) if 'fathur_to_others' in locals() else None,
        },
        "loocv_knn_performance": {
            "overall_accuracy_percent": round(overall_acc, 2),
            "page1_accuracy_percent": round(page1_acc, 2),
            "page_other_accuracy_percent": round(page_other_acc, 2),
            "total_misclassified": len(misclassified_samples),
            "total_razor_thin": len(razor_thin_samples),
            "misclassified_samples": misclassified_samples,
            "razor_thin_samples": razor_thin_samples,
        },
        "outliers": outliers,
        "extract_roi_audit": {
            "mean_roi_area_ratio_correct": float(np.mean(correct_roi_areas)),
            "mean_roi_area_ratio_misclassified": float(np.mean(misclass_roi_areas)) if len(misclass_roi_areas) > 0 else None,
            "full_page_rois_count": len(full_page_rois),
        }
    }
    
    report_json_path = os.path.join(DIAG_OUT_DIR, "diagnostic_audit_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(
            report, f, indent=2,
            default=lambda o: int(o) if isinstance(o, (np.integer, int)) else float(o) if isinstance(o, (np.floating, float)) else str(o)
        )
        
    print(f"\nDiagnostic audit report saved to: {report_json_path}")
    print(f"Sample step-by-step visualizations saved to: {VIS_OUT_DIR}")
    print("=" * 80)
    print("AUDIT EXECUTION FINISHED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_full_diagnosis()
