"""
tests/pilot_pure_crop_check.py — Pilot Visual Sanity Check for Pure Crop Protocol
===================================================================================
Generates 3-panel visual contact sheets (ORIGINAL | BASELINE ROI | PURE CROP)
for Positions #1, #16, #4 (highest), and #15 (median) across all 18 students.
"""

import os
import sys
import sqlite3
import numpy as np
import cv2
import math
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import DB_PATH, DATASET_RAW_DIR
from preprocessing.image_processor import (
    normalize_orientation, convert_to_grayscale,
    apply_gaussian_blur, apply_otsu_threshold, remove_noise,
    extract_roi, resize_with_aspect_ratio,
)

OUT_DIR = os.path.join(BASE_DIR, "tests", "evaluation_results", "pure_crop_experiment")
CONTACT_DIR = os.path.join(OUT_DIR, "contact_sheets")
os.makedirs(CONTACT_DIR, exist_ok=True)


def pure_handwriting_crop(image_bgr):
    """
    Deterministic layout/geometric pure handwriting crop.
    Rule:
      1. Normalize orientation (portrait).
      2. Exclude top header zone (top 15% of page where student metadata/printed headers/titles reside).
      3. Exclude bottom footer zone (bottom 8% margin).
      4. Exclude side margins (6% on left and right).
      5. Apply standard grayscale -> blur -> Otsu -> noise removal -> baseline extract_roi()
         on the extracted central region.
      6. Preserves ALL handwriting strokes within the region (NO aggressive component masking).
    Returns: (cropped_roi_bgr, crop_bbox_coords, full_processed_binary)
    """
    oriented = normalize_orientation(image_bgr)
    h, w = oriented.shape[:2]

    # Deterministic spatial crop bounds (15% top header cut, 8% bottom footer cut, 6% side margin cuts)
    y1 = int(h * 0.15)
    y2 = int(h * 0.92)
    x1 = int(w * 0.06)
    x2 = int(w * 0.94)

    # Sub-image containing pure handwriting body
    sub_bgr = oriented[y1:y2, x1:x2]

    # Preprocessing pipeline on sub-image
    gray = convert_to_grayscale(sub_bgr)
    blurred = apply_gaussian_blur(gray)
    binary = apply_otsu_threshold(blurred)
    denoised = remove_noise(binary)
    roi_binary = extract_roi(denoised)

    # Calculate actual bounding box coordinates in original image space for visualization
    # Find bounding box in sub-image
    contours, _ = cv2.findContours(denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        all_pts = np.concatenate(contours, axis=0)
        bx, by, bw, bh = cv2.boundingRect(all_pts)
        pad = max(4, int(min(bw, bh) * 0.05))
        bx1 = max(0, bx - pad) + x1
        by1 = max(0, by - pad) + y1
        bx2 = min(w, bx + bw + pad) + x1
        by2 = min(h, by + bh + pad) + y1
    else:
        bx1, by1, bx2, by2 = x1, y1, x2, y2

    return roi_binary, (bx1, by1, bx2 - bx1, by2 - by1), oriented


def make_pilot_contact_sheet(samples, position_num, title, save_path):
    """
    Creates a 3-panel comparison contact sheet for 18 students at position position_num:
    For each student, shows:
      [1. Original + Boxes]  [2. Baseline ROI]  [3. Pure Crop 256]
    """
    thumb_h = 240
    thumb_w = 180
    card_w = thumb_w * 3 + 40
    card_h = thumb_h + 50

    cols = 2  # 2 students per row -> 9 rows for 18 students
    rows = math.ceil(len(samples) / cols)

    sheet = np.full((rows * card_h + 60, cols * card_w, 3), 20, dtype=np.uint8)

    # Main header
    cv2.putText(sheet, title, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2, cv2.LINE_AA)

    for i, s in enumerate(samples):
        r_idx = i // cols
        c_idx = i % cols
        x_base = c_idx * card_w + 15
        y_base = 60 + r_idx * card_h

        img_bgr = cv2.imread(s["raw_path"])
        if img_bgr is None:
            continue

        # Baseline pipeline
        oriented = normalize_orientation(img_bgr)
        orig_h, orig_w = oriented.shape[:2]
        gray = convert_to_grayscale(oriented)
        blurred = apply_gaussian_blur(gray)
        binary = apply_otsu_threshold(blurred)
        denoised = remove_noise(binary)
        base_roi = extract_roi(denoised)
        base_256 = resize_with_aspect_ratio(base_roi, (256, 256))

        # Baseline bbox
        b_cnts, _ = cv2.findContours(denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if b_cnts:
            b_pts = np.concatenate(b_cnts, axis=0)
            bbx, bby, bbw, bbh = cv2.boundingRect(b_pts)
        else:
            bbx, bby, bbw, bbh = 0, 0, orig_w, orig_h

        # Pure crop pipeline
        pure_roi, (px, py, pw, ph), _ = pure_handwriting_crop(img_bgr)
        pure_256 = resize_with_aspect_ratio(pure_roi, (256, 256))

        # 1. Original with bounding boxes: Baseline (Red) vs Pure Crop (Cyan)
        vis_orig = oriented.copy()
        cv2.rectangle(vis_orig, (bbx, bby), (bbx + bbw, bby + bbh), (0, 0, 255), max(2, int(min(orig_w, orig_h)*0.004)))
        cv2.rectangle(vis_orig, (px, py), (px + pw, py + ph), (255, 255, 0), max(2, int(min(orig_w, orig_h)*0.004)))
        scale_orig = min(thumb_w / orig_w, thumb_h / orig_h)
        t_orig = cv2.resize(vis_orig, (int(orig_w * scale_orig), int(orig_h * scale_orig)))
        p1 = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
        p1[(thumb_h - t_orig.shape[0])//2 : (thumb_h - t_orig.shape[0])//2 + t_orig.shape[0],
           (thumb_w - t_orig.shape[1])//2 : (thumb_w - t_orig.shape[1])//2 + t_orig.shape[1]] = t_orig

        # 2. Baseline 256 thumbnail
        b_disp = cv2.resize(cv2.cvtColor(base_256, cv2.COLOR_GRAY2BGR), (thumb_w, thumb_h))

        # 3. Pure Crop 256 thumbnail
        c_disp = cv2.resize(cv2.cvtColor(pure_256, cv2.COLOR_GRAY2BGR), (thumb_w, thumb_h))

        # Place panels side-by-side
        sheet[y_base:y_base + thumb_h, x_base:x_base + thumb_w] = p1
        sheet[y_base:y_base + thumb_h, x_base + thumb_w + 10:x_base + thumb_w * 2 + 10] = b_disp
        sheet[y_base:y_base + thumb_h, x_base + thumb_w * 2 + 20:x_base + thumb_w * 3 + 20] = c_disp

        # Labels
        student_label = f"{s['student_name']} (Sample #{position_num:02d})"
        cv2.putText(sheet, student_label, (x_base, y_base + thumb_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(sheet, "Red=Base | Cyan=PureCrop", (x_base, y_base + thumb_h + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(sheet, "Baseline ROI", (x_base + thumb_w + 10, y_base + thumb_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(sheet, "Pure Crop 256", (x_base + thumb_w * 2 + 20, y_base + thumb_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 120), 1, cv2.LINE_AA)

        # Border around card
        cv2.rectangle(sheet, (x_base - 5, y_base - 5), (x_base + card_w - 20, y_base + card_h - 10), (60, 60, 60), 1)

    cv2.imwrite(save_path, sheet)
    print(f"Saved pilot contact sheet: {save_path}")


def run_pilot():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, student_name, original_filename, saved_filename, file_path "
        "FROM dataset ORDER BY student_name ASC, original_filename ASC, saved_filename ASC"
    ).fetchall()
    conn.close()

    students = sorted(list(set(r["student_name"] for r in rows)))
    all_samples = []
    student_counts = defaultdict(int)
    for r in rows:
        st = r["student_name"]
        student_counts[st] += 1
        raw_name = r["saved_filename"] or os.path.basename(r["file_path"])
        raw_path = os.path.join(DATASET_RAW_DIR, raw_name)
        all_samples.append({
            "id": r["id"],
            "student_name": st,
            "filename": r["original_filename"],
            "raw_path": raw_path,
            "sample_num": student_counts[st],
        })

    # Generate pilot contact sheets for Positions #1, #16, #4 (best), #15 (median)
    pos_targets = [
        (1, "Position #01 (Known Outlier: Headers/Titles)"),
        (16, "Position #16 (Low Accuracy Position)"),
        (4, "Position #04 (Highest Accuracy Position)"),
        (15, "Position #15 (Median Accuracy Position)"),
    ]

    for pos, desc in pos_targets:
        pos_samps = [s for s in all_samples if s["sample_num"] == pos]
        save_file = os.path.join(CONTACT_DIR, f"pilot_contact_sheet_position_{pos:02d}.png")
        make_pilot_contact_sheet(pos_samps, pos, f"PILOT CHECK — {desc}", save_file)


if __name__ == "__main__":
    run_pilot()
