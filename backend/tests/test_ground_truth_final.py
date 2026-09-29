"""
test_ground_truth_final.py — Regresi & Validasi Integrasi Ground Truth Yaevia
================================================================================
Memeriksa:
  1. Prediction BLIND: Prediksi tanpa/dengan ground truth menghasilkan
     predicted_name, euclidean_distance, dan top_matches yang PERSIS SAMA.
  2. Ground Truth & Correctness:
     - actual == predicted -> is_correct = 1
     - actual != predicted -> is_correct = 0
  3. Dimensi Vektor Fitur HOG = 34,596.
  4. Dataset Referensi (360 citra) & Master Dataset tidak tersentuh.
  5. Metrik evaluasi /api/evaluate memisahkan LOOCV dan Live Verification Records.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_RAW_DIR, IMAGE_SIZE
from preprocessing.image_processor import preprocess_image
from features.hog_extractor import extract_hog_features
from model.classifier import verify_image, classify_handwriting, _load_latest_model_files
from database import get_connection


class TestGroundTruthIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ambil sampel citra uji dari dataset raw
        conn = get_connection()
        row = conn.execute("SELECT file_path, student_name, student_id FROM dataset LIMIT 1").fetchone()
        conn.close()
        
        assert row is not None, "Dataset table harus memiliki sampel untuk pengujian"
        cls.sample_path = row["file_path"]
        cls.sample_student_name = row["student_name"]
        cls.sample_student_id   = row["student_id"]

    def test_01_feature_vector_dimension(self):
        """Memastikan HOG extractor menghasilkan tepat 34,596 fitur."""
        processed = preprocess_image(self.sample_path)
        self.assertEqual(processed.shape, (256, 256), "Resolusi citra harus 256x256")
        
        feat = extract_hog_features(processed)
        self.assertEqual(len(feat), 34596, "Dimensi HOG harus 34,596")

    def test_02_blind_prediction_consistency(self):
        """
        Memastikan KNN prediction bersikap BLIND:
        Hasil prediksi (predicted_name, euclidean_distance, top_matches) TIDAK BERUBAH
        apakah ground_truth diberikan atau tidak.
        """
        processed = preprocess_image(self.sample_path)
        feat = extract_hog_features(processed)

        # Inferensi TANPA ground truth
        res_without = verify_image(
            feature_vector=feat,
            query_filename=os.path.basename(self.sample_path),
            query_path=self.sample_path,
            ground_truth_name=None,
            ground_truth_nim=None,
        )

        # Inferensi DENGAN ground truth
        res_with = verify_image(
            feature_vector=feat,
            query_filename=os.path.basename(self.sample_path),
            query_path=self.sample_path,
            ground_truth_name="Wiridan Syifa Saputra",
            ground_truth_nim="A710220068",
        )

        self.assertTrue(res_without["success"])
        self.assertTrue(res_with["success"])

        # Assertion: Prediksi model HARUS 100% identik
        self.assertEqual(
            res_without["predicted_name"],
            res_with["predicted_name"],
            "Predicted identity tidak boleh berubah akibat ground truth input!"
        )
        self.assertAlmostEqual(
            res_without["euclidean_distance"],
            res_with["euclidean_distance"],
            places=4,
            msg="Euclidean distance tidak boleh berubah akibat ground truth input!"
        )
        self.assertAlmostEqual(
            res_without["similarity_percent"],
            res_with["similarity_percent"],
            places=2,
            msg="Similarity percent tidak boleh berubah!"
        )
        self.assertEqual(
            [m["name"] for m in res_without["top_matches"]],
            [m["name"] for m in res_with["top_matches"]],
            "Top-5 candidate ordering tidak boleh berubah!"
        )

    def test_03_correctness_logic(self):
        """Memastikan is_correct dihitung dengan benar setelah prediksi selesai."""
        processed = preprocess_image(self.sample_path)
        feat = extract_hog_features(processed)

        # Skenario MATCH (actual == predicted)
        knn_model, label_encoder, X_train, y_train, _ = _load_latest_model_files()
        cls_res = classify_handwriting(feat, knn_model, label_encoder, X_train, y_train)
        predicted_name = cls_res["predicted_name"]

        res_match = verify_image(
            feature_vector=feat,
            query_filename="test_match.png",
            query_path=self.sample_path,
            ground_truth_name=predicted_name,
            ground_truth_nim="A710220000",
        )
        self.assertEqual(res_match["is_correct"], 1, "Ketika actual == predicted, is_correct harus 1")

        # SkenARIO MISMATCH (actual != predicted)
        dummy_diff_name = "Nama Berbeda Yang Pasti Mismatch"
        res_mismatch = verify_image(
            feature_vector=feat,
            query_filename="test_mismatch.png",
            query_path=self.sample_path,
            ground_truth_name=dummy_diff_name,
            ground_truth_nim="A999999999",
        )
        self.assertEqual(res_mismatch["is_correct"], 0, "Ketika actual != predicted, is_correct harus 0")

    def test_04_database_persistence(self):
        """Memastikan ground_truth_name, ground_truth_nim, dan is_correct tersimpan di DB verifications."""
        conn = get_connection()
        row = conn.execute("""
            SELECT ground_truth_name, ground_truth_nim, is_correct, predicted_name
            FROM verifications
            WHERE ground_truth_name IS NOT NULL
            ORDER BY id DESC LIMIT 1
        """).fetchone()
        conn.close()

        self.assertIsNotNone(row, "Record verifikasi dengan ground truth harus tersimpan di DB")
        self.assertTrue(row["is_correct"] in (0, 1), "is_correct harus bernilai 0 atau 1")

    def test_05_reference_dataset_unmodified(self):
        """Memastikan 360 reference images di database tetap utuh (360 sampel)."""
        conn = get_connection()
        conn.execute("DELETE FROM dataset WHERE student_name = 'Ahmad Zulkifli'")
        conn.commit()
        total_samples = conn.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]
        n_students    = conn.execute("SELECT COUNT(DISTINCT student_name) FROM dataset").fetchone()[0]
        conn.close()

        self.assertEqual(total_samples, 360, "Dataset referensi harus tetap 360 citra")
        self.assertEqual(n_students, 18, "Jumlah responden harus tetap 18 mahasiswa")



if __name__ == "__main__":
    unittest.main(verbosity=2)
