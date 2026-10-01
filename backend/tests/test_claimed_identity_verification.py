"""
test_claimed_identity_verification.py — Comprehensive Unit & Integration Tests
==============================================================================
Evaluasi dan Verifikasi Claimed-Identity Verification Mode (Experiment K1 Frozen Protocol).

11 Required Test Cases:
- TEST 1: Claimed writer reference filtering
- TEST 2: Mean Top-5 calculation accuracy
- TEST 3: Score < threshold -> ACCEPT (VALID)
- TEST 4: Score == threshold -> ACCEPT (VALID)
- TEST 5: Score > threshold -> REJECT (TIDAK VALID)
- TEST 6: Decoupled Case A: predicted != claimed, score <= threshold -> ACCEPT
- TEST 7: Decoupled Case B: predicted == claimed, score > threshold -> REJECT
- TEST 8: Unknown claimed writer -> validation error
- TEST 9: Reference count < 5 -> error handling
- TEST 10: Backward compatibility (1-to-N identification output unchanged)
- TEST 11: Production H0 model file SHA-256 hash frozen check
"""

import os
import sys
import hashlib
import unittest
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model.verifier import verify_claim
from model.classifier import classify_handwriting, verify_image, _load_latest_model_files
from database import init_db, get_connection


class DummyLabelEncoder:
    def __init__(self, classes):
        self.classes_ = np.array(classes)

    def transform(self, names):
        mapping = {c: i for i, c in enumerate(self.classes_)}
        return np.array([mapping[n] for n in names])

    def inverse_transform(self, indices):
        return np.array([self.classes_[i] for i in indices])


class TestClaimedIdentityVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.knn_model, cls.label_encoder, cls.X_train, cls.y_train, cls.timestamp = _load_latest_model_files()
        cls.expected_sha256 = "c68e2f9ea54bbba04ea7d7560252df77e08ba96e69c94d89c26711516cc437ef"

    def test_01_claimed_writer_reference_filtering(self):
        """TEST 1: Memastikan HANYA sampel milik claimed_writer yang digunakan."""
        classes = ["Writer_A", "Writer_B", "Writer_C"]
        le = DummyLabelEncoder(classes)
        # 3 writers, 10 samples each, 34596 features
        np.random.seed(42)
        X = np.random.randn(30, 34596).astype(np.float32)
        y = np.array([0]*10 + [1]*10 + [2]*10)

        # Query identical to first sample of Writer_B (index 10)
        query = X[10].copy()

        result = verify_claim(
            query_feature=query,
            claimed_writer="Writer_B",
            X_train=X,
            y_train_encoded=y,
            label_encoder=le,
            k=5,
            threshold=25.1291,
        )

        self.assertEqual(result["claimed_writer"], "Writer_B")
        self.assertEqual(result["available_reference_count"], 10)
        # Distance to self should be exactly 0.0
        self.assertAlmostEqual(result["neighbor_distances"][0], 0.0, places=4)

    def test_02_mean_top5_calculation(self):
        """TEST 2: Verifikasi perhitungan Mean Top-5 terhadap perhitungan manual numpy."""
        classes = ["Writer_A"]
        le = DummyLabelEncoder(classes)
        # 6 samples
        X = np.zeros((6, 10), dtype=np.float32)
        X[0, 0] = 1.0  # dist = 1
        X[1, 0] = 2.0  # dist = 2
        X[2, 0] = 3.0  # dist = 3
        X[3, 0] = 4.0  # dist = 4
        X[4, 0] = 5.0  # dist = 5
        X[5, 0] = 100.0 # dist = 100 (should be excluded from top 5)
        y = np.zeros(6, dtype=int)

        query = np.zeros(10, dtype=np.float32)
        result = verify_claim(
            query_feature=query,
            claimed_writer="Writer_A",
            X_train=X,
            y_train_encoded=y,
            label_encoder=le,
            k=5,
            threshold=25.1291,
        )

        # Expected top 5 distances: 1, 2, 3, 4, 5. Mean = 3.0
        expected_mean = (1.0 + 2.0 + 3.0 + 4.0 + 5.0) / 5.0
        self.assertAlmostEqual(result["score"], expected_mean, places=4)
        self.assertEqual(result["neighbor_distances"], [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_03_score_below_threshold_accept(self):
        """TEST 3: Score < threshold (e.g. 24.0 < 25.1291) -> ACCEPT / VALID."""
        classes = ["Writer_A"]
        le = DummyLabelEncoder(classes)
        # Distances all 24.0
        X = np.full((5, 1), 24.0, dtype=np.float32)
        y = np.zeros(5, dtype=int)
        query = np.zeros(1, dtype=np.float32)

        result = verify_claim(
            query_feature=query,
            claimed_writer="Writer_A",
            X_train=X,
            y_train_encoded=y,
            label_encoder=le,
            k=5,
            threshold=25.1291,
        )

        self.assertEqual(result["decision"], "ACCEPT")
        self.assertEqual(result["status"], "VALID")
        self.assertLess(result["score"], 25.1291)

    def test_04_score_equal_threshold_accept(self):
        """TEST 4: Score == threshold (25.1291) -> ACCEPT / VALID (boundary condition)."""
        classes = ["Writer_A"]
        le = DummyLabelEncoder(classes)
        # 5 samples with distance exactly 25.1291
        X = np.zeros((5, 1), dtype=np.float32)
        X[:, 0] = 25.1291
        y = np.zeros(5, dtype=int)
        query = np.zeros(1, dtype=np.float32)

        result = verify_claim(
            query_feature=query,
            claimed_writer="Writer_A",
            X_train=X,
            y_train_encoded=y,
            label_encoder=le,
            k=5,
            threshold=25.1291,
        )

        self.assertEqual(result["score"], 25.1291)
        self.assertEqual(result["decision"], "ACCEPT")
        self.assertEqual(result["status"], "VALID")

    def test_05_score_above_threshold_reject(self):
        """TEST 5: Score > threshold (e.g. 26.0 > 25.1291) -> REJECT / TIDAK VALID."""
        classes = ["Writer_A"]
        le = DummyLabelEncoder(classes)
        X = np.zeros((5, 1), dtype=np.float32)
        X[:, 0] = 26.0
        y = np.zeros(5, dtype=int)
        query = np.zeros(1, dtype=np.float32)

        result = verify_claim(
            query_feature=query,
            claimed_writer="Writer_A",
            X_train=X,
            y_train_encoded=y,
            label_encoder=le,
            k=5,
            threshold=25.1291,
        )

        self.assertEqual(result["decision"], "REJECT")
        self.assertEqual(result["status"], "TIDAK VALID")
        self.assertGreater(result["score"], 25.1291)

    def test_06_decoupled_case_a(self):
        """
        TEST 6: Decoupled Case A:
        predicted_name != claimed_writer (e.g. KNN predicts Writer_B because it is closer),
        tapi claimed_writer adalah Writer_A dan score(Writer_A) <= threshold (24.0 <= 25.1291).
        Keputusan verifikasi klaim: ACCEPT (VALID).
        """
        classes = ["Writer_A", "Writer_B"]
        le = DummyLabelEncoder(classes)
        # Writer_A samples at distance 24.0, Writer_B samples at distance 22.0
        X = np.zeros((10, 1), dtype=np.float32)
        X[:5, 0] = 24.0  # Writer_A
        X[5:, 0] = 22.0  # Writer_B
        y = np.array([0]*5 + [1]*5)
        query = np.zeros(1, dtype=np.float32)

        # Verifikasi klaim terhadap Writer_A
        result = verify_claim(
            query_feature=query,
            claimed_writer="Writer_A",
            X_train=X,
            y_train_encoded=y,
            label_encoder=le,
            k=5,
            threshold=25.1291,
        )

        self.assertEqual(result["claimed_writer"], "Writer_A")
        self.assertEqual(result["score"], 24.0)
        self.assertEqual(result["decision"], "ACCEPT")
        self.assertEqual(result["status"], "VALID")

    def test_07_decoupled_case_b(self):
        """
        TEST 7: Decoupled Case B:
        predicted_name == claimed_writer (KNN voting memenangkan Writer_A karena ia paling relatif dekat dibanding kelas lain),
        tapi jarak absolut score(Writer_A) > threshold (26.5 > 25.1291).
        Keputusan verifikasi klaim: REJECT (TIDAK VALID).
        """
        classes = ["Writer_A", "Writer_B"]
        le = DummyLabelEncoder(classes)
        # Writer_A samples at distance 26.5, Writer_B samples at distance 35.0
        X = np.zeros((10, 1), dtype=np.float32)
        X[:5, 0] = 26.5  # Writer_A (closest relative, but > 25.1291)
        X[5:, 0] = 35.0  # Writer_B
        y = np.array([0]*5 + [1]*5)
        query = np.zeros(1, dtype=np.float32)

        # Verifikasi klaim terhadap Writer_A
        result = verify_claim(
            query_feature=query,
            claimed_writer="Writer_A",
            X_train=X,
            y_train_encoded=y,
            label_encoder=le,
            k=5,
            threshold=25.1291,
        )

        self.assertEqual(result["claimed_writer"], "Writer_A")
        self.assertEqual(result["score"], 26.5)
        self.assertEqual(result["decision"], "REJECT")
        self.assertEqual(result["status"], "TIDAK VALID")

    def test_08_unknown_claimed_writer_error(self):
        """TEST 8: Claimed writer tidak dikenal -> raise ValueError."""
        sample_query = self.X_train[0]
        with self.assertRaises(ValueError):
            verify_claim(
                query_feature=sample_query,
                claimed_writer="Unknown_Person_XYZ",
                X_train=self.X_train,
                y_train_encoded=self.y_train,
                label_encoder=self.label_encoder,
                k=5,
                threshold=25.1291,
            )

    def test_09_reference_count_less_than_k(self):
        """TEST 9: Referensi sampel claimed writer < K=5 -> raise ValueError."""
        classes = ["Writer_Few"]
        le = DummyLabelEncoder(classes)
        X = np.zeros((3, 10), dtype=np.float32) # only 3 samples
        y = np.zeros(3, dtype=int)
        query = np.zeros(10, dtype=np.float32)

        with self.assertRaises(ValueError) as ctx:
            verify_claim(
                query_feature=query,
                claimed_writer="Writer_Few",
                X_train=X,
                y_train_encoded=y,
                label_encoder=le,
                k=5,
                threshold=25.1291,
            )
        self.assertIn("kurang dari K=5", str(ctx.exception))

    def test_10_identification_output_unchanged(self):
        """TEST 10: Memastikan 1-to-N identification output tidak berubah saat claimed_writer diberikan vs tidak."""
        sample_query = self.X_train[5]
        writer_name = self.label_encoder.inverse_transform([self.y_train[5]])[0]

        res_without = verify_image(
            feature_vector=sample_query,
            query_filename="test_query.png",
            claimed_writer=None,
        )

        res_with = verify_image(
            feature_vector=sample_query,
            query_filename="test_query.png",
            claimed_writer=writer_name,
        )

        self.assertEqual(res_without["predicted_name"], res_with["predicted_name"])
        self.assertAlmostEqual(res_without["similarity_percent"], res_with["similarity_percent"], places=2)
        self.assertAlmostEqual(res_without["euclidean_distance"], res_with["euclidean_distance"], places=4)
        self.assertEqual(len(res_without["top_matches"]), len(res_with["top_matches"]))
        self.assertIn("verification", res_with)
        self.assertEqual(res_with["verification"]["claimed_writer"], writer_name)

    def test_11_model_hash_unchanged(self):
        """TEST 11: Memastikan file model H0 asli tidak bermutasi (SHA-256 hash check)."""
        model_path = os.path.join(config.MODEL_SAVED_DIR, "knn_model_20260926_190940.joblib")
        self.assertTrue(os.path.exists(model_path), f"Model {model_path} must exist.")
        
        with open(model_path, "rb") as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()

        self.assertEqual(
            current_hash,
            self.expected_sha256,
            "CRITICAL: Production model file SHA-256 has changed! Zero-mutation invariant violated."
        )


if __name__ == "__main__":
    unittest.main()
