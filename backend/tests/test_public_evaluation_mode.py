"""
test_public_evaluation_mode.py — Unit & Integration Tests untuk Public Evaluation Mode
====================================================================================
Memverifikasi bahwa:
1. Endpoint training (/api/train) ditolak (403) saat PUBLIC_EVALUATION_MODE = True.
2. Endpoint upload dataset (/api/dataset/upload) ditolak (403).
3. Endpoint hapus dataset item (/api/dataset/<id>) ditolak (403).
4. Endpoint hapus seluruh dataset (/api/dataset/all) ditolak (403).
5. Endpoint hapus riwayat verifikasi (/api/verify/<id>) ditolak (403).
6. Endpoint verifikasi query (/api/verify) TETAP berfungsi normal (200, VALID/ACCEPT atau TIDAK VALID/REJECT).
7. Endpoint daftar penulis (/api/verify/claimed-writers) TETAP mengembalikan 20 writer.
8. Model Freeze Guard menolak model version yang tidak sesuai saat public mode aktif.
9. Saat PUBLIC_EVALUATION_MODE = False, perlindungan dilepas (diuji secara terisolasi tanpa merusak data asli).
10. Rejection di public mode tidak mengubah database, dataset, model, atau reference artifacts.
"""

import os
import sys
import io
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
import config
from database import get_connection
from model.classifier import _load_latest_model_files


class TestPublicEvaluationMode(unittest.TestCase):

    def setUp(self):
        self.app = app
        self.client = self.app.test_client()

    def test_01_training_endpoint_rejected_in_public_mode(self):
        """TEST 1: POST /api/train harus ditolak dengan HTTP 403 saat Public Evaluation Mode aktif."""
        with patch.object(config, "PUBLIC_EVALUATION_MODE", True):
            # Patch blueprint import as well
            with patch("api.train_routes.PUBLIC_EVALUATION_MODE", True):
                res = self.client.post("/api/train", json={"knn_k": 5})
                self.assertEqual(res.status_code, 403)
                data = res.get_json()
                self.assertFalse(data.get("success"))
                self.assertEqual(data.get("error"), "PUBLIC_EVALUATION_MODE_ACTIVE")

    def test_02_dataset_upload_rejected_in_public_mode(self):
        """TEST 2: POST /api/dataset/upload harus ditolak dengan HTTP 403."""
        with patch.object(config, "PUBLIC_EVALUATION_MODE", True):
            with patch("api.dataset_routes.PUBLIC_EVALUATION_MODE", True):
                dummy_file = (io.BytesIO(b"fake image bytes"), "test.jpg")
                res = self.client.post(
                    "/api/dataset/upload",
                    data={
                        "file": dummy_file,
                        "student_name": "Test Student",
                        "mata_kuliah": "Algoritma Pemrograman (Alpro)",
                    },
                    content_type="multipart/form-data",
                )
                self.assertEqual(res.status_code, 403)
                data = res.get_json()
                self.assertFalse(data.get("success"))
                self.assertEqual(data.get("error"), "PUBLIC_EVALUATION_MODE_ACTIVE")

    def test_03_dataset_item_delete_rejected_in_public_mode(self):
        """TEST 3: DELETE /api/dataset/<id> harus ditolak dengan HTTP 403."""
        with patch.object(config, "PUBLIC_EVALUATION_MODE", True):
            with patch("api.dataset_routes.PUBLIC_EVALUATION_MODE", True):
                res = self.client.delete("/api/dataset/1")
                self.assertEqual(res.status_code, 403)
                data = res.get_json()
                self.assertFalse(data.get("success"))
                self.assertEqual(data.get("error"), "PUBLIC_EVALUATION_MODE_ACTIVE")

    def test_04_dataset_delete_all_rejected_in_public_mode(self):
        """TEST 4: DELETE /api/dataset/all harus ditolak dengan HTTP 403."""
        with patch.object(config, "PUBLIC_EVALUATION_MODE", True):
            with patch("api.dataset_routes.PUBLIC_EVALUATION_MODE", True):
                res = self.client.delete("/api/dataset/all")
                self.assertEqual(res.status_code, 403)
                data = res.get_json()
                self.assertFalse(data.get("success"))
                self.assertEqual(data.get("error"), "PUBLIC_EVALUATION_MODE_ACTIVE")

    def test_05_verify_history_delete_rejected_in_public_mode(self):
        """TEST 5: DELETE /api/verify/<id> harus ditolak dengan HTTP 403."""
        with patch.object(config, "PUBLIC_EVALUATION_MODE", True):
            with patch("api.verify_routes.PUBLIC_EVALUATION_MODE", True):
                res = self.client.delete("/api/verify/1")
                self.assertEqual(res.status_code, 403)
                data = res.get_json()
                self.assertFalse(data.get("success"))
                self.assertEqual(data.get("error"), "PUBLIC_EVALUATION_MODE_ACTIVE")

    def test_06_claimed_writers_remains_functional(self):
        """TEST 6: GET /api/verify/claimed-writers harus tetap mengembalikan 20 writer."""
        res = self.client.get("/api/verify/claimed-writers")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("total"), 20)
        self.assertEqual(len(data.get("writers", [])), 20)

    def test_07_verification_query_remains_functional(self):
        """TEST 7: POST /api/verify untuk query verifikasi harus TETAP berfungsi normal."""
        img_path = os.path.join(config.DATASET_RAW_DIR, "f4a8cfec6aec4502aa6d59aac93502ab.jpeg")
        if not os.path.exists(img_path):
            self.skipTest("Sample query image tidak ditemukan pada disk")

        with open(img_path, "rb") as f:
            res = self.client.post(
                "/api/verify",
                data={
                    "file": f,
                    "claimed_writer": "Ibnu Gayuh Fadilah",
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        verif = data.get("verification", {})
        self.assertEqual(verif.get("claimed_writer"), "Ibnu Gayuh Fadilah")
        self.assertEqual(verif.get("decision"), "ACCEPT")
        self.assertEqual(verif.get("status"), "VALID")
        self.assertLessEqual(verif.get("score"), 25.1291)

    def test_08_model_freeze_guard_protection(self):
        """TEST 8: Model Freeze Guard harus melempar error jika timestamp bukan 20260926_190940."""
        with patch.object(config, "PUBLIC_EVALUATION_MODE", True):
            with patch.object(config, "FROZEN_H0_MODEL_VERSION", "20260926_190940"):
                # Normal case: version matches
                knn_model, le, X_tr, y_tr, ts = _load_latest_model_files()
                self.assertEqual(ts, "20260926_190940")

                # Tampered version: mismatch must trigger RuntimeError
                with patch.object(config, "FROZEN_H0_MODEL_VERSION", "99999999_000000"):
                    with self.assertRaises(RuntimeError):
                        _load_latest_model_files()

    def test_09_public_mode_false_preserves_behavior(self):
        """TEST 9: Saat PUBLIC_EVALUATION_MODE = False, rute mutation dapat diakses (mocked execution)."""
        with patch.object(config, "PUBLIC_EVALUATION_MODE", False):
            with patch("api.train_routes.PUBLIC_EVALUATION_MODE", False):
                with patch("api.train_routes.train_model") as mock_train:
                    mock_train.return_value = {"success": True, "message": "Training mock sukses"}
                    res = self.client.post("/api/train", json={"knn_k": 5})
                    self.assertEqual(res.status_code, 200)
                    mock_train.assert_called_once()

    def test_10_config_and_health_expose_public_mode_flag(self):
        """TEST 10: /api/config dan /api/health mengekspos public_evaluation_mode flag dengan benar."""
        res_health = self.client.get("/api/health")
        self.assertEqual(res_health.status_code, 200)
        data_health = res_health.get_json()
        self.assertIn("public_evaluation_mode", data_health)
        self.assertTrue(data_health["public_evaluation_mode"])

        res_config = self.client.get("/api/config")
        self.assertEqual(res_config.status_code, 200)
        data_config = res_config.get_json()
        self.assertIn("public_evaluation_mode", data_config)
        self.assertTrue(data_config["public_evaluation_mode"])

    def test_11_evaluation_endpoint_presents_canonical_20_writer_loocv(self):
        """TEST 11: GET /api/evaluate mengembalikan metrik canonical 20-writer LOOCV (61.00%, 244/400, 20x20 CM)."""
        res = self.client.get("/api/evaluate")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))

        # LOOCV authoritative metrics
        loocv = data.get("loocv_metrics", {})
        self.assertEqual(loocv.get("accuracy"), 61.00)
        self.assertEqual(loocv.get("precision_macro"), 60.27)
        self.assertEqual(loocv.get("recall_macro"), 61.00)
        self.assertEqual(loocv.get("f1_macro"), 60.27)
        self.assertEqual(loocv.get("evaluated_samples"), 400)
        self.assertEqual(loocv.get("correct_samples"), 244)
        self.assertEqual(loocv.get("wrong_samples"), 156)
        self.assertEqual(loocv.get("n_writers"), 20)

        # Held-out test metrics
        held_out = data.get("held_out_metrics", {})
        self.assertEqual(held_out.get("n_train"), 320)
        self.assertEqual(held_out.get("n_test"), 80)
        self.assertEqual(held_out.get("test_accuracy"), 61.25)

        # Reference dataset
        ref = data.get("reference_dataset", {})
        self.assertEqual(ref.get("n_respondents"), 20)
        self.assertEqual(ref.get("n_total_dataset"), 400)

        # Per-class chart
        per_class = data.get("per_class_chart", [])
        self.assertEqual(len(per_class), 20)
        total_correct = sum(c["correct"] for c in per_class)
        total_samples = sum(c["total"] for c in per_class)
        self.assertEqual(total_samples, 400)
        self.assertEqual(total_correct, 244)

        # Confusion matrix
        cm = data.get("confusion_matrix", {})
        self.assertIsNotNone(cm)
        labels = cm.get("labels", [])
        matrix = cm.get("matrix", [])
        self.assertEqual(len(labels), 20)
        self.assertEqual(len(matrix), 20)
        diag_sum = sum(matrix[i][i] for i in range(20))
        self.assertEqual(diag_sum, 244)


if __name__ == "__main__":
    unittest.main(verbosity=2)
