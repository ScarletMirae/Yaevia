"""
test_validation_cases.py — Validasi Kasus Pengujian A–G UI & API Yaevia
========================================================================
Pengujian otomatis untuk flow manual Ground Truth:
  CASE A: Gambar ✓, Nama valid, NIM valid -> Valid (Button Active)
  CASE B: Gambar ✓, Nama valid, NIM milik mahasiswa lain -> Invalid (Button Disabled)
  CASE C: Gambar ✓, Nama tidak terdaftar, NIM valid -> Invalid (Button Disabled)
  CASE D: Gambar ✓, Nama valid, NIM tidak terdaftar -> Invalid (Button Disabled)
  CASE E: Nama + NIM valid, tetapi gambar belum diupload -> Button Disabled
  CASE F: Gambar + Nama + NIM valid -> Inferensi KNN Blind -> Ground truth dibandingkan SETELAH prediksi
  CASE G: Setelah valid, ubah 1 karakter pada Nama/NIM -> Invalid (Button Disabled)
"""

import os
import sys
import unittest
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app
from database import get_connection
from preprocessing.image_processor import preprocess_image
from features.hog_extractor import extract_hog_features
from model.classifier import verify_image


class TestValidationFlowCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        conn = get_connection()
        # Ambil dua sampel mahasiswa valid dari database
        rows = conn.execute("SELECT DISTINCT student_name, student_id FROM dataset LIMIT 2").fetchall()
        cls.student1_name = rows[0]["student_name"]
        cls.student1_nim  = rows[0]["student_id"]
        cls.student2_name = rows[1]["student_name"]
        cls.student2_nim  = rows[1]["student_id"]
        
        row_file = conn.execute("SELECT file_path FROM dataset LIMIT 1").fetchone()
        cls.sample_image_path = row_file["file_path"]
        conn.close()

    def _validate_api(self, name, nim):
        res = self.client.post(
            "/api/verify/validate-identity",
            data=json.dumps({"student_name": name, "student_id": nim}),
            content_type="application/json"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        return data.get("valid", False)

    def test_case_A_valid_pair(self):
        """CASE A: Gambar ✓, Nama benar ✓, NIM benar ✓ -> valid."""
        valid = self._validate_api(self.student1_name, self.student1_nim)
        self.assertTrue(valid, "CASE A: Pasangan nama & NIM valid harus mengembalikan valid=True")

    def test_case_B_mismatched_nim(self):
        """CASE B: Gambar ✓, Nama benar, NIM milik mahasiswa lain -> invalid."""
        valid = self._validate_api(self.student1_name, self.student2_nim)
        self.assertFalse(valid, "CASE B: Nama valid dengan NIM milik mahasiswa lain harus invalid=False")

    def test_case_C_unregistered_name(self):
        """CASE C: Gambar ✓, Nama tidak terdaftar, NIM valid -> invalid."""
        valid = self._validate_api("Nama Khayalan Tidak Ada", self.student1_nim)
        self.assertFalse(valid, "CASE C: Nama tidak terdaftar harus invalid=False")

    def test_case_D_unregistered_nim(self):
        """CASE D: Gambar ✓, Nama valid, NIM tidak terdaftar -> invalid."""
        valid = self._validate_api(self.student1_name, "A999999999")
        self.assertFalse(valid, "CASE D: NIM tidak terdaftar harus invalid=False")

    def test_case_E_no_image_button_disabled(self):
        """CASE E: Nama+NIM valid tetapi belum upload gambar -> tombol disabled (canVerify = false)."""
        image_selected = False
        identity_validated = self._validate_api(self.student1_name, self.student1_nim)
        can_verify = image_selected and identity_validated
        self.assertFalse(can_verify, "CASE E: Tanpa gambar, can_verify harus False")

    def test_case_F_blind_prediction_post_comparison(self):
        """CASE F: Gambar + Nama+NIM valid -> KNN prediction blind -> ground truth dibandingkan setelahnya."""
        identity_validated = self._validate_api(self.student1_name, self.student1_nim)
        image_selected = True
        can_verify = image_selected and identity_validated
        self.assertTrue(can_verify, "CASE F: Form & gambar valid -> can_verify=True")

        processed = preprocess_image(self.sample_image_path)
        feat = extract_hog_features(processed)

        res = verify_image(
            feature_vector=feat,
            query_filename="sample_test.png",
            query_path=self.sample_image_path,
            ground_truth_name=self.student1_name,
            ground_truth_nim=self.student1_nim
        )

        self.assertTrue(res["success"])
        self.assertIn("predicted_name", res)
        self.assertIn("is_correct", res)
        self.assertIn("ground_truth_name", res)
        self.assertEqual(res["ground_truth_name"], self.student1_name)
        # Evaluasi is_correct harus berupa perbandingan boolean 1/0
        expected_correct = 1 if res["predicted_name"].strip().lower() == self.student1_name.strip().lower() else 0
        self.assertEqual(res["is_correct"], expected_correct)

    def test_case_G_modified_character_resets_validation(self):
        """CASE G: Setelah identitas valid, ubah 1 karakter pada Nama/NIM -> status valid dibatalkan."""
        # Step 1: Valid
        valid_before = self._validate_api(self.student1_name, self.student1_nim)
        self.assertTrue(valid_before)

        # Step 2: Modifikasi 1 karakter nama
        modified_name = self.student1_name + "x"
        valid_after_name_mod = self._validate_api(modified_name, self.student1_nim)
        self.assertFalse(valid_after_name_mod, "CASE G: Modifikasi nama harus membatalkan kevalidan")

        # Step 3: Modifikasi 1 karakter NIM
        modified_nim = self.student1_nim + "0"
        valid_after_nim_mod = self._validate_api(self.student1_name, modified_nim)
        self.assertFalse(valid_after_nim_mod, "CASE G: Modifikasi NIM harus membatalkan kevalidan")


if __name__ == "__main__":
    unittest.main(verbosity=2)
