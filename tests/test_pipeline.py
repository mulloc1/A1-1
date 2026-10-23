"""작은 H&M 형식 CSV로 파이프라인 계약을 검증한다."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.pipeline import DataAnalyzer


class PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data_dir = Path(self.temp.name)
        self.articles = pd.DataFrame({
            "article_id": ["0000000001", "0000000002", "0000000003"],
            "prod_name": ["Red top", "Blue jacket", "Plain shirt"],
            "product_group_name": ["Top", "Jacket", "Top"],
            "detail_desc": ["red", None, "plain"],
        })
        self.customers = pd.DataFrame({
            "customer_id": ["c1", "c2", "c3", "c4"],
            "club_member_status": ["A", "A", "B", "C"],
            "age": [20, None, None, 40],
        })
        self.transactions = pd.DataFrame({
            "t_dat": ["2020-09-20", "2020-09-20", "2020-09-22", "2020-09-21", "2020-09-21"],
            "customer_id": ["c1", "c1", "c1", "c3", "c4"],
            "article_id": ["0000000001", "0000000001", "0000000002", "0000000003", "0000000001"],
            "price": [0.1, 0.1, 0.2, 0.3, -1.0],
            "sales_channel_id": [1, 1, 2, 1, 1],
        })
        self._write_csv()
        for product_id in ("0000000001", "0000000002"):
            folder = self.data_dir / "images" / product_id[:3]
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{product_id}.jpg").touch()
        self.pixels = {
            "0000000001": np.full((4, 4, 3), 10, dtype=np.uint8),
            "0000000002": np.stack([
                np.full((4, 4), 0, dtype=np.uint8),
                np.full((4, 4), 20, dtype=np.uint8),
                np.full((4, 4), 40, dtype=np.uint8),
            ], axis=2),
        }

    def _write_csv(self) -> None:
        self.articles.to_csv(self.data_dir / "articles.csv", index=False)
        self.customers.to_csv(self.data_dir / "customers.csv", index=False)
        self.transactions.to_csv(self.data_dir / "transactions_train.csv", index=False)

    def analyzer(self, **options: object) -> DataAnalyzer:
        return DataAnalyzer(
            self.data_dir, customer_count=4, image_count=2, chunk_size=2,
            crop_shape=(4, 4), downsample=(2, 2),
            image_reader=lambda path: self.pixels[Path(path).stem],
            **options,
        )

    def test_load_preserves_ids_duplicates_and_originals(self) -> None:
        analyzer = self.analyzer().load_data()
        self.assertEqual(analyzer.reference_date, pd.Timestamp("2020-09-23"))
        self.assertEqual(len(analyzer.raw_transactions), 5)
        self.assertEqual(len(analyzer.transactions), 4)
        self.assertEqual(analyzer.quality_report["invalid_amount_rows"], 1)
        self.assertEqual(analyzer.transactions["product_id"].iloc[0], "0000000001")
        self.assertEqual(analyzer.transactions["product_id"].value_counts()["0000000001"], 2)
        self.assertIn("t_dat", analyzer.raw_transactions.columns)
        self.assertNotIn("order_date", analyzer.raw_transactions.columns)
        self.assertEqual(analyzer.raw_customers["age"].isna().sum(), 2)
        self.assertEqual(analyzer.customers["age"].isna().sum(), 0)
        self.assertEqual(analyzer.quality_report["age_imputation"]["global_filled"], 1)
        self.assertEqual(analyzer.products.set_index("product_id").loc["0000000001", "reference_price"], 0.1)
        self.assertEqual(analyzer.products.set_index("product_id").loc["0000000002", "product_name_word_count"], 2)

    def test_sampling_is_repeatable(self) -> None:
        options = {"customer_count": 3, "image_count": 0, "chunk_size": 2}
        first = DataAnalyzer(self.data_dir, **options).load_data(include_images=False)
        second = DataAnalyzer(self.data_dir, **options).load_data(include_images=False)
        assert_frame_equal(first.raw_customers, second.raw_customers)
        assert_frame_equal(first.raw_transactions, second.raw_transactions)
        assert_frame_equal(first.rfm, second.rfm)

    def test_missing_group_then_global_fallback(self) -> None:
        original = pd.DataFrame({"group": ["A", "A", "B", "C"], "value": [20.0, None, None, 40.0]})
        result, report = DataAnalyzer.handle_missing_values(original, "value", "group")
        self.assertEqual(result["value"].tolist(), [20.0, 20.0, 30.0, 40.0])
        self.assertEqual((report["group_filled"], report["global_filled"]), (1, 1))
        self.assertEqual(original["value"].isna().sum(), 2)
        mean_result, _ = DataAnalyzer.handle_missing_values(original, "value", "group", strategy="mean")
        self.assertEqual(mean_result.loc[2, "value"], 30.0)
        with self.assertRaisesRegex(ValueError, "유효한 값"):
            DataAnalyzer.handle_missing_values(pd.DataFrame({"g": ["A"], "v": [float("nan")]}), "v", "g")

    def test_iqr_detection_and_separate_treatment(self) -> None:
        original = pd.DataFrame({"amount": [1.0, 2.0, 2.0, 3.0, 100.0]})
        mask, bounds = DataAnalyzer.detect_outliers(original, "amount")
        self.assertEqual(mask.tolist(), [False, False, False, False, True])
        self.assertEqual(bounds["lower"], 0.5)
        self.assertEqual(bounds["upper"], 4.5)
        clipped, clip_report = DataAnalyzer.treat_outliers(original, "amount", method="clip")
        dropped, drop_report = DataAnalyzer.treat_outliers(original, "amount", method="drop")
        self.assertEqual(clipped["amount"].tolist(), [1.0, 2.0, 2.0, 3.0, 4.5])
        self.assertEqual(len(dropped), 4)
        self.assertEqual(clip_report["after_rows"], 5)
        self.assertEqual(drop_report["after_rows"], 4)
        self.assertEqual(original.iloc[-1, 0], 100.0)
        with self.assertRaisesRegex(ValueError, "iqr_multiplier"):
            DataAnalyzer.detect_outliers(original, "amount", iqr_multiplier=0)

    def test_image_features_are_per_product_and_vectorized(self) -> None:
        analyzer = self.analyzer().load_data()
        features = analyzer.image_features.set_index("product_id")
        self.assertEqual(analyzer.image_arrays.shape, (2, 2, 2, 3))
        self.assertEqual(features.loc["0000000001", "image_mean"], 10.0)
        self.assertEqual(features.loc["0000000001", "image_std"], 0.0)
        self.assertEqual(features.loc["0000000002", "image_mean"], 20.0)
        self.assertAlmostEqual(features.loc["0000000002", "image_std"], np.std([0, 20, 40]))
        self.assertEqual(features.loc["0000000002", "image_b_mean"], 40.0)
        self.assertTrue(pd.isna(analyzer.products.set_index("product_id").loc["0000000003", "image_mean"]))
        self.assertEqual(analyzer.products.set_index("product_id").loc["0000000001", "image_array"].shape, (2, 2, 3))

    def test_image_error_reports_product_id(self) -> None:
        pixels = self.pixels.copy()
        pixels["0000000002"] = np.zeros((4, 4), dtype=np.uint8)
        analyzer = self.analyzer()
        analyzer.image_reader = lambda path: pixels[Path(path).stem]
        with self.assertRaisesRegex(ValueError, "0000000002"):
            analyzer.load_data()
        self.assertEqual(analyzer.quality_report["image_errors"][0]["product_id"], "0000000002")

    def test_rfm_uses_distinct_days_and_unclipped_amounts(self) -> None:
        analyzer = self.analyzer().load_data()
        rfm = analyzer.rfm.set_index("customer_id")
        self.assertEqual((rfm.loc["c1", "recency_days"], rfm.loc["c1", "frequency_days"]), (1, 2))
        self.assertAlmostEqual(rfm.loc["c1", "monetary"], 0.4)
        self.assertEqual(rfm.loc["c1", "transaction_rows"], 3)
        self.assertEqual((rfm.loc["c3", "recency_days"], rfm.loc["c3", "frequency_days"]), (2, 1))
        self.assertNotIn("c4", rfm.index)
        changed = analyzer.calculate_rfm(reference_date="2020-09-24")
        self.assertEqual(changed.set_index("customer_id").loc["c1", "recency_days"], 2)
        with self.assertRaisesRegex(ValueError, "기준일"):
            analyzer.calculate_rfm(reference_date="2020-09-22")

    def test_bad_schema_and_unknown_product_fail(self) -> None:
        self.transactions = self.transactions.drop(columns=["price"])
        self._write_csv()
        with self.assertRaisesRegex(ValueError, "price"):
            self.analyzer().load_data(include_images=False)
        self.transactions["price"] = [0.1, 0.1, 0.2, 0.3, -1.0]
        self.transactions.loc[0, "article_id"] = "9999999999"
        self._write_csv()
        with self.assertRaisesRegex(ValueError, "9999999999"):
            self.analyzer().load_data(include_images=False)

    def test_invalid_date_is_counted_without_changing_reference_date(self) -> None:
        self.transactions.loc[len(self.transactions)] = [
            "not-a-date", "c1", "0000000001", 0.5, 1
        ]
        self._write_csv()
        analyzer = self.analyzer().load_data(include_images=False)
        self.assertEqual(analyzer.reference_date, pd.Timestamp("2020-09-23"))
        self.assertEqual(analyzer.quality_report["invalid_date_rows"], 1)
        self.assertEqual(len(analyzer.transactions), 4)


if __name__ == "__main__":
    unittest.main()
