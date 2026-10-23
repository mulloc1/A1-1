"""로컬 H&M 원본을 사용하는 느린 통합 테스트. 명시적으로 실행한다."""

from __future__ import annotations

import gc
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.pipeline import DataAnalyzer


DATA_DIR = Path("data/h-and-m-personalized-fashion-recommendations")


@unittest.skipUnless((DATA_DIR / "transactions_train.csv").is_file(), "로컬 H&M 원본이 없습니다")
class RealHAndMTest(unittest.TestCase):
    def test_full_sample_images_and_repeatability(self) -> None:
        first = DataAnalyzer(DATA_DIR).load_data()
        self.assertEqual(len(first.raw_customers), 10_000)
        self.assertGreater(len(first.raw_transactions), 200_000)
        self.assertEqual(first.reference_date, pd.Timestamp("2020-09-23"))
        self.assertEqual(len(first.image_features), 1_500)
        self.assertEqual(first.image_arrays.shape, (1_500, 64, 64, 3))
        self.assertEqual(len(first.rfm), first.transactions["customer_id"].nunique())
        self.assertEqual(first.quality_report["image_errors"], [])
        self.assertIn("t_dat", first.raw_transactions.columns)
        self.assertTrue(first.raw_articles["article_id"].str.fullmatch(r"\d{10}").all())
        self.assertTrue(first.transactions["product_id"].str.fullmatch(r"\d{10}").all())
        self.assertGreaterEqual(first.raw_customers["age"].isna().sum(), first.customers["age"].isna().sum())

        ids = first.raw_customers["customer_id"].tolist()
        rfm = first.rfm.copy(deep=True)
        features = first.image_features.copy(deep=True)
        arrays = first.image_arrays.copy()
        transaction_count = len(first.raw_transactions)
        del first
        gc.collect()

        second = DataAnalyzer(DATA_DIR).load_data()
        self.assertEqual(second.raw_customers["customer_id"].tolist(), ids)
        self.assertEqual(len(second.raw_transactions), transaction_count)
        assert_frame_equal(second.rfm, rfm)
        assert_frame_equal(second.image_features, features)
        self.assertTrue(np.array_equal(second.image_arrays, arrays))


if __name__ == "__main__":
    unittest.main()
