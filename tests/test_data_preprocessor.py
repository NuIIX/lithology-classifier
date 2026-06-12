import unittest

import numpy as np
import pandas as pd

from src.data_preprocessor import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """
    Unit tests for the DataPreprocessor class.
    """

    def setUp(self):
        self.preprocessor = DataPreprocessor(step=0.2, impute=True, add_indicators=False)

    def test_depth_resampling_numeric_and_lithology(self):
        data = {
            "WELL": ["WELL_1"] * 3,
            "DEPTH": [100.0, 100.3, 100.6],
            "GR": [10.0, 20.0, 30.0],
            "LITHOLOGY": [1000, 2000, 3000],
        }
        df = pd.DataFrame(data)

        self.preprocessor.outlier_method = "physical"
        processed = self.preprocessor.fit_transform(df)

        expected_depths = [100.0, 100.2, 100.4, 100.6]
        np.testing.assert_allclose(processed["DEPTH"].values, expected_depths)

        expected_gr = [10.0, 16.666667, 23.333333, 30.0]
        np.testing.assert_allclose(processed["GR"].values, expected_gr, rtol=1e-5)

        expected_lith = [1000, 2000, 2000, 3000]
        np.testing.assert_array_equal(processed["LITHOLOGY"].values, expected_lith)

    def test_outlier_clipping(self):
        data = {
            "WELL": ["WELL_1"] * 5,
            "DEPTH": [100.0, 101.0, 102.0, 103.0, 104.0],
            "GR": [10.0, 20.0, 600.0, 15.0, 25.0],
            "RHOB": [2.0, 0.5, 2.5, 2.2, 4.0],
        }
        df = pd.DataFrame(data)

        self.preprocessor.outlier_method = "physical"
        processed = self.preprocessor.fit_transform(df)

        self.assertTrue((processed["GR"] <= 500.0).all())
        self.assertTrue((processed["RHOB"] >= 1.0).all())
        self.assertTrue((processed["RHOB"] <= 3.5).all())

    def test_missing_log_imputation(self):
        data_train = {
            "WELL": ["WELL_1"] * 3 + ["WELL_2"] * 3,
            "DEPTH": [100.0, 101.0, 102.0] * 2,
            "GR": [10.0, 20.0, 30.0, np.nan, np.nan, np.nan],
            "RHOB": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
        }
        df_train = pd.DataFrame(data_train)

        self.preprocessor.step = 1.0
        self.preprocessor.impute = True
        self.preprocessor.add_indicators = True

        processed = self.preprocessor.fit_transform(df_train)

        expected_median_gr = 20.0
        well2_gr = processed[processed["WELL"] == "WELL_2"]["GR"]
        np.testing.assert_allclose(well2_gr.values, [expected_median_gr] * 3)

        self.assertIn("GR_isnull", processed.columns)
        well2_indicator = processed[processed["WELL"] == "WELL_2"]["GR_isnull"]
        np.testing.assert_array_equal(well2_indicator.values, [1, 1, 1])

        well1_indicator = processed[processed["WELL"] == "WELL_1"]["GR_isnull"]
        np.testing.assert_array_equal(well1_indicator.values, [0, 0, 0])

    def test_no_imputation_option(self):
        data = {
            "WELL": ["WELL_1"] * 3 + ["WELL_2"] * 3,
            "DEPTH": [100.0, 101.0, 102.0] * 2,
            "GR": [10.0, 20.0, 30.0, np.nan, np.nan, np.nan],
            "RHOB": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
        }
        df = pd.DataFrame(data)

        self.preprocessor.step = 1.0
        self.preprocessor.impute = False
        processed = self.preprocessor.fit_transform(df)

        well2_gr = processed[processed["WELL"] == "WELL_2"]["GR"]
        self.assertTrue(well2_gr.isna().all())


if __name__ == "__main__":
    unittest.main()
