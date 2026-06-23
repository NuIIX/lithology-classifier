import unittest

import numpy as np
import pandas as pd

from src.data_preprocessor import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """
    Unit tests for the DataPreprocessor class.
    """

    def setUp(self):
        self.preprocessor = DataPreprocessor(
            step=0.2, impute=True, add_indicators=False
        )

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

    def test_preprocessor_missing_columns(self):
        """
        Verify preprocessor raises error when required columns are missing.
        """
        df = pd.DataFrame({"GR": [10.0, 20.0]})
        with self.assertRaises(ValueError):
            self.preprocessor.transform(df)

    def test_preprocessor_empty_well(self):
        """
        Verify preprocessor handles empty well DataFrame correctly.
        """
        df = pd.DataFrame(columns=["WELL", "DEPTH", "GR"])
        processed = self.preprocessor._resample_well(df)
        self.assertTrue(processed.empty)

    def test_preprocessor_invalid_depth_range(self):
        """
        Verify preprocessor handles start_depth > end_depth gracefully.
        """
        data = {
            "WELL": ["WELL_1"] * 2,
            "DEPTH": [100.01, 100.19],
            "GR": [10.0, 20.0],
        }
        df = pd.DataFrame(data)
        processed = self.preprocessor.fit_transform(df)
        self.assertEqual(len(processed), 0)

    def test_preprocessor_single_valid_value(self):
        """
        Verify preprocessor handles curve with single valid value correctly.
        """
        data = {
            "WELL": ["WELL_1"] * 3,
            "DEPTH": [100.0, 101.0, 102.0],
            "GR": [10.0, np.nan, np.nan],
        }
        df = pd.DataFrame(data)
        self.preprocessor.step = 1.0
        self.preprocessor.impute = False
        processed = self.preprocessor.fit_transform(df)
        np.testing.assert_allclose(processed["GR"].values, [10.0, 10.0, 10.0])

    def test_preprocessor_all_nan_lithology(self):
        """
        Verify preprocessor handles curve with all nan lithology correctly.
        """
        data = {
            "WELL": ["WELL_1"] * 3,
            "DEPTH": [100.0, 101.0, 102.0],
            "LITHOLOGY": [np.nan, np.nan, np.nan],
        }
        df = pd.DataFrame(data)
        self.preprocessor.step = 1.0
        processed = self.preprocessor.fit_transform(df)
        self.assertTrue(processed["LITHOLOGY"].isna().all())

    def test_preprocessor_impute_with_nan_global_median(self):
        """
        Verify imputation defaults to 0.0 when global median is NaN.
        """
        data = {
            "WELL": ["WELL_1"] * 3,
            "DEPTH": [100.0, 101.0, 102.0],
            "GR": [np.nan, np.nan, np.nan],
        }
        df = pd.DataFrame(data)
        self.preprocessor.step = 1.0
        self.preprocessor.impute = True
        processed = self.preprocessor.fit_transform(df)
        np.testing.assert_allclose(processed["GR"].values, [0.0, 0.0, 0.0])

    def test_preprocessor_duplicate_depth_aggregation(self):
        """
        Verify that duplicate depths are aggregated using first non-NaN value per column.
        """
        data = {
            "WELL": ["WELL_1"] * 3,
            "DEPTH": [100.0, 100.0, 101.0],
            "GR": [np.nan, 55.0, 60.0],
            "RHOB": [2.3, np.nan, 2.4],
        }
        df = pd.DataFrame(data)
        self.preprocessor.step = 1.0
        self.preprocessor.impute = False
        self.preprocessor.outlier_method = "physical"
        processed = self.preprocessor.fit_transform(df)

        self.assertEqual(len(processed), 2)
        np.testing.assert_allclose(processed["DEPTH"].values, [100.0, 101.0])
        np.testing.assert_allclose(processed["GR"].values, [55.0, 60.0])
        np.testing.assert_allclose(processed["RHOB"].values, [2.3, 2.4])

    def test_preprocessor_string_lithology(self):
        """
        Verify that preprocessor does not crash on string/object lithologies and processes them.
        """
        data = {
            "WELL": ["WELL_1"] * 3,
            "DEPTH": [100.0, 103.0, 106.0],
            "GR": [10.0, 20.0, 30.0],
            "LITHOLOGY": ["Sandstone", np.nan, "Shale"],
        }
        df = pd.DataFrame(data)
        self.preprocessor.step = 1.0
        processed = self.preprocessor.fit_transform(df)
        self.assertEqual(processed["LITHOLOGY"].iloc[0], "Sandstone")
        self.assertTrue(pd.isna(processed["LITHOLOGY"].iloc[3]))
        self.assertEqual(processed["LITHOLOGY"].iloc[6], "Shale")

    def test_preprocessor_lithology_tolerance_and_extrapolation(self):
        """
        Verify nearest-neighbor reindexing uses tolerance and suppresses extrapolation.
        """
        data = {
            "WELL": ["WELL_1"] * 3,
            "DEPTH": [100.0, 150.0, 200.0],
            "LITHOLOGY": [3000.0, np.nan, 4000.0],
        }
        df = pd.DataFrame(data)
        self.preprocessor.step = 10.0
        processed = self.preprocessor.fit_transform(df)

        self.assertEqual(
            processed[processed["DEPTH"] == 100.0]["LITHOLOGY"].iloc[0], 3000.0
        )
        self.assertEqual(
            processed[processed["DEPTH"] == 200.0]["LITHOLOGY"].iloc[0], 4000.0
        )
        self.assertTrue(
            pd.isna(processed[processed["DEPTH"] == 150.0]["LITHOLOGY"].iloc[0])
        )
        self.assertEqual(
            processed[processed["DEPTH"] == 110.0]["LITHOLOGY"].iloc[0], 3000.0
        )

    def test_preprocessor_empty_well_returns_correct_columns(self):
        """
        Verify that _resample_well returns an empty DataFrame with the correct schema
        when df is empty or when start_idx > end_idx.
        """
        df_empty = pd.DataFrame(columns=["WELL", "DEPTH", "GR", "LITHOLOGY"])
        resampled_empty = self.preprocessor._resample_well(df_empty)
        self.assertTrue(resampled_empty.empty)
        self.assertEqual(
            list(resampled_empty.columns), ["WELL", "DEPTH", "GR", "LITHOLOGY"]
        )

        df_invalid = pd.DataFrame(
            {
                "WELL": ["WELL_1"] * 2,
                "DEPTH": [100.01, 100.19],
                "GR": [10.0, 20.0],
                "LITHOLOGY": [3000, 3000],
            }
        )
        self.preprocessor.step = 0.2
        resampled_invalid = self.preprocessor._resample_well(df_invalid)
        self.assertTrue(resampled_invalid.empty)
        self.assertEqual(
            list(resampled_invalid.columns), ["WELL", "DEPTH", "GR", "LITHOLOGY"]
        )

    def test_preprocessor_inverted_clipping_bounds_fallback(self):
        """
        Verify that if quantile-based clipping bounds are inverted (lower > upper)
        due to out-of-bounds quantile ranges, the preprocessor falls back to
        physical bounds to prevent collapsing the entire column.
        """
        data_fit = {
            "WELL": ["WELL_1"] * 3,
            "DEPTH": [100.0, 101.0, 102.0],
            "RHOB": [0.45, 0.45, 0.45],
        }
        df_fit = pd.DataFrame(data_fit)

        data_trans = {
            "WELL": ["WELL_1"] * 3,
            "DEPTH": [100.0, 101.0, 102.0],
            "RHOB": [0.45, 2.65, 3.8],
        }
        df_trans = pd.DataFrame(data_trans)

        self.preprocessor.step = 1.0
        self.preprocessor.impute = False
        self.preprocessor.outlier_method = "clip_percentiles"

        self.preprocessor.fit(df_fit)

        self.assertEqual(self.preprocessor.clipping_bounds["RHOB"], (1.0, 3.5))

        processed = self.preprocessor.transform(df_trans)

        np.testing.assert_allclose(processed["RHOB"].values, [1.0, 2.65, 3.5])


if __name__ == "__main__":
    unittest.main()
