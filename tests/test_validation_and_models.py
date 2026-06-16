import unittest

import numpy as np
import pandas as pd

from src.data.split import GreedyStratifiedGroupSplit, train_test_split_by_well
from src.models.baseline_lr import BaselineLogisticRegression
from src.models.random_forest import LithotypeRandomForest


class TestValidationAndModels(unittest.TestCase):
    """
    Unit tests for the validation splitting logic and ML models.
    """

    def setUp(self) -> None:

        self.dummy_data = {
            "WELL": (
                ["WELL_1"] * 20
                + ["WELL_2"] * 20
                + ["WELL_3"] * 20
                + ["WELL_4"] * 20
                + ["WELL_5"] * 20
            ),
            "DEPTH": list(np.linspace(100, 200, 20)) * 5,
            "GR": np.random.default_rng(42).normal(50, 15, 100),
            "RHOB": np.random.default_rng(42).normal(2.3, 0.2, 100),
            "NPHI": np.random.default_rng(42).normal(0.2, 0.05, 100),
            "DT": np.random.default_rng(42).normal(80, 10, 100),
            "SP": np.random.default_rng(42).normal(0, 50, 100),
            "RDEP": np.random.default_rng(42).uniform(0.1, 10, 100),
            "RMED": np.random.default_rng(42).uniform(0.1, 10, 100),
            "RSHA": np.random.default_rng(42).uniform(0.1, 10, 100),
            "LITHOLOGY": (
                [30000] * 15
                + [65000] * 5
                + [30000] * 10
                + [65000] * 10
                + [30000] * 18
                + [93000] * 2
                + [30000] * 18
                + [93000] * 2
                + [65000] * 20
            ),
        }
        self.df = pd.DataFrame(self.dummy_data)

    def test_greedy_stratified_group_split_no_leakage(self) -> None:
        """
        Verify that the custom greedy group-stratified splitter results in completely disjoint sets of wells
        between train and test splits, and both splits contain the rare class if possible.
        """
        train_df, test_df = train_test_split_by_well(
            self.df, test_size=0.4, random_state=42
        )

        train_wells = set(train_df["WELL"].unique())
        test_wells = set(test_df["WELL"].unique())

        self.assertEqual(train_wells.intersection(test_wells), set())

        all_wells = set(self.df["WELL"].unique())
        self.assertEqual(train_wells.union(test_wells), all_wells)

        self.assertIn(93000, train_df["LITHOLOGY"].values)
        self.assertIn(93000, test_df["LITHOLOGY"].values)

    def test_split_missing_columns(self) -> None:
        """
        Test that ValueError is raised when groups or target columns are missing.
        """
        splitter = GreedyStratifiedGroupSplit(test_size=0.2, random_state=42)
        with self.assertRaises(ValueError):
            splitter.split(self.df, groups_col="INVALID_WELL")

        with self.assertRaises(ValueError):
            splitter.split(self.df, target_col="INVALID_LITHOLOGY")

    def test_split_empty_dataframe(self) -> None:
        """
        Test that empty lists are returned for an empty DataFrame or one with only NaNs.
        """
        splitter = GreedyStratifiedGroupSplit(test_size=0.2, random_state=42)

        tr, ts = splitter.split(pd.DataFrame(columns=["WELL", "LITHOLOGY"]))
        self.assertEqual(tr, [])
        self.assertEqual(ts, [])

        nan_df = pd.DataFrame({"WELL": ["W1", "W2"], "LITHOLOGY": [np.nan, np.nan]})
        tr, ts = splitter.split(nan_df)
        self.assertEqual(tr, [])
        self.assertEqual(ts, [])

    def test_split_remaining_groups_and_no_unassigned(self) -> None:
        """
        Test that groups with no valid class labels or groups left over are correctly processed
        and assigned to train or test.
        """

        df_extra = self.df.copy()

        extra_rows = pd.DataFrame(
            {
                "WELL": ["WELL_6"] * 5 + ["WELL_7"] * 5,
                "DEPTH": [10, 20, 30, 40, 50] * 2,
                "GR": [50] * 10,
                "RHOB": [2.3] * 10,
                "NPHI": [0.2] * 10,
                "DT": [80] * 10,
                "SP": [0] * 10,
                "RDEP": [1.0] * 10,
                "RMED": [1.0] * 10,
                "RSHA": [1.0] * 10,
                "LITHOLOGY": [np.nan] * 10,
            }
        )
        df_combined = pd.concat([df_extra, extra_rows], ignore_index=True)

        splitter = GreedyStratifiedGroupSplit(test_size=0.2, random_state=42)
        train_wells, test_wells = splitter.split(df_combined)

        self.assertTrue("WELL_6" in train_wells or "WELL_6" in test_wells)
        self.assertTrue("WELL_7" in train_wells or "WELL_7" in test_wells)

    def test_baseline_logistic_regression(self) -> None:
        """
        Verify that BaselineLogisticRegression pipeline trains and predicts successfully.
        """
        X = self.df[["GR", "RHOB", "NPHI", "DT", "SP", "RDEP", "RMED", "RSHA"]]
        y = self.df["LITHOLOGY"]

        model = BaselineLogisticRegression(random_state=42)
        model.fit(X, y)

        preds = model.predict(X)
        self.assertEqual(len(preds), len(X))

        probs = model.predict_proba(X)
        self.assertEqual(probs.shape, (len(X), len(np.unique(y))))

    def test_lithotype_random_forest(self) -> None:
        """
        Verify that LithotypeRandomForest pipeline trains and predicts successfully.
        """
        X = self.df[["GR", "RHOB", "NPHI", "DT", "SP", "RDEP", "RMED", "RSHA"]]
        y = self.df["LITHOLOGY"]

        model = LithotypeRandomForest(random_state=42, n_estimators=10)
        model.fit(X, y)

        preds = model.predict(X)
        self.assertEqual(len(preds), len(X))

        probs = model.predict_proba(X)
        self.assertEqual(probs.shape, (len(X), len(np.unique(y))))


if __name__ == "__main__":
    unittest.main()
