import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_PHYSICAL_BOUNDS = {
    "GR": (0.0, 500.0),
    "RHOB": (1.0, 3.5),
    "NPHI": (-0.1, 1.0),
    "DT": (30.0, 240.0),
    "SP": (-500.0, 500.0),
    "RDEP": (0.01, 10000.0),
    "RMED": (0.01, 10000.0),
    "RSHA": (0.01, 10000.0),
}


class DataPreprocessor:
    """
    Preprocesses well log data including depth synchronization, outlier clipping, and missing value imputation.
    """

    def __init__(
        self,
        step: float = 0.152,
        outlier_method: str = "clip_percentiles",
        lower_quantile: float = 0.001,
        upper_quantile: float = 0.999,
        impute: bool = True,
        add_indicators: bool = False,
    ):
        self.step = step
        self.outlier_method = outlier_method
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.impute = impute
        self.add_indicators = add_indicators
        self.global_medians: dict[str, float] = {}
        self.clipping_bounds: dict[str, tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """
        Fits the preprocessor using the global training statistics.

        Args:
            df (pd.DataFrame): Training DataFrame containing standardized curves.

        Returns:
            DataPreprocessor: The fitted instance.
        """
        numeric_cols = [
            col
            for col in ["GR", "RHOB", "NPHI", "DT", "SP", "RDEP", "RMED", "RSHA"]
            if col in df.columns
        ]

        for col in numeric_cols:
            self.global_medians[col] = df[col].median()
            if self.outlier_method == "clip_percentiles":
                q_low = df[col].quantile(self.lower_quantile)
                q_high = df[col].quantile(self.upper_quantile)

                phys_low, phys_high = DEFAULT_PHYSICAL_BOUNDS.get(
                    col, (-np.inf, np.inf)
                )
                lower_bound = max(q_low, phys_low) if not np.isnan(q_low) else phys_low
                upper_bound = (
                    min(q_high, phys_high) if not np.isnan(q_high) else phys_high
                )

                if lower_bound > upper_bound:
                    logger.warning(
                        "Inverted clipping bounds for column '%s': lower (%s) > upper (%s). "
                        "Falling back to physical bounds (%s, %s).",
                        col,
                        lower_bound,
                        upper_bound,
                        phys_low,
                        phys_high,
                    )
                    lower_bound, upper_bound = phys_low, phys_high

                self.clipping_bounds[col] = (lower_bound, upper_bound)
            else:
                self.clipping_bounds[col] = DEFAULT_PHYSICAL_BOUNDS.get(
                    col, (-np.inf, np.inf)
                )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by performing depth synchronization, clipping outliers,
        and optionally imputing missing values.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        if "WELL" not in df.columns or "DEPTH" not in df.columns:
            raise ValueError("Input DataFrame must contain 'WELL' and 'DEPTH' columns.")

        processed_wells = []
        wells = df["WELL"].unique()

        for well in wells:
            well_df = df[df["WELL"] == well].copy()
            well_df = self._resample_well(well_df)
            processed_wells.append(well_df)

        combined = pd.concat(processed_wells, ignore_index=True)

        numeric_cols = [
            col
            for col in ["GR", "RHOB", "NPHI", "DT", "SP", "RDEP", "RMED", "RSHA"]
            if col in combined.columns
        ]

        for col in numeric_cols:
            if col in self.clipping_bounds:
                low, high = self.clipping_bounds[col]
                combined[col] = combined[col].clip(lower=low, upper=high)

        if self.add_indicators:
            for col in numeric_cols:
                combined[f"{col}_isnull"] = combined[col].isna().astype(int)

        if self.impute:
            for col in numeric_cols:
                fill_val = self.global_medians.get(col, 0.0)
                if np.isnan(fill_val):
                    fill_val = 0.0
                combined[col] = combined[col].fillna(fill_val)

        return combined

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the preprocessor and transforms the data.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        return self.fit(df).transform(df)

    def _resample_well(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resamples a single well's curves onto a uniform depth grid.
        """
        numeric_cols_standard = [
            "GR",
            "RHOB",
            "NPHI",
            "DT",
            "SP",
            "RDEP",
            "RMED",
            "RSHA",
        ]
        numeric_cols = [col for col in numeric_cols_standard if col in df.columns]

        if df.empty:
            empty_cols = ["WELL", "DEPTH", *numeric_cols]
            if "LITHOLOGY" in df.columns:
                empty_cols.append("LITHOLOGY")
            return pd.DataFrame(columns=empty_cols)

        well_name = df["WELL"].iloc[0]
        min_depth = df["DEPTH"].min()
        max_depth = df["DEPTH"].max()
        start_idx = int(np.ceil(round(min_depth / self.step, 9)))
        end_idx = int(np.floor(round(max_depth / self.step, 9)))

        if start_idx > end_idx:
            empty_cols = ["WELL", "DEPTH", *numeric_cols]
            if "LITHOLOGY" in df.columns:
                empty_cols.append("LITHOLOGY")
            return pd.DataFrame(columns=empty_cols)

        grid = np.round(np.arange(start_idx, end_idx + 1) * self.step, 9)

        df_sorted = df.copy()
        df_sorted["DEPTH"] = np.round(df_sorted["DEPTH"].values, 9)
        df_sorted = df_sorted.groupby("DEPTH", as_index=False).first()
        df_sorted = df_sorted.sort_values("DEPTH")

        original_depths = df_sorted["DEPTH"].values

        new_data = {
            "WELL": [well_name] * len(grid),
            "DEPTH": grid,
        }

        for col in numeric_cols:
            vals = df_sorted[col].values
            valid_mask = pd.notna(vals)

            if valid_mask.sum() > 1:
                new_vals = np.interp(
                    grid,
                    original_depths[valid_mask],
                    vals[valid_mask],
                    left=np.nan,
                    right=np.nan,
                )
                new_data[col] = new_vals
            elif valid_mask.sum() == 1:
                new_data[col] = np.full_like(grid, vals[valid_mask][0])
            else:
                new_data[col] = np.full_like(grid, np.nan)

        if "LITHOLOGY" in df_sorted.columns:
            lith_vals = df_sorted["LITHOLOGY"].values
            valid_mask = pd.notna(lith_vals)

            if valid_mask.any():
                df_lith = df_sorted[["DEPTH", "LITHOLOGY"]].dropna().set_index("DEPTH")
                min_lith_depth = df_lith.index.min()
                max_lith_depth = df_lith.index.max()
                df_reindexed = df_lith.reindex(
                    grid, method="nearest", tolerance=self.step * 1.5
                )
                lith_values = df_reindexed["LITHOLOGY"].values

                if not (
                    np.issubdtype(lith_values.dtype, np.floating)
                    or lith_values.dtype == object
                ):
                    lith_values = lith_values.astype(object)

                extrapolation_mask = (grid < min_lith_depth) | (grid > max_lith_depth)
                lith_values[extrapolation_mask] = np.nan

                new_data["LITHOLOGY"] = lith_values
            else:
                new_data["LITHOLOGY"] = np.full_like(grid, np.nan)

        return pd.DataFrame(new_data)
