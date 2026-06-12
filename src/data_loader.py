import contextlib
import logging
import os

import lasio
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MNEMONIC_MAP = {
    "DEPTH": ["DEPTH", "DEPT", "DEPTH_MD", "DEPTH:1", "DEPTH:2"],
    "GR": ["GR", "GRD", "SGR", "GRS", "GRCO"],
    "RHOB": ["RHOB", "DEN", "ZDEN", "RHOZ", "RHO8"],
    "NPHI": ["NPHI", "NPHI_LS", "CNC", "CNPOR", "NPOR"],
    "DT": ["DT", "DTC", "DTCO", "AC", "DTLN"],
    "SP": ["SP", "SPD", "SPCG"],
    "RDEP": ["RDEP", "RT", "RD", "ILD", "LLD", "AHTT"],
    "RMED": ["RMED", "RM", "ILM", "LLS", "SN", "AHT30"],
    "RSHA": ["RSHA", "RXO", "RS", "MSFL", "SFL", "AHT10", "AHT20"],
    "LITHOLOGY": [
        "FORCE_2020_LITHOFACIES_LITHOLOGY",
        "LITHOLOGY",
        "LITHOFACIES",
        "CLASS",
    ],
}


class DataLoader:
    """
    A flexible data loader class for reading petrophysical logs from CSV and LAS formats.

    Standardizes curve mnemonics, handles missing values, and merges them into a
    consistent DataFrame format for modeling.
    """

    def __init__(self, mnemonic_map: dict[str, list[str]] | None = None):
        """
        Initializes the DataLoader with a mnemonic mapping dictionary.

        Args:
            mnemonic_map (Optional[Dict[str, List[str]]]): Custom dictionary mapping standard curve names
                to lists of synonyms. If None, DEFAULT_MNEMONIC_MAP is used.
        """
        self.mnemonic_map = mnemonic_map or DEFAULT_MNEMONIC_MAP
        self._standardized_map = {
            k.upper(): [v.upper() for v in values]
            for k, values in self.mnemonic_map.items()
        }

    def _map_columns(self, df: pd.DataFrame, keep_other: bool = False) -> pd.DataFrame:
        """
        Renames DataFrame columns to standard names using the mnemonic map.

        Args:
            df (pd.DataFrame): Input DataFrame with raw column/mnemonic names.
            keep_other (bool): Whether to retain non-standard columns in the final DataFrame.

        Returns:
            pd.DataFrame: DataFrame with renamed standard columns. Missing standard columns
            are populated with NaN.
        """
        df_cols_upper = {col.upper(): col for col in df.columns}
        rename_dict = {}

        for std_name, synonyms in self._standardized_map.items():
            for synonym in synonyms:
                if synonym in df_cols_upper:
                    source_col = df_cols_upper[synonym]
                    rename_dict[source_col] = std_name
                    break

        df_renamed = df.rename(columns=rename_dict)

        standard_cols = [
            "WELL",
            "DEPTH",
            "GR",
            "RHOB",
            "NPHI",
            "DT",
            "SP",
            "RDEP",
            "RMED",
            "RSHA",
        ]

        has_lithology = (
            "LITHOLOGY" in rename_dict.values() or "LITHOLOGY" in df_renamed.columns
        )
        if has_lithology:
            standard_cols.append("LITHOLOGY")

        for col in standard_cols:
            if col not in df_renamed.columns:
                df_renamed[col] = np.nan

        if not keep_other:
            df_renamed = df_renamed[standard_cols]

        return df_renamed

    def load_csv(
        self,
        file_path: str,
        sep: str = ";",
        wells: list[str] | None = None,
        keep_other: bool = False,
    ) -> pd.DataFrame:
        """
        Loads well log data from a CSV file.

        Args:
            file_path (str): Path to the CSV file.
            sep (str): Delimiter used in the CSV file. Default is ';'.
            wells (Optional[List[str]]): List of specific well names to filter. If None, all wells are loaded.
            keep_other (bool): Whether to retain non-standard columns in the loaded DataFrame.

        Returns:
            pd.DataFrame: A pandas DataFrame containing standardized curve columns.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty or cannot be parsed.
        """
        logger.info(f"Loading CSV data from {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        try:
            df = pd.read_csv(file_path, sep=sep)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV file {file_path}: {e}") from e

        if df.empty:
            raise ValueError(f"CSV file is empty: {file_path}")

        df_standardized = self._map_columns(df, keep_other=keep_other)

        if wells is not None:
            if (
                "WELL" in df_standardized.columns
                and not df_standardized["WELL"].isna().all()
            ):
                df_standardized = df_standardized[
                    df_standardized["WELL"].isin(wells)
                ].copy()
            else:
                logger.warning(
                    "WELL column not found or empty. Skipping well filtering."
                )

        return df_standardized

    def load_las(
        self,
        file_path: str,
        null_values: list[float] | None = None,
        keep_other: bool = False,
    ) -> pd.DataFrame:
        """
        Loads well log data from a single LAS file.

        Args:
            file_path (str): Path to the LAS file.
            null_values (Optional[List[float]]): List of additional values to treat as NaN (e.g., -999.25).
                If None, the default null values read from the LAS file headers are used.
            keep_other (bool): Whether to retain non-standard columns in the loaded DataFrame.

        Returns:
            pd.DataFrame: A pandas DataFrame containing standardized curve columns.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be parsed by lasio.
        """
        logger.info(f"Loading LAS data from {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"LAS file not found: {file_path}")

        try:
            las = lasio.read(file_path)
        except Exception as e:
            raise ValueError(
                f"Failed to parse LAS file {file_path} using lasio: {e}"
            ) from e

        well_name = "UNKNOWN"
        if "WELL" in las.well:
            well_val = las.well["WELL"].value
            if well_val and str(well_val).strip():
                well_name = str(well_val).strip()

        df = las.df().reset_index()

        target_nulls = [-999.25]
        if "NULL" in las.well:
            with contextlib.suppress(ValueError, TypeError):
                target_nulls.append(float(las.well["NULL"].value))
        if null_values:
            target_nulls.extend(null_values)

        target_nulls = list(set(target_nulls))

        for val in target_nulls:
            df = df.replace(val, np.nan)
            df = df.apply(
                lambda col, val=val: col.map(
                    lambda x, val=val: (
                        np.nan
                        if isinstance(x, (int, float)) and np.isclose(x, val, atol=1e-4)
                        else x
                    )
                )
            )

        df["WELL"] = well_name

        df_standardized = self._map_columns(df, keep_other=keep_other)

        return df_standardized

    def load_las_directory(
        self,
        dir_path: str,
        null_values: list[float] | None = None,
        keep_other: bool = False,
    ) -> pd.DataFrame:
        """
        Loads and merges all LAS files in a given directory into a single DataFrame.

        Args:
            dir_path (str): Path to the directory containing LAS files.
            null_values (Optional[List[float]]): List of additional values to treat as NaN.
            keep_other (bool): Whether to retain non-standard columns in the loaded DataFrame.

        Returns:
            pd.DataFrame: Combined pandas DataFrame with standardized curves from all loaded wells.

        Raises:
            FileNotFoundError: If the directory does not exist.
            ValueError: If no valid LAS files are found or loaded.
        """
        logger.info(f"Loading all LAS files from directory {dir_path}")
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        las_files = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith(".las")
        ]
        if not las_files:
            raise ValueError(f"No LAS files found in directory: {dir_path}")

        dfs = []
        for file_path in las_files:
            try:
                df = self.load_las(
                    file_path, null_values=null_values, keep_other=keep_other
                )
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        if not dfs:
            raise ValueError(f"No LAS files were successfully loaded from: {dir_path}")

        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
