import os
import tempfile
import unittest

import lasio
import numpy as np
import pandas as pd

from src.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.temp_files = []
        self.temp_dirs = []
        self.loader = DataLoader()

    def tearDown(self):
        for f in self.temp_files:
            if os.path.exists(f):
                os.remove(f)
        for d in self.temp_dirs:
            if os.path.exists(d):
                for f in os.listdir(d):
                    os.remove(os.path.join(d, f))
                os.rmdir(d)

    def test_load_csv_basic(self):
        data = (
            "WELL;DEPTH_MD;GR;RHOB;FORCE_2020_LITHOFACIES_LITHOLOGY\n"
            "WELL_A;1000.0;45.2;2.3;3000\n"
            "WELL_A;1001.0;48.5;2.4;3000\n"
            "WELL_B;2000.0;110.1;2.1;6000\n"
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write(data)
            temp_csv = f.name
            self.temp_files.append(temp_csv)

        df = self.loader.load_csv(temp_csv, sep=";")

        expected_cols = [
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
            "LITHOLOGY",
        ]
        self.assertEqual(list(df.columns), expected_cols)

        self.assertEqual(len(df), 3)

        self.assertTrue(np.allclose(df["DEPTH"], [1000.0, 1001.0, 2000.0]))
        self.assertTrue(np.allclose(df["LITHOLOGY"], [3000, 3000, 6000]))

        self.assertTrue(df["NPHI"].isna().all())
        self.assertTrue(df["DT"].isna().all())

    def test_load_csv_well_filter(self):
        data = "WELL;DEPTH_MD;GR\nWELL_A;1000.0;45.2\nWELL_B;2000.0;110.1\n"

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write(data)
            temp_csv = f.name
            self.temp_files.append(temp_csv)

        df = self.loader.load_csv(temp_csv, sep=";", wells=["WELL_B"])

        self.assertEqual(len(df), 1)
        self.assertEqual(df["WELL"].iloc[0], "WELL_B")
        self.assertEqual(df["GR"].iloc[0], 110.1)

    def test_load_las_basic(self):
        las = lasio.LASFile()
        las.well["WELL"] = lasio.HeaderItem("WELL", value="TEST_WELL_1")
        las.well["NULL"] = lasio.HeaderItem("NULL", value="-999.25")

        las.append_curve("DEPT", [100.0, 100.5, 101.0], unit="M")
        las.append_curve("GRD", [50.0, 52.3, -999.25], unit="GAPI")
        las.append_curve("DEN", [2.2, np.nan, 2.4], unit="G/CC")

        with tempfile.NamedTemporaryFile(suffix=".las", delete=False, mode="w") as f:
            las.write(f)
            temp_las = f.name
            self.temp_files.append(temp_las)

        df = self.loader.load_las(temp_las)

        expected_cols = [
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
        self.assertEqual(list(df.columns), expected_cols)

        self.assertEqual(len(df), 3)
        self.assertEqual(df["WELL"].iloc[0], "TEST_WELL_1")

        self.assertTrue(np.allclose(df["DEPTH"], [100.0, 100.5, 101.0]))

        self.assertEqual(df["GR"].iloc[0], 50.0)
        self.assertEqual(df["GR"].iloc[1], 52.3)
        self.assertTrue(np.isnan(df["GR"].iloc[2]))
        self.assertTrue(np.isnan(df["RHOB"].iloc[1]))

    def test_load_las_directory(self):
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)

        for i, name in enumerate(["WELL_X", "WELL_Y"]):
            las = lasio.LASFile()
            las.well["WELL"] = lasio.HeaderItem("WELL", value=name)
            las.append_curve("DEPTH", [100.0 + i], unit="M")
            las.append_curve("GR", [45.0 + i * 10], unit="GAPI")

            file_path = os.path.join(temp_dir, f"{name}.las")
            with open(file_path, "w") as f:
                las.write(f)

        df = self.loader.load_las_directory(temp_dir)

        self.assertEqual(len(df), 2)
        self.assertEqual(set(df["WELL"]), {"WELL_X", "WELL_Y"})
        self.assertEqual(df[df["WELL"] == "WELL_X"]["GR"].iloc[0], 45.0)
        self.assertEqual(df[df["WELL"] == "WELL_Y"]["GR"].iloc[0], 55.0)

    def test_custom_mnemonic_map(self):
        custom_map = {"DEPTH": ["D"], "GR": ["GAMMA_RAY"]}
        custom_loader = DataLoader(mnemonic_map=custom_map)

        data = "WELL;D;GAMMA_RAY\nWELL_X;10.0;99.9\n"
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write(data)
            temp_csv = f.name
            self.temp_files.append(temp_csv)

        df = custom_loader.load_csv(temp_csv)
        self.assertEqual(df["DEPTH"].iloc[0], 10.0)
        self.assertEqual(df["GR"].iloc[0], 99.9)

    def test_config_import(self):
        """
        Verify config constants are correctly defined.
        """
        from src.config import RANDOM_STATE

        self.assertEqual(RANDOM_STATE, 42)

    def test_load_csv_errors(self):
        """
        Verify that CSV loading handles errors correctly.
        """
        with self.assertRaises(FileNotFoundError):
            self.loader.load_csv("non_existent_file.csv")

        empty_csv = os.path.join(tempfile.gettempdir(), "empty.csv")
        with open(empty_csv, "w") as f:
            pass
        self.temp_files.append(empty_csv)
        with self.assertRaises(ValueError):
            self.loader.load_csv(empty_csv)

        header_only_csv = os.path.join(tempfile.gettempdir(), "header_only.csv")
        with open(header_only_csv, "w") as f:
            f.write("WELL;DEPTH_MD;GR\n")
        self.temp_files.append(header_only_csv)
        with self.assertRaises(ValueError):
            self.loader.load_csv(header_only_csv)

        no_well_csv = os.path.join(tempfile.gettempdir(), "no_well.csv")
        with open(no_well_csv, "w") as f:
            f.write("DEPTH_MD;GR\n1000.0;45.2\n")
        self.temp_files.append(no_well_csv)
        df = self.loader.load_csv(no_well_csv, wells=["WELL_A"])
        self.assertEqual(len(df), 1)

    def test_load_las_with_null_values(self):
        """
        Verify load_las handles custom null values correctly.
        """
        las = lasio.LASFile()
        las.well["WELL"] = lasio.HeaderItem("WELL", value="TEST_WELL_1")
        las.append_curve("DEPT", [100.0, 101.0], unit="M")
        las.append_curve("GR", [50.0, -9999.0], unit="GAPI")
        with tempfile.NamedTemporaryFile(suffix=".las", delete=False, mode="w") as f:
            las.write(f)
            temp_las = f.name
            self.temp_files.append(temp_las)
        df = self.loader.load_las(temp_las, null_values=[-9999.0])
        self.assertTrue(np.isnan(df["GR"].iloc[1]))

    def test_load_las_errors(self):
        """
        Verify that LAS loading handles errors correctly.
        """
        with self.assertRaises(FileNotFoundError):
            self.loader.load_las("non_existent_file.las")

        corrupt_las = os.path.join(tempfile.gettempdir(), "corrupt.las")
        with open(corrupt_las, "w") as f:
            f.write("this is not a valid las file structure\n")
        self.temp_files.append(corrupt_las)
        with self.assertRaises(ValueError):
            self.loader.load_las(corrupt_las)

        las = lasio.LASFile()
        las.well["WELL"] = lasio.HeaderItem("WELL", value="TEST")
        las.append_curve("GR", [50.0], unit="GAPI")
        no_depth_las = os.path.join(tempfile.gettempdir(), "no_depth.las")
        with open(no_depth_las, "w") as f:
            las.write(f)
        self.temp_files.append(no_depth_las)
        with self.assertRaises(ValueError):
            self.loader.load_las(no_depth_las)

    def test_load_las_directory_errors(self):
        """
        Verify directory loaders handle skipped files and missing directories correctly.
        """
        with self.assertRaises(FileNotFoundError):
            self.loader.load_las_directory("non_existent_directory")

        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)

        txt_file = os.path.join(temp_dir, "readme.txt")
        with open(txt_file, "w") as f:
            f.write("readme content")

        with self.assertRaises(ValueError):
            self.loader.load_las_directory(temp_dir)

        corrupt_las = os.path.join(temp_dir, "corrupt.las")
        with open(corrupt_las, "w") as f:
            f.write("invalid las structure")

        with self.assertRaises(ValueError):
            self.loader.load_las_directory(temp_dir)

    def test_load_csv_duplicate_columns(self):
        """
        Verify that duplicate columns (e.g. LITHOLOGY and FORCE_2020_LITHOFACIES_LITHOLOGY)
        are handled gracefully without raising duplicate column exceptions or duplicate headers.
        """
        data = (
            "WELL;DEPTH_MD;FORCE_2020_LITHOFACIES_LITHOLOGY;LITHOLOGY;LITHOLOGY_RAW\n"
            "WELL_A;1000.0;3000;4000;5000\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write(data)
            temp_csv = f.name
            self.temp_files.append(temp_csv)

        df = self.loader.load_csv(temp_csv, keep_other=False)
        self.assertEqual(df["LITHOLOGY"].iloc[0], 3000)
        self.assertNotIn("LITHOLOGY_RAW", df.columns)
        self.assertNotIn("LITHOLOGY_RAW_RAW", df.columns)

        df_all = self.loader.load_csv(temp_csv, keep_other=True)
        self.assertEqual(df_all["LITHOLOGY"].iloc[0], 3000)
        self.assertEqual(df_all["LITHOLOGY_RAW"].iloc[0], 5000)
        self.assertEqual(df_all["LITHOLOGY_RAW_RAW"].iloc[0], 4000)

    def test_load_csv_case_insensitive_well(self):
        """
        Verify that well or Well columns are mapped to WELL and filtering works.
        """
        data = "well;DEPTH_MD;GR\nWELL_A;1000.0;45.2\nWELL_B;2000.0;110.1\n"
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write(data)
            temp_csv = f.name
            self.temp_files.append(temp_csv)

        df = self.loader.load_csv(temp_csv, wells=["WELL_B"])
        self.assertEqual(len(df), 1)
        self.assertEqual(df["WELL"].iloc[0], "WELL_B")

    def test_load_csv_null_normalization(self):
        """
        Verify load_csv normalizes null values like -999.25 and custom values to NaN.
        """
        data = "WELL;DEPTH_MD;GR\nWELL_A;1000.0;-999.25\nWELL_A;1001.0;-9999.0\n"
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write(data)
            temp_csv = f.name
            self.temp_files.append(temp_csv)

        df = self.loader.load_csv(temp_csv, null_values=[-9999.0])
        self.assertTrue(np.isnan(df["GR"].iloc[0]))
        self.assertTrue(np.isnan(df["GR"].iloc[1]))

    def test_null_value_tolerance(self):
        """
        Verify that near-match sentinels (e.g. -999.24) are cleaned using absolute tolerance,
        but relative tolerance is 0 so -999.24 is cleaned while larger deviations are not.
        """
        df_test = pd.DataFrame({"GR": [-999.25, -999.25005, -999.24, -999.26]})
        cleaned = self.loader._clean_null_values(df_test, target_nulls=[-999.25])
        self.assertTrue(np.isnan(cleaned["GR"].iloc[0]))
        self.assertTrue(np.isnan(cleaned["GR"].iloc[1]))
        self.assertFalse(np.isnan(cleaned["GR"].iloc[2]))
        self.assertFalse(np.isnan(cleaned["GR"].iloc[3]))

    def test_load_las_well_curve_preservation(self):
        """
        Verify that if the LAS file has a curve named WELL, it is preserved and not overwritten.
        """
        las = lasio.LASFile()
        las.well["WELL"] = lasio.HeaderItem("WELL", value="TEST_HEADER_WELL")

        las.append_curve("DEPT", [100.0, 101.0], unit="M")
        las.append_curve("WELL", [111.0, np.nan], unit="")

        with tempfile.NamedTemporaryFile(suffix=".las", delete=False, mode="w") as f:
            las.write(f)
            temp_las = f.name
            self.temp_files.append(temp_las)

        df = self.loader.load_las(temp_las)
        self.assertEqual(df["WELL"].iloc[0], "111.0")
        self.assertEqual(df["WELL"].iloc[1], "TEST_HEADER_WELL")


if __name__ == "__main__":
    unittest.main()
