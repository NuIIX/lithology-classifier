import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GreedyStratifiedGroupSplit:
    """
    Greedy Stratified Group Split algorithm.
    Partitions groups (wells) into train and test sets such that the class distribution
    in both sets is as close as possible to the target proportions, while ensuring
    no group is split across train and test (no spatial leakage).
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.train_size = 1.0 - test_size
        self.random_state = random_state

    def split(
        self, df: pd.DataFrame, groups_col: str = "WELL", target_col: str = "LITHOLOGY"
    ) -> tuple[list[str], list[str]]:
        """
        Computes the split of groups into train and test sets.

        Args:
            df: Input DataFrame.
            groups_col: Column containing group IDs (wells).
            target_col: Column containing target class labels.

        Returns:
            Tuple of (train_groups, test_groups).
        """
        if groups_col not in df.columns:
            raise ValueError(f"Groups column '{groups_col}' not found in DataFrame.")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        df_clean = df.dropna(subset=[groups_col, target_col])

        if df_clean.empty:
            logger.warning("DataFrame has no valid rows for splitting.")
            return [], []

        group_class_counts = (
            df_clean.groupby([groups_col, target_col]).size().unstack(fill_value=0)
        )

        global_counts = group_class_counts.sum(axis=0)
        sorted_classes = global_counts.sort_values().index.tolist()

        groups = df[groups_col].dropna().unique().tolist()

        train_groups: list[str] = []
        test_groups: list[str] = []

        class_train_counts = pd.Series(0.0, index=group_class_counts.columns)
        class_test_counts = pd.Series(0.0, index=group_class_counts.columns)

        assigned_groups: set[str] = set()

        for cls in sorted_classes:
            cls_groups = group_class_counts[group_class_counts[cls] > 0].index
            unassigned_cls_groups = [g for g in cls_groups if g not in assigned_groups]

            if not unassigned_cls_groups:
                continue

            unassigned_cls_groups = (
                group_class_counts.loc[unassigned_cls_groups, cls]
                .sort_values(ascending=False)
                .index.tolist()
            )

            for g in unassigned_cls_groups:
                g_counts = group_class_counts.loc[g]

                cand_train_counts = class_train_counts + g_counts
                cand_test_counts = class_test_counts + g_counts

                loss_train = 0.0
                loss_test = 0.0

                for c_name in global_counts.index:
                    t_val = global_counts[c_name]

                    diff_tr_train = (
                        cand_train_counts[c_name] - self.train_size * t_val
                    ) / t_val
                    diff_ts_train = (
                        class_test_counts[c_name] - self.test_size * t_val
                    ) / t_val
                    loss_train += diff_tr_train**2 + diff_ts_train**2

                    diff_tr_test = (
                        class_train_counts[c_name] - self.train_size * t_val
                    ) / t_val
                    diff_ts_test = (
                        cand_test_counts[c_name] - self.test_size * t_val
                    ) / t_val
                    loss_test += diff_tr_test**2 + diff_ts_test**2

                n_tr_train = len(train_groups) + 1
                n_ts_train = len(test_groups)
                n_tr_test = len(train_groups)
                n_ts_test = len(test_groups) + 1
                n_tot = len(groups)

                well_loss_train = (
                    (n_tr_train - self.train_size * n_tot) / n_tot
                ) ** 2 + ((n_ts_train - self.test_size * n_tot) / n_tot) ** 2
                well_loss_test = (
                    (n_tr_test - self.train_size * n_tot) / n_tot
                ) ** 2 + ((n_ts_test - self.test_size * n_tot) / n_tot) ** 2

                total_loss_train = loss_train + 0.1 * well_loss_train
                total_loss_test = loss_test + 0.1 * well_loss_test

                if total_loss_train <= total_loss_test:
                    train_groups.append(g)
                    class_train_counts += g_counts
                    assigned_groups.add(g)
                else:
                    test_groups.append(g)
                    class_test_counts += g_counts
                    assigned_groups.add(g)

        remaining_groups = [g for g in groups if g not in assigned_groups]
        if remaining_groups:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(remaining_groups)
            for g in remaining_groups:
                if (
                    len(train_groups) / (len(train_groups) + len(test_groups) + 1e-9)
                    < self.train_size
                ):
                    train_groups.append(g)
                else:
                    test_groups.append(g)

        return train_groups, test_groups


def train_test_split_by_well(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    groups_col: str = "WELL",
    target_col: str = "LITHOLOGY",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into train and test sets based on well groups using
    GreedyStratifiedGroupSplit.

    Args:
        df: Input DataFrame.
        test_size: Proportion of wells to assign to the test/hold-out split.
        random_state: Random state for deterministic shuffling.
        groups_col: Column containing group IDs (wells).
        target_col: Column containing target class labels.

    Returns:
        Tuple of (train_df, test_df).
    """
    splitter = GreedyStratifiedGroupSplit(
        test_size=test_size, random_state=random_state
    )
    train_wells, test_wells = splitter.split(
        df, groups_col=groups_col, target_col=target_col
    )

    train_df = df[df[groups_col].isin(train_wells)].copy()
    test_df = df[df[groups_col].isin(test_wells)].copy()

    logger.info("Split summary:")
    logger.info(f"Total wells: {df[groups_col].nunique()}")
    logger.info(f"Train wells: {len(train_wells)} {train_wells}")
    logger.info(f"Test wells: {len(test_wells)} {test_wells}")
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Test samples: {len(test_df)}")

    train_dist = train_df[target_col].value_counts(normalize=True) * 100
    test_dist = test_df[target_col].value_counts(normalize=True) * 100
    logger.info("Class distributions (%):")
    all_classes = sorted(df[target_col].dropna().unique())
    for cls in all_classes:
        tr_pct = train_dist.get(cls, 0.0)
        ts_pct = test_dist.get(cls, 0.0)
        logger.info(f"  Class {cls}: Train={tr_pct:.2f}%, Test={ts_pct:.2f}%")

    return train_df, test_df
