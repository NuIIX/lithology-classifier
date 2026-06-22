import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class ForcePenaltyScorer:
    """
    Computes the custom penalty score for the FORCE 2020 Lithology Classification Challenge.
    """

    def __init__(
        self,
        matrix: np.ndarray | None = None,
        matrix_path: str | Path | None = None,
    ):
        if matrix is not None:
            self.matrix = matrix
        else:
            path = Path(matrix_path or "data/penalty_matrix.npy")
            logger.info(f"Loading penalty matrix from {path}")
            self.matrix = np.load(path)

        self.mapping = {
            30000: 0,
            65030: 1,
            65000: 2,
            80000: 3,
            74000: 4,
            70000: 5,
            70032: 6,
            88000: 7,
            86000: 8,
            99000: 9,
            90000: 10,
            93000: 11,
        }

    def calculate_penalty_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the average penalty score between true and predicted labels.
        If inputs contain raw lithology codes, they are mapped to 0..11 indices.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if len(y_true) == 0:
            return 0.0

        is_true_raw = np.any(y_true > 11)
        is_pred_raw = np.any(y_pred > 11)

        mapped_true = np.vectorize(self.mapping.get)(y_true) if is_true_raw else y_true

        mapped_pred = np.vectorize(self.mapping.get)(y_pred) if is_pred_raw else y_pred

        penalties = self.matrix[mapped_true, mapped_pred]
        return float(-np.mean(penalties))
