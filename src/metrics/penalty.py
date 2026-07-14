import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class CustomPenaltyScorer:
    """
    Computes the custom penalty score using a provided penalty matrix and label mapping.
    """

    def __init__(
        self,
        matrix: np.ndarray | None = None,
        matrix_path: str | Path | None = None,
        mapping: dict[int, int] | None = None,
    ):
        if matrix is not None:
            self.matrix = np.asarray(matrix)
        elif matrix_path is not None:
            path = Path(matrix_path)
            logger.info(f"Loading penalty matrix from {path}")
            self.matrix = np.load(path)
        else:
            raise ValueError("Either 'matrix' or 'matrix_path' must be provided.")

        self.mapping = mapping

    def calculate_penalty_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the average penalty score between true and predicted labels.
        If a mapping is provided, raw labels are mapped to matrix indices.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if len(y_true) == 0:
            return 0.0

        if self.mapping is not None:
            max_idx = self.matrix.shape[0] - 1
            is_true_raw = np.any((y_true < 0) | (y_true > max_idx))
            is_pred_raw = np.any((y_pred < 0) | (y_pred > max_idx))

            mapped_true = (
                np.vectorize(self.mapping.get)(y_true) if is_true_raw else y_true
            )
            mapped_pred = (
                np.vectorize(self.mapping.get)(y_pred) if is_pred_raw else y_pred
            )
        else:
            mapped_true = y_true
            mapped_pred = y_pred

        penalties = self.matrix[mapped_true, mapped_pred]
        return float(-np.mean(penalties))
