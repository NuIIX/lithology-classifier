import logging
from typing import cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class BaselineLogisticRegression:
    """
    Logistic Regression baseline model for lithotype classification.
    Wraps scaling and model training inside a Scikit-Learn pipeline to prevent scaling leakage.
    Imputation is handled globally by the DataPreprocessor.
    """

    def __init__(self, random_state: int = 42, max_iter: int = 1000, n_jobs: int = -1):
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        random_state=self.random_state,
                        max_iter=self.max_iter,
                        n_jobs=self.n_jobs,
                    ),
                ),
            ]
        )

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> "BaselineLogisticRegression":
        """
        Fits the pipeline on features X and labels y.
        """
        logger.info("Fitting BaselineLogisticRegression model pipeline...")
        self.pipeline.fit(X, y)
        logger.info("BaselineLogisticRegression fitting complete.")
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predicts classes for the features X.
        """
        logger.info("Predicting with BaselineLogisticRegression...")
        return cast(np.ndarray, self.pipeline.predict(X))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities for the features X.
        """
        logger.info("Predicting probabilities with BaselineLogisticRegression...")
        return cast(np.ndarray, self.pipeline.predict_proba(X))
