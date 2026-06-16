import logging
from typing import cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class LithotypeRandomForest:
    """
    Random Forest model for lithotype classification.
    Uses balanced_subsample to handle severe class imbalance.
    Imputation is handled globally by the DataPreprocessor.
    """

    def __init__(
        self, random_state: int = 42, n_estimators: int = 200, n_jobs: int = -1
    ):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        class_weight="balanced_subsample",
                        random_state=self.random_state,
                        n_estimators=self.n_estimators,
                        n_jobs=self.n_jobs,
                    ),
                ),
            ]
        )

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> "LithotypeRandomForest":
        """
        Fits the pipeline on features X and labels y.
        """
        logger.info("Fitting LithotypeRandomForest model pipeline...")
        self.pipeline.fit(X, y)
        logger.info("LithotypeRandomForest fitting complete.")
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predicts classes for the features X.
        """
        logger.info("Predicting with LithotypeRandomForest...")
        return cast(np.ndarray, self.pipeline.predict(X))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities for the features X.
        """
        logger.info("Predicting probabilities with LithotypeRandomForest...")
        return cast(np.ndarray, self.pipeline.predict_proba(X))
