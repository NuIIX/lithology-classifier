import logging
from typing import Any, cast

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight

logger = logging.getLogger(__name__)


class LithotypeCatBoostWrapper:
    """
    CatBoost Classifier wrapper for lithotype classification.
    """

    def __init__(
        self,
        iterations: int = 500,
        depth: int = 6,
        learning_rate: float | None = None,
        early_stopping_rounds: int = 30,
        random_seed: int = 42,
        thread_count: int = -1,
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.random_seed = random_seed
        self.thread_count = thread_count

        self.model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            random_seed=self.random_seed,
            thread_count=self.thread_count,
            verbose=False,
        )

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        eval_set: tuple[pd.DataFrame | np.ndarray, pd.Series | np.ndarray]
        | None = None,
    ) -> "LithotypeCatBoostWrapper":
        """
        Fits the CatBoost model, dynamically computing sample weights to handle class imbalance.
        """
        logger.info("Fitting LithotypeCatBoostWrapper...")
        sample_weights = compute_sample_weight(class_weight="balanced", y=y)

        fit_params: dict[str, Any] = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["early_stopping_rounds"] = self.early_stopping_rounds

        self.model.fit(X, y, sample_weight=sample_weights, verbose=False, **fit_params)
        logger.info("LithotypeCatBoostWrapper fitting complete.")
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predicts classes for the features X.
        """
        logger.info("Predicting with LithotypeCatBoostWrapper...")
        preds = self.model.predict(X)
        if preds.ndim == 2:
            preds = preds.ravel()
        return cast(np.ndarray, preds)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities for the features X.
        """
        logger.info("Predicting probabilities with LithotypeCatBoostWrapper...")
        return cast(np.ndarray, self.model.predict_proba(X))
