import logging
import os
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data.split import train_test_split_by_well
from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.metrics.penalty import ForcePenaltyScorer
from src.models.baseline_lr import BaselineLogisticRegression
from src.models.catboost_classifier import LithotypeCatBoostWrapper
from src.models.random_forest import LithotypeRandomForest

os.makedirs("reports", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("reports/experiment_run.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("run_experiments")

LITHOLOGY_NUMBERS = {
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

LITHOLOGY_NAMES = {
    30000: "Sandstone",
    65030: "Sandstone/Shale",
    65000: "Shale",
    80000: "Marl",
    74000: "Dolomite",
    70000: "Limestone",
    70032: "Chalk",
    88000: "Halite",
    86000: "Anhydrite",
    99000: "Tuff",
    90000: "Coal",
    93000: "Basement",
}


class LithoLabelEncoder:
    """
    Encodes geology lithology codes into dense [0..11] indices.
    """

    def __init__(self) -> None:
        self.mapping = LITHOLOGY_NUMBERS
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}
        self.classes_ = np.array(sorted(self.mapping.keys()))

    def fit(self, y: pd.Series | np.ndarray | None = None) -> "LithoLabelEncoder":
        return self

    def transform(self, y: pd.Series | np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y)
        return cast(np.ndarray, np.vectorize(self.mapping.get)(y_arr))

    def fit_transform(self, y: pd.Series | np.ndarray) -> np.ndarray:
        return self.transform(y)

    def inverse_transform(self, y: pd.Series | np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y)
        return cast(np.ndarray, np.vectorize(self.inverse_mapping.get)(y_arr))


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list,
    model_name: str,
    output_path: str,
) -> None:
    """
    Plots a beautiful confusion matrix normalized by recall (row totals) and saves it.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.get_cmap("Blues"))
    ax.figure.colorbar(im, ax=ax)

    label_names = [f"{lbl} ({LITHOLOGY_NAMES.get(lbl, 'Unknown')})" for lbl in labels]

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=label_names,
        yticklabels=label_names,
        title=f"Confusion Matrix: {model_name} (Normalized Recall)",
        ylabel="True Lithology",
        xlabel="Predicted Lithology",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f"
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm_norm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix plot for {model_name} to {output_path}")


def main() -> None:
    logger.info("Starting well log lithotype classification experiments...")

    train_csv_path = "data/CSV_train.csv"
    logger.info(f"Loading raw data from {train_csv_path}")
    loader = DataLoader()
    raw_df = loader.load_csv(train_csv_path, sep=";")

    logger.info("Splitting wells into Train and Hold-Out splits...")
    train_df, test_df = train_test_split_by_well(raw_df, test_size=0.2, random_state=42)

    logger.info("Initializing DataPreprocessor with impute=True...")
    preprocessor = DataPreprocessor(step=0.1524, impute=True, add_indicators=False)

    logger.info("Fitting DataPreprocessor on Train split...")
    train_df_proc = preprocessor.fit_transform(train_df)

    logger.info("Transforming Hold-Out split using Train statistics...")
    test_df_proc = preprocessor.transform(test_df)

    logger.info("Filtering out missing and out-of-taxonomy targets after resampling...")
    train_df_proc = train_df_proc[
        train_df_proc["LITHOLOGY"].isin(LITHOLOGY_NUMBERS.keys())
    ]
    test_df_proc = test_df_proc[
        test_df_proc["LITHOLOGY"].isin(LITHOLOGY_NUMBERS.keys())
    ]

    features = ["GR", "RHOB", "NPHI", "DT", "SP", "RDEP", "RMED", "RSHA"]
    logger.info(f"Using features: {features}")

    X_train = train_df_proc[features]
    y_train = train_df_proc["LITHOLOGY"].astype(int)

    X_test = test_df_proc[features]
    y_test = test_df_proc["LITHOLOGY"].astype(int)

    logger.info(f"Train shape: {X_train.shape}, Hold-Out test shape: {X_test.shape}")

    unique_labels = sorted(LITHOLOGY_NUMBERS.keys())
    logger.info(f"Target classes across splits based on taxonomy: {unique_labels}")

    le = LithoLabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    scorer = ForcePenaltyScorer(matrix_path="data/penalty_matrix.npy")

    results = []

    logger.info("Initializing Baseline Logistic Regression model...")
    lr_model = BaselineLogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train_encoded)

    lr_preds_encoded = lr_model.predict(X_test)
    lr_preds = le.inverse_transform(lr_preds_encoded)

    lr_acc = accuracy_score(y_test, lr_preds)
    lr_prec_macro, lr_rec_macro, lr_f1_macro, _ = precision_recall_fscore_support(
        y_test, lr_preds, average="macro", labels=unique_labels, zero_division=0
    )
    _, _, lr_f1_weighted, _ = precision_recall_fscore_support(
        y_test, lr_preds, average="weighted", labels=unique_labels, zero_division=0
    )
    lr_penalty = scorer.calculate_penalty_score(y_test.values, lr_preds)

    logger.info("--- Logistic Regression Metrics ---")
    logger.info(f"Accuracy: {lr_acc:.4f}")
    logger.info(f"Macro F1-score: {lr_f1_macro:.4f}")
    logger.info(f"Weighted F1-score: {lr_f1_weighted:.4f}")
    logger.info(f"Penalty Score: {lr_penalty:.4f}")

    results.append(
        {
            "Model": "Logistic Regression (Baseline)",
            "Macro-F1": lr_f1_macro,
            "Weighted-F1": lr_f1_weighted,
            "Accuracy": lr_acc,
            "Precision_macro": lr_prec_macro,
            "Recall_macro": lr_rec_macro,
            "Penalty_Score": lr_penalty,
        }
    )

    plot_and_save_confusion_matrix(
        y_test.values,
        lr_preds,
        unique_labels,
        "Logistic Regression",
        "reports/plots/confusion_matrix_lr.png",
    )

    logger.info("Initializing Random Forest Classifier...")
    rf_model = LithotypeRandomForest(random_state=42)
    rf_model.fit(X_train, y_train_encoded)

    rf_preds_encoded = rf_model.predict(X_test)
    rf_preds = le.inverse_transform(rf_preds_encoded)

    rf_acc = accuracy_score(y_test, rf_preds)
    rf_prec_macro, rf_rec_macro, rf_f1_macro, _ = precision_recall_fscore_support(
        y_test, rf_preds, average="macro", labels=unique_labels, zero_division=0
    )
    _, _, rf_f1_weighted, _ = precision_recall_fscore_support(
        y_test, rf_preds, average="weighted", labels=unique_labels, zero_division=0
    )
    rf_penalty = scorer.calculate_penalty_score(y_test.values, rf_preds)

    logger.info("--- Random Forest Metrics ---")
    logger.info(f"Accuracy: {rf_acc:.4f}")
    logger.info(f"Macro F1-score: {rf_f1_macro:.4f}")
    logger.info(f"Weighted F1-score: {rf_f1_weighted:.4f}")
    logger.info(f"Penalty Score: {rf_penalty:.4f}")

    results.append(
        {
            "Model": "Random Forest",
            "Macro-F1": rf_f1_macro,
            "Weighted-F1": rf_f1_weighted,
            "Accuracy": rf_acc,
            "Precision_macro": rf_prec_macro,
            "Recall_macro": rf_rec_macro,
            "Penalty_Score": rf_penalty,
        }
    )

    plot_and_save_confusion_matrix(
        y_test.values,
        rf_preds,
        unique_labels,
        "Random Forest",
        "reports/plots/confusion_matrix_rf.png",
    )

    logger.info("Initializing CatBoost Classifier...")
    cb_model = LithotypeCatBoostWrapper(random_seed=42)
    cb_model.fit(X_train, y_train_encoded)

    cb_preds_encoded = cb_model.predict(X_test)
    cb_preds = le.inverse_transform(cb_preds_encoded)

    cb_acc = accuracy_score(y_test, cb_preds)
    cb_prec_macro, cb_rec_macro, cb_f1_macro, _ = precision_recall_fscore_support(
        y_test, cb_preds, average="macro", labels=unique_labels, zero_division=0
    )
    _, _, cb_f1_weighted, _ = precision_recall_fscore_support(
        y_test, cb_preds, average="weighted", labels=unique_labels, zero_division=0
    )
    cb_penalty = scorer.calculate_penalty_score(y_test.values, cb_preds)

    logger.info("--- CatBoost Metrics ---")
    logger.info(f"Accuracy: {cb_acc:.4f}")
    logger.info(f"Macro F1-score: {cb_f1_macro:.4f}")
    logger.info(f"Weighted F1-score: {cb_f1_weighted:.4f}")
    logger.info(f"Penalty Score: {cb_penalty:.4f}")

    results.append(
        {
            "Model": "CatBoost",
            "Macro-F1": cb_f1_macro,
            "Weighted-F1": cb_f1_weighted,
            "Accuracy": cb_acc,
            "Precision_macro": cb_prec_macro,
            "Recall_macro": cb_rec_macro,
            "Penalty_Score": cb_penalty,
        }
    )

    plot_and_save_confusion_matrix(
        y_test.values,
        cb_preds,
        unique_labels,
        "CatBoost",
        "reports/plots/confusion_matrix_cb.png",
    )

    results_df = pd.DataFrame(results)
    results_df.to_csv("reports/baseline_vs_rf_vs_cb.csv", index=False)
    logger.info("Saved summary report to reports/baseline_vs_rf_vs_cb.csv")

    print("\nSummary Experiments Report:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
