import logging
import os

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
from src.models.baseline_lr import BaselineLogisticRegression
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


def calculate_penalty_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    penalty_matrix_path: str = "data/penalty_matrix.npy",
) -> float:
    """
    Computes the FORCE 2020 custom penalty score.
    S = - (1/N) * sum(A[y_true, y_pred])
    """
    if not os.path.exists(penalty_matrix_path):
        logger.error(f"Penalty matrix not found at {penalty_matrix_path}")
        return np.nan

    penalty_matrix = np.load(penalty_matrix_path)

    y_true_mapped = np.vectorize(lambda x: LITHOLOGY_NUMBERS.get(x, -1))(y_true)
    y_pred_mapped = np.vectorize(lambda x: LITHOLOGY_NUMBERS.get(x, -1))(y_pred)

    valid_mask = (y_true_mapped != -1) & (y_pred_mapped != -1)
    if not np.all(valid_mask):
        logger.warning(
            "Some lithology values could not be mapped to penalty matrix indices."
        )
        y_true_mapped = y_true_mapped[valid_mask]
        y_pred_mapped = y_pred_mapped[valid_mask]

    if len(y_true_mapped) == 0:
        return 0.0

    penalties = penalty_matrix[y_true_mapped.astype(int), y_pred_mapped.astype(int)]
    return float(-np.mean(penalties))


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list,
    model_name: str,
    output_path: str,
):
    """
    Plots a beautiful confusion matrix normalized by recall (row totals) and saves it.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
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


def main():
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

    results = []

    logger.info("Initializing Baseline Logistic Regression model...")
    lr_model = BaselineLogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)

    lr_preds = lr_model.predict(X_test)

    lr_acc = accuracy_score(y_test, lr_preds)
    lr_prec_macro, lr_rec_macro, lr_f1_macro, _ = precision_recall_fscore_support(
        y_test, lr_preds, average="macro", labels=unique_labels, zero_division=0
    )
    _, _, lr_f1_weighted, _ = precision_recall_fscore_support(
        y_test, lr_preds, average="weighted", labels=unique_labels, zero_division=0
    )
    lr_penalty = calculate_penalty_score(
        y_test.values, lr_preds, "data/penalty_matrix.npy"
    )

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
    rf_model.fit(X_train, y_train)

    rf_preds = rf_model.predict(X_test)

    rf_acc = accuracy_score(y_test, rf_preds)
    rf_prec_macro, rf_rec_macro, rf_f1_macro, _ = precision_recall_fscore_support(
        y_test, rf_preds, average="macro", labels=unique_labels, zero_division=0
    )
    _, _, rf_f1_weighted, _ = precision_recall_fscore_support(
        y_test, rf_preds, average="weighted", labels=unique_labels, zero_division=0
    )
    rf_penalty = calculate_penalty_score(
        y_test.values, rf_preds, "data/penalty_matrix.npy"
    )

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

    results_df = pd.DataFrame(results)
    results_df.to_csv("reports/baseline_vs_rf.csv", index=False)
    logger.info("Saved summary report to reports/baseline_vs_rf.csv")

    print("\nSummary Experiments Report:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
