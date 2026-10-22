"""Train and evaluate an income classifier on the UCI Adult dataset."""

from io import BytesIO
from urllib.request import urlopen

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASET_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
TARGET = "income"


def load_dataset():
    """Download and normalize the official UCI training and test splits."""
    with (
        urlopen(f"{DATASET_BASE_URL}adult.data", timeout=60) as train_response,
        urlopen(f"{DATASET_BASE_URL}adult.test", timeout=60) as test_response,
    ):
        train = pd.read_csv(
            BytesIO(train_response.read()),
            names=COLUMNS,
            skipinitialspace=True,
            na_values=["?"],
        )
        test = pd.read_csv(
            BytesIO(test_response.read()),
            names=COLUMNS,
            skipinitialspace=True,
            na_values=["?"],
            comment="|",
        )

    categorical_columns = [
        column for column in COLUMNS if column not in NUMERIC_FEATURES
    ]
    for frame in (train, test):
        frame[categorical_columns] = frame[categorical_columns].apply(
            lambda column: column.str.strip()
        )
        frame[TARGET] = frame[TARGET].str.rstrip(".")

    train = train.dropna(subset=[TARGET])
    test = test.dropna(subset=[TARGET])
    return train, test


def build_model(x_train):
    """Create the shared preprocessing and logistic-regression pipeline."""
    categorical_features = [
        column for column in x_train.columns if column not in NUMERIC_FEATURES
    ]
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent",
                                keep_empty_features=True,
                            ),
                        ),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def main():
    train, test = load_dataset()
    x_train = train.drop(columns=TARGET)
    y_train = train[TARGET].eq(">50K").astype(int)
    x_test = test.drop(columns=TARGET)
    y_test = test[TARGET].eq(">50K").astype(int)

    model = build_model(x_train)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    positive_class_scores = model.predict_proba(x_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision (>50K)": precision_score(y_test, predictions, zero_division=0),
        "Recall (>50K)": recall_score(y_test, predictions, zero_division=0),
        "F1-score (>50K)": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, positive_class_scores),
    }

    print("Dataset: UCI Adult Census Income (official train/test split)")
    print(f"Rows: {len(train):,} train / {len(test):,} test")
    missing_values = x_train.isna().sum().sum() + x_test.isna().sum().sum()
    print(f"Missing feature values imputed: {int(missing_values):,}")
    print("\nTest-set metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("\nConfusion matrix (rows=true, columns=predicted; labels: <=50K, >50K):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print(
        "\nMetric meanings:\n"
        "- Accuracy: fraction of all predictions that are correct.\n"
        "- Precision: fraction of predicted >50K cases that truly exceed $50K.\n"
        "- Recall: fraction of actual >50K cases correctly identified.\n"
        "- F1-score: harmonic mean of positive-class precision and recall.\n"
        "- ROC-AUC: ability to rank a >50K case above a <=50K case across thresholds."
    )
    print(
        "\nConclusion: Logistic regression provides an interpretable baseline for "
        "predicting the >50K income class. Consider precision and recall alongside "
        "accuracy because the income classes are imbalanced; the confusion matrix "
        "shows the corresponding false-positive and false-negative counts."
    )


if __name__ == "__main__":
    main()
