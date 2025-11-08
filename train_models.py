from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Networks outage log with sol.xlsx"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = ["cause_input", "country_input", "region_input", "severity_input"]
DURATION_MODEL_PATH = MODELS_DIR / "duration_model.joblib"
SOLUTION_MODEL_PATH = MODELS_DIR / "solution_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


def load_dataset() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()

    duration_col = None
    if "duration" in df.columns:
        duration_col = "duration"
    else:
        for candidate in df.columns:
            if "duration" in candidate:
                duration_col = candidate
                break
    if duration_col is None:
        raise ValueError("Unable to locate a duration column in the dataset.")

    df["duration"] = pd.to_numeric(df[duration_col], errors="coerce")

    cause_col = None
    if "actual_cause" in df.columns:
        cause_col = "actual_cause"
    else:
        for candidate in df.columns:
            if "cause" in candidate:
                cause_col = candidate
                break
    if cause_col is None:
        raise ValueError("Unable to locate a cause column in the dataset.")

    if "solution" not in df.columns:
        if "solutions" in df.columns:
            df["solution"] = df["solutions"]
        else:
            df["solution"] = "Restart core routers, check ISP peering and DNS."

    columns_map = {cause_col: "cause", "duration": "duration", "solution": "solution"}
    if "country" in df.columns:
        columns_map["country"] = "country"
    if "region" in df.columns:
        columns_map["region"] = "region"

    missing_columns = {"country", "region"} - set(df.columns)
    for missing in missing_columns:
        df[missing] = "unknown"

    df = df[list(columns_map.keys())].copy()
    df = df.rename(columns=columns_map)

    df["cause"] = df["cause"].fillna("unknown").astype(str)
    df["country"] = df["country"].fillna("unknown").astype(str)
    df["region"] = df["region"].fillna("unknown").astype(str)
    df["solution"] = df["solution"].fillna("Restart core routers, check ISP peering and DNS.").astype(str)

    df = df.dropna(subset=["duration"])
    df = df[df["duration"] > 0]

    df["cause_input"] = df["cause"].str.strip().str.lower()
    df["country_input"] = df["country"].str.strip().str.lower()
    df["region_input"] = df["region"].str.strip().str.lower()
    df["severity_input"] = df["duration"].apply(derive_severity_label)

    df["solution_clean"] = df["solution"].str.strip()
    df["solution_clean"] = df["solution_clean"].replace("", "Restart core routers, check ISP peering and DNS.")

    return df


def derive_severity_label(duration_value: float) -> str:
    if pd.isna(duration_value):
        return "medium"
    if duration_value <= 1:
        return "low"
    if duration_value <= 3:
        return "medium"
    if duration_value <= 7:
        return "high"
    return "critical"


def build_preprocessor() -> ColumnTransformer:
    categorical_features = FEATURE_COLUMNS
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(transformers=[("categorical", categorical_transformer, categorical_features)])


def train_and_save_models() -> None:
    df = load_dataset()

    X = df[FEATURE_COLUMNS]
    y_duration = df["duration"]
    y_solution = df["solution_clean"]

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_duration, test_size=0.2, random_state=42
    )
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X, y_solution, test_size=0.2, random_state=42, stratify=y_solution
    )

    duration_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )

    solution_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )

    duration_pipeline.fit(X_train_reg, y_train_reg)
    solution_pipeline.fit(X_train_clf, y_train_clf)

    duration_predictions = duration_pipeline.predict(X_test_reg)
    duration_mae = mean_absolute_error(y_test_reg, duration_predictions)

    solution_predictions = solution_pipeline.predict(X_test_clf)
    solution_accuracy = accuracy_score(y_test_clf, solution_predictions)

    print(f"Duration model MAE: {duration_mae:.3f} days")
    print(f"Solution model accuracy: {solution_accuracy:.3f}")

    joblib.dump(duration_pipeline, DURATION_MODEL_PATH)
    joblib.dump(solution_pipeline, SOLUTION_MODEL_PATH)

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "valid_causes": sorted(df["cause_input"].dropna().unique().tolist()),
        "training_rows": int(len(df)),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print(f"Saved regression model to {DURATION_MODEL_PATH}")
    print(f"Saved classification model to {SOLUTION_MODEL_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    train_and_save_models()
