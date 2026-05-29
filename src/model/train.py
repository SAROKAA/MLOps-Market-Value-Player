from pathlib import Path
import json
import sys
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

# =====================================================
# PATH
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# CONFIG
# =====================================================

EXPERIMENT_NAME = "market_value_prediction"
REGISTERED_MODEL_NAME = "market_value_model"

# threshold validation
R2_THRESHOLD = 0.60

# =====================================================
# MLFLOW CONFIG
# =====================================================

MLRUNS_DIR = BASE_DIR / "mlruns"

MLRUNS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

mlflow.set_tracking_uri(
    f"file://{MLRUNS_DIR.resolve().as_posix()}"
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)

# =====================================================
# LOAD DATA
# =====================================================

print("Loading training data...")

df = pd.read_csv(
    PROCESSED_DIR / "final_training.csv"
)

print("Raw shape:", df.shape)

# =====================================================
# FILTER GB1
# =====================================================

df = df[
    df["current_club_domestic_competition_id"]
    == "GB1"
].copy()

print("After GB1 filter:", df.shape)

# =====================================================
# FEATURE ENGINEERING
# =====================================================

print("Feature engineering...")

# one hot encoding position
position_dummies = pd.get_dummies(
    df[["position"]],
    prefix_sep="_"
)

df = pd.concat(
    [df, position_dummies],
    axis=1
)

# remove null target
print(
    "Missing target before:",
    df["market_value_in_eur"].isna().sum()
)

df = df.dropna(
    subset=["market_value_in_eur"]
).copy()

print(
    "After removing null target:",
    df.shape
)

FEATURES = [
    "goals",
    "goals_against",
    "goals_for",
    "games",
    "assists",
    "minutes_played",
    "age",
    "squad_size",
    "term_days_remaining",
    "position_Attack",
    "position_Defender",
    "position_Goalkeeper",
    "position_Midfield",
]

# ensure missing dummy columns exist
for col in FEATURES:
    if col not in df.columns:
        df[col] = 0

X = df[FEATURES].fillna(-1000)
y = df["market_value_in_eur"]

print("X shape:", X.shape)
print("y shape:", y.shape)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=21
)

# =====================================================
# TRAIN + TRACKING
# =====================================================

with mlflow.start_run() as run:

    run_id = run.info.run_id

    print("Training model...")

    model = GradientBoostingRegressor(
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    print("Training complete")

    # =====================================================
    # PREDICTION
    # =====================================================

    pred = model.predict(X_test)

    r2 = r2_score(
        y_test,
        pred
    )

    mae = mean_absolute_error(
        y_test,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )

    print("\n===== METRICS =====")
    print(f"R²   : {r2:.4f}")
    print(f"MAE  : €{mae:,.2f}")
    print(f"RMSE : €{rmse:,.2f}")

    # =====================================================
    # LOG PARAMS
    # =====================================================

    mlflow.log_param(
        "model_name",
        "GradientBoostingRegressor"
    )

    mlflow.log_param(
        "random_state",
        0
    )

    mlflow.log_param(
        "n_features",
        len(FEATURES)
    )

    mlflow.log_param(
        "features",
        ",".join(FEATURES)
    )

    # =====================================================
    # LOG METRICS
    # =====================================================

    mlflow.log_metric(
        "r2_score",
        r2
    )

    mlflow.log_metric(
        "mae",
        mae
    )

    mlflow.log_metric(
        "rmse",
        rmse
    )

    # =====================================================
    # SAVE MODEL
    # =====================================================

    model_path = (
        MODEL_DIR
        / "gradient_boosting.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        "\nModel saved:",
        model_path
    )

    mlflow.log_artifact(
        str(model_path)
    )

    # =====================================================
    # VALIDATION (STEP 3/4)
    # =====================================================

    metrics = {
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
    }

    with open(
        BASE_DIR / "metrics.json",
        "w"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    print("\nmetrics.json saved")

    if r2 < R2_THRESHOLD:
        print(
            f"\nFAILED: "
            f"R²={r2:.4f} "
            f"< threshold={R2_THRESHOLD}"
        )
        sys.exit(1)

    print("\nModel passed validation")

    # =====================================================
    # REGISTER MODEL (STEP 5)
    # =====================================================

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME
    )

    print("\nModel registered")

    client = mlflow.tracking.MlflowClient()

    latest_versions = client.get_latest_versions(
        REGISTERED_MODEL_NAME
    )

    latest_version = latest_versions[-1].version

    # alias staging
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        "staging",
        latest_version
    )

    print(
        f"Model version {latest_version} "
        f"set to alias 'staging'"
    )

print("\nDone!")