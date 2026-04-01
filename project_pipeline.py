"""
Shared stock price prediction pipeline used by the runnable project scripts.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(PROJECT_ROOT, "inputs")
ARTIFACTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "model_artifacts")
DATA_FILE = os.path.join(INPUT_PATH, "ge.us.txt")
TRAIN_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
TIME_STEPS = 60


def create_features(data, time_steps=TIME_STEPS):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, :])
        y.append(data[i + time_steps, 3])
    return np.array(X), np.array(y)


def invert_close_values(scaler, values):
    dummy = np.zeros((len(values), len(TRAIN_COLUMNS)))
    dummy[:, 3] = values
    return scaler.inverse_transform(dummy)[:, 3]


def safe_mape(y_true, y_pred):
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask):
        return 0.0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]))


def load_prepared_data():
    df = pd.read_csv(DATA_FILE, engine="python")
    df_train, df_test = train_test_split(df, train_size=0.8, test_size=0.2, shuffle=False)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(df_train[TRAIN_COLUMNS])
    scaled_test = scaler.transform(df_test[TRAIN_COLUMNS])

    X_train, y_train = create_features(scaled_train)
    X_test, y_test = create_features(scaled_test)

    return {
        "dataframe": df,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_reshaped": X_train.reshape(X_train.shape[0], -1),
        "X_test_reshaped": X_test.reshape(X_test.shape[0], -1),
    }


def build_model():
    return GradientBoostingRegressor(
        n_estimators=20,
        learning_rate=0.3,
        max_depth=2,
        subsample=0.7,
        random_state=42,
    )


def train_model():
    prepared = load_prepared_data()
    model = build_model()
    model.fit(prepared["X_train_reshaped"], prepared["y_train"])

    y_pred_train = model.predict(prepared["X_train_reshaped"])
    y_pred_test = model.predict(prepared["X_test_reshaped"])

    metrics = {
        "train_r2": r2_score(prepared["y_train"], y_pred_train),
        "test_r2": r2_score(prepared["y_test"], y_pred_test),
        "train_rmse": np.sqrt(mean_squared_error(prepared["y_train"], y_pred_train)),
        "test_rmse": np.sqrt(mean_squared_error(prepared["y_test"], y_pred_test)),
        "train_mae": mean_absolute_error(prepared["y_train"], y_pred_train),
        "test_mae": mean_absolute_error(prepared["y_test"], y_pred_test),
        "train_mape": safe_mape(prepared["y_train"], y_pred_train),
        "test_mape": safe_mape(prepared["y_test"], y_pred_test),
    }

    return {
        **prepared,
        "model": model,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "y_train_original": invert_close_values(prepared["scaler"], prepared["y_train"]),
        "y_test_original": invert_close_values(prepared["scaler"], prepared["y_test"]),
        "y_pred_train_original": invert_close_values(prepared["scaler"], y_pred_train),
        "y_pred_test_original": invert_close_values(prepared["scaler"], y_pred_test),
        "metrics": metrics,
    }


def save_artifacts(model, scaler):
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    model_path = os.path.join(ARTIFACTS_PATH, "model.pkl")
    scaler_path = os.path.join(ARTIFACTS_PATH, "scaler.pkl")

    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)
    with open(scaler_path, "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    return model_path, scaler_path
