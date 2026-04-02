"""
Reusable helpers for serving stock predictions in scripts and the web UI.
"""

import os
import pickle

import pandas as pd

from live_terminal_graphs import save_plots
from project_pipeline import (
    ARTIFACTS_PATH,
    DATA_FILE,
    TIME_STEPS,
    TRAIN_COLUMNS,
    invert_close_values,
    save_artifacts,
    train_model,
)


MODEL_FILE = os.path.join(ARTIFACTS_PATH, "model.pkl")
SCALER_FILE = os.path.join(ARTIFACTS_PATH, "scaler.pkl")


def ensure_artifacts():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        with open(MODEL_FILE, "rb") as model_file:
            model = pickle.load(model_file)
        with open(SCALER_FILE, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler

    results = train_model()
    save_artifacts(results["model"], results["scaler"])
    return results["model"], results["scaler"]


def load_stock_dataframe():
    df = pd.read_csv(DATA_FILE, engine="python")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def build_dashboard_context():
    results = train_model()
    dashboard_path, prediction_path = save_plots(results)
    df = load_stock_dataframe()
    recent_rows = df.tail(TIME_STEPS).copy()
    prediction = predict_next_close(recent_rows[TRAIN_COLUMNS].to_dict(orient="records"))
    latest_close = float(recent_rows["Close"].iloc[-1])
    delta = prediction - latest_close

    return {
        "company_name": "General Electric (GE)",
        "time_steps": TIME_STEPS,
        "metrics": results["metrics"],
        "dashboard_path": dashboard_path,
        "prediction_plot_path": prediction_path,
        "latest_close": latest_close,
        "predicted_close": prediction,
        "predicted_delta": delta,
        "predicted_direction": "Up" if delta >= 0 else "Down",
        "recent_rows": recent_rows.tail(8).assign(Date=recent_rows.tail(8)["Date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
        "default_form_rows": recent_rows.assign(Date=recent_rows["Date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
        "chart_points": df.tail(30).assign(Date=df.tail(30)["Date"].dt.strftime("%Y-%m-%d"))[["Date", "Close"]].to_dict(orient="records"),
    }


def predict_next_close(rows):
    if len(rows) != TIME_STEPS:
        raise ValueError(f"Exactly {TIME_STEPS} rows are required for prediction.")

    model, scaler = ensure_artifacts()
    input_frame = pd.DataFrame(rows, columns=TRAIN_COLUMNS)

    for column in TRAIN_COLUMNS:
        input_frame[column] = pd.to_numeric(input_frame[column], errors="raise")

    scaled_input = scaler.transform(input_frame[TRAIN_COLUMNS])
    features = scaled_input.reshape(1, -1)
    scaled_prediction = model.predict(features)
    predicted_close = invert_close_values(scaler, scaled_prediction)[0]
    return float(predicted_close)
