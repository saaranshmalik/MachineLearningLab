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


def get_prediction_window(df, selected_date):
    selected_timestamp = pd.Timestamp(selected_date)
    matched_rows = df.index[df["Date"] == selected_timestamp]
    if len(matched_rows) == 0:
        raise ValueError("Selected date was not found in the dataset.")

    selected_index = int(matched_rows[0])
    minimum_index = TIME_STEPS - 1
    if selected_index < minimum_index:
        earliest_date = df.iloc[minimum_index]["Date"].strftime("%Y-%m-%d")
        raise ValueError(
            f"Please choose a date on or after {earliest_date} so the model has {TIME_STEPS} prior rows."
        )

    window = df.iloc[selected_index - TIME_STEPS + 1:selected_index + 1].copy()
    next_row = df.iloc[selected_index + 1].copy() if selected_index + 1 < len(df) else None
    return window, selected_index, next_row


def predict_next_close_from_date(selected_date):
    df = load_stock_dataframe()
    window, selected_index, next_row = get_prediction_window(df, selected_date)
    predicted_close = predict_next_close(window[TRAIN_COLUMNS].to_dict(orient="records"))
    latest_close = float(window["Close"].iloc[-1])

    result = {
        "selected_date": window["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "latest_close": latest_close,
        "predicted_close": predicted_close,
        "delta": predicted_close - latest_close,
        "direction": "Bullish" if predicted_close >= latest_close else "Bearish",
        "window_start_date": window["Date"].iloc[0].strftime("%Y-%m-%d"),
        "window_end_date": window["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "lookback_rows": TIME_STEPS,
        "next_actual_date": None,
        "next_actual_close": None,
    }

    if next_row is not None:
        result["next_actual_date"] = next_row["Date"].strftime("%Y-%m-%d")
        result["next_actual_close"] = float(next_row["Close"])
        result["actual_error"] = predicted_close - result["next_actual_close"]

    return result, window, selected_index


def build_dashboard_context():
    results = train_model()
    dashboard_path, prediction_path = save_plots(results)
    df = load_stock_dataframe()
    recent_rows = df.tail(TIME_STEPS).copy()
    default_prediction, default_window, _ = predict_next_close_from_date(recent_rows["Date"].iloc[-1])
    min_prediction_date = df.iloc[TIME_STEPS - 1]["Date"].strftime("%Y-%m-%d")
    max_prediction_date = df.iloc[-1]["Date"].strftime("%Y-%m-%d")

    return {
        "company_name": "General Electric (GE)",
        "time_steps": TIME_STEPS,
        "metrics": results["metrics"],
        "dashboard_path": dashboard_path,
        "prediction_plot_path": prediction_path,
        "latest_close": default_prediction["latest_close"],
        "predicted_close": default_prediction["predicted_close"],
        "predicted_delta": default_prediction["delta"],
        "predicted_direction": "Up" if default_prediction["delta"] >= 0 else "Down",
        "recent_rows": recent_rows.tail(8).assign(Date=recent_rows.tail(8)["Date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
        "selected_prediction_date": default_prediction["selected_date"],
        "min_prediction_date": min_prediction_date,
        "max_prediction_date": max_prediction_date,
        "prediction_result": default_prediction,
        "prediction_window_preview": default_window.tail(8).assign(Date=default_window.tail(8)["Date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
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
