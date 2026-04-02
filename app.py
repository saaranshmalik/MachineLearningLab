"""
Flask frontend for the stock prediction project.
"""

import os

from flask import Flask, render_template, request, send_from_directory

from prediction_service import build_dashboard_context, predict_next_close
from project_pipeline import PROJECT_ROOT, TIME_STEPS, TRAIN_COLUMNS


app = Flask(__name__)


def rows_to_text(rows):
    return "\n".join(
        f"{row['Open']},{row['High']},{row['Low']},{row['Close']},{row['Volume']}"
        for row in rows
    )


def parse_input_rows(raw_text):
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if len(lines) != TIME_STEPS:
        raise ValueError(f"Please provide exactly {TIME_STEPS} rows of OHLCV data.")

    parsed_rows = []
    for index, line in enumerate(lines, start=1):
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(TRAIN_COLUMNS):
            raise ValueError(
                f"Line {index} must contain {len(TRAIN_COLUMNS)} comma-separated values: "
                "Open, High, Low, Close, Volume."
            )

        row = {}
        for column, value in zip(TRAIN_COLUMNS, values):
            row[column] = float(value)
        parsed_rows.append(row)

    return parsed_rows


@app.route("/", methods=["GET", "POST"])
def index():
    context = build_dashboard_context()
    context["form_text"] = rows_to_text(context["default_form_rows"])
    context["prediction_result"] = None
    context["error_message"] = None

    if request.method == "POST":
        form_text = request.form.get("history_rows", "").strip()
        context["form_text"] = form_text
        try:
            parsed_rows = parse_input_rows(form_text)
            predicted_close = predict_next_close(parsed_rows)
            latest_close = float(parsed_rows[-1]["Close"])
            context["prediction_result"] = {
                "predicted_close": predicted_close,
                "latest_close": latest_close,
                "delta": predicted_close - latest_close,
                "direction": "Bullish" if predicted_close >= latest_close else "Bearish",
            }
        except ValueError as exc:
            context["error_message"] = str(exc)

    return render_template("index.html", **context)


@app.route("/generated/<path:filename>")
def generated_file(filename):
    return send_from_directory(os.path.join(PROJECT_ROOT, "outputs"), filename)


if __name__ == "__main__":
    app.run(debug=True)
