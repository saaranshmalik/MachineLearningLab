"""
Flask frontend for the stock prediction project.
"""

import os
from datetime import datetime

from flask import Flask, render_template, request, send_from_directory

from prediction_service import build_dashboard_context, predict_next_close_from_date
from project_pipeline import PROJECT_ROOT


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    context = build_dashboard_context()
    context["error_message"] = None

    if request.method == "POST":
        selected_date = request.form.get("prediction_date", "").strip()
        context["selected_prediction_date"] = selected_date
        try:
            datetime.strptime(selected_date, "%Y-%m-%d")
            prediction_result, prediction_window, _ = predict_next_close_from_date(selected_date)
            context["prediction_result"] = prediction_result
            context["prediction_window_preview"] = prediction_window.tail(8).assign(
                Date=prediction_window.tail(8)["Date"].dt.strftime("%Y-%m-%d")
            ).to_dict(orient="records")
        except ValueError as exc:
            context["error_message"] = str(exc)

    return render_template("index.html", **context)


@app.route("/generated/<path:filename>")
def generated_file(filename):
    return send_from_directory(os.path.join(PROJECT_ROOT, "outputs"), filename)


if __name__ == "__main__":
    app.run(debug=True)
