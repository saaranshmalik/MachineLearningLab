"""
Generate matplotlib-based visualizations for the stock price prediction project.
"""

import os
import warnings

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MPL_CONFIG_DIR = os.path.join(PROJECT_ROOT, ".matplotlib")
os.makedirs(MPL_CONFIG_DIR, exist_ok=True)
os.environ["MPLCONFIGDIR"] = MPL_CONFIG_DIR

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from project_pipeline import train_model

warnings.filterwarnings("ignore")

OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "matplotlib_graphs")


def save_plots(results):
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    metrics = results["metrics"]
    y_test_original = results["y_test_original"]
    y_pred_test_original = results["y_pred_test_original"]
    residuals_test = y_test_original - y_pred_test_original
    error_test = np.abs(residuals_test)

    dashboard_path = os.path.join(OUTPUT_PATH, "model_dashboard.png")
    prediction_path = os.path.join(OUTPUT_PATH, "actual_vs_predicted.png")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Stock Price Prediction Dashboard", fontsize=16, fontweight="bold")

    axes[0, 0].bar(
        ["Training R2", "Testing R2"],
        [metrics["train_r2"] * 100, metrics["test_r2"] * 100],
        color=["#2d6a4f", "#40916c"],
    )
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_title("Accuracy Comparison")
    axes[0, 0].set_ylim(0, 100)

    axes[0, 1].bar(
        ["Train RMSE", "Test RMSE", "Train MAE", "Test MAE"],
        [
            metrics["train_rmse"],
            metrics["test_rmse"],
            metrics["train_mae"],
            metrics["test_mae"],
        ],
        color=["#1d3557", "#457b9d", "#e76f51", "#f4a261"],
    )
    axes[0, 1].set_title("Error Metrics")
    axes[0, 1].tick_params(axis="x", rotation=20)

    sample_size = min(200, len(y_test_original))
    axes[0, 2].plot(y_test_original[:sample_size], label="Actual", linewidth=2, color="#264653")
    axes[0, 2].plot(y_pred_test_original[:sample_size], label="Predicted", linewidth=2, color="#e76f51")
    axes[0, 2].set_title("Actual vs Predicted")
    axes[0, 2].set_xlabel("Sample")
    axes[0, 2].set_ylabel("Close Price")
    axes[0, 2].legend()

    axes[1, 0].hist(residuals_test, bins=30, color="#6d597a", edgecolor="white")
    axes[1, 0].set_title("Residual Distribution")
    axes[1, 0].set_xlabel("Residual")
    axes[1, 0].set_ylabel("Frequency")

    metric_names = ["Train R2", "Test R2", "Train RMSE", "Test RMSE", "Train MAE", "Test MAE"]
    metric_values = [
        metrics["train_r2"],
        metrics["test_r2"],
        metrics["train_rmse"],
        metrics["test_rmse"],
        metrics["train_mae"],
        metrics["test_mae"],
    ]
    axes[1, 1].axis("off")
    table = axes[1, 1].table(
        cellText=[[name, f"{value:.6f}"] for name, value in zip(metric_names, metric_values)],
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.6)
    axes[1, 1].set_title("Detailed Metrics")

    percentiles = [10, 25, 50, 75, 90, 95, 99]
    percentile_values = [np.percentile(error_test, p) for p in percentiles]
    axes[1, 2].plot(percentiles, percentile_values, marker="o", linewidth=2, color="#bc4749")
    axes[1, 2].set_title("Error Magnitude Percentiles")
    axes[1, 2].set_xlabel("Percentile")
    axes[1, 2].set_ylabel("Absolute Error")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(dashboard_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.plot(y_test_original[:sample_size], label="Actual", linewidth=2, color="#264653")
    ax2.plot(y_pred_test_original[:sample_size], label="Predicted", linewidth=2, color="#e76f51")
    ax2.set_title("Stock Closing Price: Actual vs Predicted")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Close Price")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(prediction_path, dpi=160, bbox_inches="tight")
    plt.close(fig2)

    return dashboard_path, prediction_path


def main():
    print("\n" + "=" * 100)
    print(" " * 20 + "STOCK PRICE PREDICTION - MATPLOTLIB VISUALIZATIONS")
    print("=" * 100 + "\n")
    print("Loading data and training model...")

    results = train_model()
    dashboard_path, prediction_path = save_plots(results)

    metrics = results["metrics"]
    print("\n[OK] Matplotlib graphs created successfully.\n")
    print(f"Training R2: {metrics['train_r2'] * 100:.2f}%")
    print(f"Testing R2:  {metrics['test_r2'] * 100:.2f}%")
    print(f"Train RMSE:  {metrics['train_rmse']:.6f}")
    print(f"Test RMSE:   {metrics['test_rmse']:.6f}")
    print(f"Train MAE:   {metrics['train_mae']:.6f}")
    print(f"Test MAE:    {metrics['test_mae']:.6f}\n")
    print(f"Dashboard plot saved to: {dashboard_path}")
    print(f"Prediction plot saved to: {prediction_path}\n")


if __name__ == "__main__":
    main()
