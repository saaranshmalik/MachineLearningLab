"""
Display model performance metrics in the terminal.
"""

from project_pipeline import train_model


def main():
    results = train_model()
    metrics = results["metrics"]

    train_accuracy_pct = metrics["train_r2"] * 100
    test_accuracy_pct = metrics["test_r2"] * 100

    print("\n" + "=" * 80)
    print(" " * 20 + "STOCK PRICE PREDICTION MODEL - ACCURACY REPORT")
    print("=" * 80 + "\n")

    print("+- TRAINING SET PERFORMANCE " + "-" * 50 + "+")
    print(f"| R2 Score:                  {metrics['train_r2']:.4f}  ({train_accuracy_pct:.2f}%)           |")
    print(f"| RMSE:                      {metrics['train_rmse']:.6f}                    |")
    print(f"| MAE:                       {metrics['train_mae']:.6f}                    |")
    print(f"| MAPE:                      {metrics['train_mape']:.4f}                      |")
    print(f"| Samples:                   {len(results['y_train'])}                           |")
    print("+" + "-" * 76 + "+\n")

    print("+- TESTING SET PERFORMANCE " + "-" * 51 + "+")
    print(f"| R2 Score:                  {metrics['test_r2']:.4f}  ({test_accuracy_pct:.2f}%)           |")
    print(f"| RMSE:                      {metrics['test_rmse']:.6f}                    |")
    print(f"| MAE:                       {metrics['test_mae']:.6f}                    |")
    print(f"| MAPE:                      {metrics['test_mape']:.4f}                      |")
    print(f"| Samples:                   {len(results['y_test'])}                           |")
    print("+" + "-" * 76 + "+\n")

    print("+- SUMMARY " + "-" * 66 + "+")
    print(f"| Training Accuracy:         {train_accuracy_pct:.2f}%                                  |")
    print(f"| Testing Accuracy:          {test_accuracy_pct:.2f}%                                  |")
    print(f"| Model Status:              READY                                     |")
    print("+" + "-" * 76 + "+\n")

    train_bar = min(50, max(0, int(round(train_accuracy_pct / 2))))
    test_bar = min(50, max(0, int(round(test_accuracy_pct / 2))))
    print("===== ACCURACY VISUALIZATION =====\n")
    print(f"Training:  [{'=' * train_bar}{' ' * (50 - train_bar)}] {train_accuracy_pct:.2f}%")
    print(f"Testing:   [{'=' * test_bar}{' ' * (50 - test_bar)}] {test_accuracy_pct:.2f}%\n")

    print("=" * 80)
    print("Report generated successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
