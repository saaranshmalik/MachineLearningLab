"""
Train the stock price prediction model and save reusable artifacts.
"""

from project_pipeline import TIME_STEPS, save_artifacts, train_model


def main():
    print("=" * 80)
    print("Stock Price Prediction - Training Pipeline")
    print("=" * 80)

    print("\n1. Loading data, building features, and training model...")
    results = train_model()

    metrics = results["metrics"]
    model_path, scaler_path = save_artifacts(results["model"], results["scaler"])

    print("\n2. Model evaluation")
    print(f"   Training R2: {metrics['train_r2']:.6f}")
    print(f"   Testing R2:  {metrics['test_r2']:.6f}")
    print(f"   Training RMSE: {metrics['train_rmse']:.6f}")
    print(f"   Testing RMSE:  {metrics['test_rmse']:.6f}")
    print(f"   Training MAE: {metrics['train_mae']:.6f}")
    print(f"   Testing MAE:  {metrics['test_mae']:.6f}")

    print("\n3. Sample test predictions (original price scale)")
    print(f"   {'Actual':<15} {'Predicted':<15} {'Error':<15}")
    print("   " + "-" * 45)
    for actual, predicted in zip(results["y_test_original"][:10], results["y_pred_test_original"][:10]):
        print(f"   {actual:<15.2f} {predicted:<15.2f} {abs(actual - predicted):<15.2f}")

    print("\n4. Saved artifacts")
    print(f"   Model saved to: {model_path}")
    print(f"   Scaler saved to: {scaler_path}")
    print(f"   Time steps used: {TIME_STEPS}")
    print(f"   Training samples: {len(results['y_train'])}")
    print(f"   Testing samples:  {len(results['y_test'])}")

    print("\n" + "=" * 80)
    print("Training completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
