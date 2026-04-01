# Stock Price Prediction

Predict the next-day closing price of a stock using historical OHLCV data, sliding-window feature engineering, and a lightweight scikit-learn pipeline.

## Description

This project is an end-to-end stock price prediction system built for coursework and viva presentation. It uses historical General Electric stock data and demonstrates the complete workflow:

- loading and preprocessing time-series data
- creating 60-day rolling input windows
- training a regression model for next-day close prediction
- evaluating predictions with regression metrics
- generating saved matplotlib visualizations

The current repository is intentionally focused on the runnable implementation only. Older experimental LSTM and hyperparameter-tuning files have been removed so the project surface matches what is actually used.

## Project Highlights

- Time-series forecasting on historical GE stock data
- 80/20 chronological train-test split with `shuffle=False`
- `MinMaxScaler` normalization for stable preprocessing
- 60-day lookback window with 5 market features
- Saved model and scaler artifacts for reuse
- Terminal and image-based result reporting

## Dataset

- Source file: `inputs/ge.us.txt`
- Stock: General Electric (GE)
- Features used for training: `Open`, `High`, `Low`, `Close`, `Volume`
- Target: next day's `Close` price

## Current Model

- Algorithm: `GradientBoostingRegressor`
- Time window: 60 days
- Input size per sample: `60 x 5 = 300` values
- Train-test split: 80/20
- Scaling: `MinMaxScaler(feature_range=(0, 1))`

Model configuration:

- `n_estimators=20`
- `learning_rate=0.3`
- `max_depth=2`
- `subsample=0.7`
- `random_state=42`

## Repository Structure

- `project_pipeline.py`: shared training, evaluation, and artifact logic
- `stock_pred_simple.py`: main training script
- `show_accuracy.py`: terminal accuracy report
- `live_terminal_graphs.py`: matplotlib plots saved to `outputs/matplotlib_graphs/`
- `run_project.py`: simple default entrypoint
- `inputs/`: input dataset files
- `outputs/`: generated plots and saved model artifacts

## Setup

Requirements:

- Python 3.11+
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Run the main project flow:

```powershell
python run_project.py
```

Train and save artifacts:

```powershell
python stock_pred_simple.py
```

Show the metrics report:

```powershell
python show_accuracy.py
```

Generate matplotlib plots:

```powershell
python live_terminal_graphs.py
```

Generated images are saved under `outputs/matplotlib_graphs/`, and model artifacts are saved under `outputs/model_artifacts/`.

## Limitations

- Uses only historical OHLCV data
- Does not include news, sentiment, or macroeconomic signals
- Evaluated on a single stock dataset
- Uses a simple chronological split instead of walk-forward validation
- Not suitable as a production trading system

## Future Improvements

- add technical indicators such as moving averages, RSI, and MACD
- compare more models such as XGBoost and other tree-based approaches
- use walk-forward validation
- support multiple stocks
- add a web dashboard for predictions and visual analytics
- automate periodic retraining
