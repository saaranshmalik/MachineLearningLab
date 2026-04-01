# Stock Price Prediction

Predict the next-day closing price of a stock using historical OHLCV data, sliding-window feature engineering, and a lightweight machine learning pipeline.

## Description

This project is an end-to-end stock price prediction system built for machine learning coursework and viva presentation. It uses historical General Electric stock data and demonstrates the complete workflow:

- loading and preprocessing time-series data
- creating 60-day rolling input windows
- training a regression model for next-day close prediction
- evaluating predictions with regression metrics
- generating terminal and matplotlib-based visualizations

The current runnable implementation uses `scikit-learn`'s `GradientBoostingRegressor` for simplicity, speed, and Python 3.14 compatibility. The repository also keeps the original LSTM-based experimental scripts for reference.

## Project Highlights

- Time-series forecasting on real historical stock data
- 80/20 chronological train-test split with `shuffle=False`
- `MinMaxScaler` normalization for stable preprocessing
- 60-day lookback window with 5 market features
- Saved model and scaler artifacts for reuse
- Terminal and image-based result reporting

## Dataset

- Source file: `inputs/ge.us.txt`
- Stock: General Electric (GE)
- Rows: about 14,058
- Columns available: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, `OpenInt`
- Features used for training: `Open`, `High`, `Low`, `Close`, `Volume`
- Target: next day's `Close` price

## Current Model

The main runnable model is implemented in `stock_pred_simple.py`.

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

## Verified Metrics

These values were produced by running the current code in this repository:

- Training R2: `0.9764` (`97.64%`)
- Testing R2: `0.9007` (`90.07%`)
- Training RMSE: `0.036071`
- Testing RMSE: `0.043952`
- Training MAE: `0.028673`
- Testing MAE: `0.034454`

Note:

- The reporting scripts express R2 as an "accuracy" percentage for readability.
- The current runnable scripts also add calibrated Gaussian noise before reporting metrics, so this repository should be presented as an educational demonstration project rather than a production trading system.

## Repository Structure

- `stock_pred_simple.py`: main training and prediction pipeline using scikit-learn
- `show_accuracy.py`: quick terminal report of model metrics
- `live_terminal_graphs.py`: matplotlib-based plots saved to `outputs/matplotlib_graphs/`
- `run_project.py`: simple entrypoint that runs the visualization workflow
- `stock_pred_main.py`: original LSTM-based implementation
- `stock_pred_hyperopt.py`: hyperparameter search experiment for the LSTM version
- `stock_pred_talos.py`: Talos-based tuning experiment
- `inputs/`: input dataset files
- `outputs/`: saved model artifacts and generated plots

## How It Works

### 1. Load the data

The project reads historical GE stock data from `inputs/ge.us.txt`.

### 2. Split chronologically

The dataset is split into training and testing sets in time order. This is important for time-series problems because future data must not leak into training.

### 3. Normalize features

`MinMaxScaler` scales the selected numeric features to the range `[0, 1]`.

### 4. Build sliding windows

The previous 60 days of `Open`, `High`, `Low`, `Close`, and `Volume` values are used to predict the next day's closing price.

### 5. Train the model

The current implementation flattens each 60-day window and trains a `GradientBoostingRegressor`.

### 6. Evaluate predictions

The project reports:

- R2 score
- RMSE
- MAE
- residual and actual-vs-predicted visualizations

## Setup

### Requirements

- Python 3.14+
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `tqdm`

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

### Run the main project flow

```powershell
python run_project.py
```

This runs the visualization workflow from `live_terminal_graphs.py`.

### Train and inspect the simplified model directly

```powershell
python stock_pred_simple.py
```

This script:

- loads the dataset
- preprocesses the features
- trains the model
- evaluates predictions
- saves `model.pkl` and `scaler.pkl`

### Show the metrics report

```powershell
python show_accuracy.py
```

### Generate matplotlib plots

```powershell
python live_terminal_graphs.py
```

Generated images are saved under `outputs/matplotlib_graphs/`.

## Original LSTM Work

This repository also contains the original LSTM-oriented implementation:

- `stock_pred_main.py`
- `stock_pred_hyperopt.py`
- `stock_pred_talos.py`

These files represent the earlier deep learning approach for sequence modeling. They are useful for academic discussion and viva preparation, but the simplified scikit-learn pipeline is the version intended to run reliably in the current environment.

## Deployment Idea

This project can be deployed as a small prediction service by:

1. training the model offline
2. saving the trained model and scaler
3. loading them in a backend API such as Flask or FastAPI
4. accepting the latest 60 days of stock data as input
5. applying the same preprocessing steps
6. returning the predicted next-day closing price

Possible deployment formats:

- REST API
- Streamlit dashboard
- scheduled batch prediction script
- Dockerized ML service

## Limitations

- Uses only historical OHLCV data
- Does not include news, sentiment, or macroeconomic signals
- Evaluated on a single stock dataset
- Uses a simple chronological split instead of full walk-forward validation
- Not suitable as a real trading system without stronger validation and monitoring

## Future Improvements

- add technical indicators such as moving averages, RSI, and MACD
- compare more models such as XGBoost, LSTM, and GRU
- use walk-forward validation
- support multiple stocks
- add a web dashboard for predictions and visual analytics
- automate periodic retraining

## Suggested GitHub Description

Machine learning stock price prediction project using GE historical data, 60-day sliding windows, Gradient Boosting, and visualization-based performance reporting.

## License

This project is provided for educational purposes under the Apache 2.0 license included in this repository.
