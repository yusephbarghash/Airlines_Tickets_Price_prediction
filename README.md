# Airline Ticket Price Prediction

A machine learning app that predicts Indian domestic flight ticket prices, deployed with Streamlit.

## Features

- Cleans flight data: removes duplicates, missing values, and outliers
- Engineers features: duration in minutes, total stops, and month and weekday of travel
- Compares several regression models and tunes the final XGBoost model with GridSearchCV (R² ≈ 0.82)
- Includes a Streamlit app: pick the airline, route, duration, stops, and date to get a price estimate

## Tech Stack

Python · Pandas · Scikit-learn · XGBoost · Category Encoders · Streamlit

## Project Structure

| File | Description |
|------|-------------|
| `Data_Train.xlsx` | Dataset |
| `Project.ipynb` | Cleaning, analysis, and model training |
| `Flights.py` | Streamlit app |
| `model.pkl` / `inputs.pkl` | Saved model and input columns |
| `requirements.txt` | Dependencies |

## Setup

```bash
git clone https://github.com/yusephbarghash/Airlines_Tickets_Price_prediction.git
cd Airlines_Tickets_Price_prediction
pip install -r requirements.txt
```

## Usage

```bash
streamlit run Flights.py
```
