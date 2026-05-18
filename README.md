Energy Demand Forecasting

- A time-series forecasting project that predicts household energy consumption using historical power usage data. The project combines statistical forecasting models like ARIMA and SARIMAX with XGBoost based machine learning to analyze trends, generate forecasts, and compare model performance.
- The pipeline includes data preprocessing, feature engineering, model evaluation, visualization, and future energy demand forecasting using hourly power consumption data.

Dataset:
https://drive.google.com/file/d/1-0PYDRQ_UhPUc8ug47IhtapmlhyRYclN/view?usp=sharing

Features:
- Data preprocessing and hourly resampling
- Missing value handling using interpolation
- Time-based feature engineering
- Lag and rolling window feature generation
- Cyclical encoding for hourly patterns
- ARIMA and SARIMAX statistical forecasting
- XGBoost regression based forecasting
- Model comparison using MAE, RMSE, and MAPE
- 24-hour future energy demand prediction
- Forecast visualization and feature importance plots

Tech Stack:
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost
- Statsmodels

Workflow:
1. Energy Consumption Data Preprocessing
2. Missing Value Handling & Hourly Resampling
3. Time-Based, Lag & Rolling Feature Engineering
4. ARIMA, SARIMAX & XGBoost Model Training
5. Forecast Generation & Performance Evaluation
6. 24 Hour Energy Demand Prediction & Visualization

Installation:
1. Clone the Repository: 
git clone https://github.com/samyukta-k/Energy-Demand-Forecasting.git
cd smart-energy-forecasting

2. Create Virtual Environment:
python -m venv venv
source venv/bin/activate

3. Running the Project:
python3 preprocess.py
python3 xgboost_model.py
python3 forecast.py

Future Improvements:
- Deploy forecasting dashboard using Streamlit
- Extend forecasting horizon beyond 24 hours
- Add weather-based external features for improved prediction accuracy
