# PM2.5 Prediction Dashboard

## Overview

This project focuses on developing a system to predict PM2.5 levels 7 days in advance using Artificial Intelligence (AI) and Machine Learning (ML) techniques. The system leverages the **PyCaret** platform for rapid model development and testing, and a **Dashboard** built with **Dash** and **Dash Leaflet** for visualizing the predictions. The dashboard allows users to access real-time PM2.5 data, view predictions, and analyze trends through interactive charts and maps.

The project involves data collection, cleaning, and preprocessing from university databases and external sources. The cleaned data is used to train and fine-tune predictive models, which are then integrated into the dashboard. The dashboard provides features such as:
- **Interactive Map**: Displays PM2.5 sensor locations with color-coded markers based on pollution levels.
- **Time Frame Selection**: Users can choose between 6-hour (6H) and 7-day (1D) prediction intervals.
- **Chart Type Selection**: Users can visualize data using line charts or bar charts.
- **Real-time Updates**: A marquee displays the latest PM2.5 values for each location.
- **Prediction Modal**: Users can click the "Predict" button to view forecasted PM2.5 levels.

The system aims to provide an efficient tool for monitoring and addressing air pollution, with potential applications in environmental planning and public health.

---

## Table of Contents

1. **Introduction**
2. **Data Management Process**
3. **ARIMA Model Training**
4. **Dashboard Development**
5. **Conclusion and Recommendations**

---

## Introduction

Air pollution, particularly PM2.5 (fine particulate matter), poses significant health and environmental risks. This project aims to develop a system for forecasting PM2.5 levels 7 days in advance using **Time Series Analysis** and the **ARIMA (AutoRegressive Integrated Moving Average)** model. The system is built using **PyCaret**, a low-code machine learning platform, and visualized through a **Dash-based Dashboard**.

The project addresses the limitations of traditional regression models, which require future values of independent variables (e.g., temperature, humidity) for prediction. Instead, the ARIMA model uses historical PM2.5 data to forecast future values, making it more suitable for time-series forecasting.

---

## Data Management Process

### Data Selection and Preparation
- **Data Source**: Historical PM2.5, temperature, and humidity data from university databases and external sources.
- **Time Frame**: Data from the past year (February 4, 2024, to February 4, 2025) was used to ensure relevance and reduce computational complexity.

### Data Cleaning
1. **Outlier Removal**: The Interquartile Range (IQR) method was used to detect and remove outliers in PM2.5, temperature, and humidity data.
2. **Missing Value Handling**: Linear interpolation was applied to fill missing values, ensuring data continuity.
3. **Resampling**: Hourly data was resampled to daily averages to reduce noise and improve model stability.

### Feature Engineering
- **Lag Features**: Created lagged variables for PM2.5, temperature, and humidity (e.g., 8, 10, 14, and 21 days).
- **Rolling Statistics**: Calculated rolling means and standard deviations for PM2.5, temperature, and humidity over 2, 3, 7, and 14-day windows.

### Preventing Data Leakage
- **Train-Test Split**: Data was split into training and test sets based on time, with the last 7 days reserved for testing.
- **Feature Creation**: Lag features and rolling statistics were calculated using `.shift(7)` to avoid using future data.
- **Cross-Validation**: Time-series cross-validation with `fold=5` and `seasonal_period="D"` was used to ensure robust model evaluation.

---

## ARIMA Model Training

### Initial Model Setup
- **Model**: ARIMA was chosen for its ability to handle trends and seasonality in time-series data.
- **Evaluation Metrics**: Initial results showed an MAE of 5.58, MAPE of 17.36%, and R² of -5.61, indicating room for improvement.

### Model Tuning
- **Tuning Method**: The `tune_model` function in PyCaret was used to optimize ARIMA parameters.
- **Results After Tuning**:
  - **MAE**: Reduced to 2.60 (53.4% improvement).
  - **MAPE**: Reduced to 8.09% (53.4% improvement).
  - **Accuracy**: Increased to 91.91% (11.2% improvement).
  - **R²**: Improved to -1.01.

### Comparison of Results
| Metric       | Before Tuning | After Tuning | Improvement |
|--------------|---------------|--------------|-------------|
| MAE          | 5.58          | 2.60         | 53.4%       |
| MAPE         | 17.36%        | 8.09%        | 53.4%       |
| Accuracy     | 82.64%        | 91.91%       | 11.2%       |
| R²           | -5.61         | -1.01        | Significant |

---

## Dashboard Development

### Key Components
1. **Prediction Display**:
   - **7-Day PM2.5 Forecast**: Visualized using line or bar charts.
   - **Historical vs. Predicted Comparison**: Shows actual vs. predicted PM2.5 levels.
   - **Daily Forecast Table**: Displays predicted PM2.5 values with confidence intervals.

2. **Parameter Adjustment**:
   - **ARIMA Parameter Control**: Allows users to adjust ARIMA parameters (p, d, q, P, D, Q).
   - **Parameter Impact Visualization**: Shows how parameter changes affect predictions.
   - **Model Performance Metrics**: Displays MAE, MAPE, and R² for model evaluation.

3. **Health Impact Analysis**:
   - **PM2.5 Level Map**: Displays sensor locations with color-coded markers (green: safe, yellow: moderate, red: hazardous).
   - **Health Risk Levels**: Based on AQI (Air Quality Index) standards.
   - **Public Recommendations**: Provides health advice based on PM2.5 levels.

### Dashboard Design
- **Color Scheme**: Cool tones (e.g., blue: `#1e3a8a`, light gray: `#e2e8f0`) for a professional look.
- **Responsive Layout**: Uses Flexbox and Grid Layout for adaptability across devices.
- **Animations**: Hover effects and transitions enhance user interaction.
- **Gradients and Shadows**: Adds depth and visual appeal to cards and components.

---

## Conclusion and Recommendations

### Summary
The project successfully developed a PM2.5 prediction system with an average accuracy of 91.91% (MAPE = 8.09%) for the tested area. However, when applied to other locations, the accuracy dropped to 70%, likely due to varying environmental factors.

### Recommendations for Future Development
1. **Incorporate External Variables**: Include traffic, weather, and pollution sources to improve prediction accuracy.
2. **Explore Advanced Models**: Test models like Prophet, LSTM, or other deep learning approaches.
3. **Enhance Data Cleaning**: Develop more sophisticated methods for outlier detection and handling.
4. **Real-time Monitoring**: Implement real-time data processing and forecasting capabilities.
5. **Increase Prediction Granularity**: Improve the model to provide hourly or location-specific forecasts.

This project demonstrates the potential of machine learning in addressing air pollution challenges and provides a foundation for future advancements in environmental monitoring and public health.

---

## Installation

To set up the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/teeranon124/team_project_pm2.5
   cd pm25-prediction-dashboard
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required libraries**:
   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, install the libraries manually:
   ```bash
   pip install dash dash-bootstrap-components dash-leaflet plotly pandas pycaret
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Access the dashboard**:
   Open your web browser and go to `http://127.0.0.1:8050/` to view the dashboard.

---

## Acknowledgments

- Thanks to the developers of **Dash**, **Plotly**, and **PyCaret** for their excellent libraries.
- Special thanks to the data providers for the PM2.5, humidity, and temperature data.

---