```markdown
# ⚡ PowerPulse — Household Energy Usage Forecast

PowerPulse is an end‑to‑end machine learning project that predicts **next‑hour household electricity consumption** using the **Individual Household Electric Power Consumption** dataset from the UCI Machine Learning Repository.

The project demonstrates a complete data science workflow:

- Exploratory Data Analysis (EDA)
- Data cleaning & preprocessing
- Time‑series feature engineering
- Model training & comparison (Linear Regression, Random Forest, XGBoost)
- Evaluation with robust metrics
- Deployment via a **Streamlit** prediction app

---

## 🎯 Objective

Build a regression model to predict **next‑hour Global Active Power (kW)** based on historical household power measurements and engineered time‑series features.

This enables:

- Smarter household energy management  
- Anticipation of peak demand  
- Potential reduction in electricity costs  
- A baseline for smarter grid and automation use‑cases  

---

## 📊 Dataset

**Source:** UCI Machine Learning Repository – *Individual Household Electric Power Consumption*.

**Original data:**

- ~2M+ timestamped measurements at 1‑minute resolution.
- Key raw fields include:
  - `Global_active_power` (kW, modeling target)
  - `Global_intensity`
  - `Voltage`
  - `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`
  - `Date` & `Time` columns

**Modeling data:**

- Aggregated and engineered to **hourly** level.
- Final modeling arrays (from notebooks):  
  - `X_train` shape: **(1,660,206, 10)**  
  - `X_test`  shape: **(415,052, 10)**  

The target used for modeling is a **log‑transformed** version of global active power, and predictions are inverse‑transformed with `expm1()` for interpretation in kW.

---

## 🔎 EDA Summary (High‑Level)

Using a structured EDA approach (15‑question guide from the project brief), the notebooks explore:

- **Data quality & missing values** – inspecting gaps and invalid readings.  
- **Distributions** – consumption is right‑skewed with occasional high‑load spikes.  
- **Correlations** – clear relationships between global power and sub‑meter signals.  
- **Temporal behavior** – visible daily and weekly patterns in energy use.  
- **Outliers** – extreme cases are inspected and treated so they don’t dominate training.  

Overall, the EDA shows that **recent usage and specific high‑load appliances** are strongly linked to short‑term demand.

---

## ⚙️ Data Preprocessing & Feature Engineering

All preprocessing logic is implemented in  
`notebooks/02_preprocessing_feature_engineering.ipynb`.

### 1️⃣ Cleaning & Resampling

- Time‑sorted the data.  
- Handled missing/invalid values with appropriate imputation/cleaning.  
- Aggregated 1‑minute readings to **hourly** level for modeling.  

### 2️⃣ Time‑Based Features

From the timestamp, the following were derived:

- `dayofweek` (0–6)  
- `isweekend` (0/1)  
- `hour` (0–23), then encoded as:
  - `hoursin = sin(2π·hour / 24)`
  - `hourcos = cos(2π·hour / 24)`

These capture **daily and weekly seasonality**.

### 3️⃣ Lag & Rolling Features

To incorporate temporal dependence in the model:

- `lag1h` – global active power (log scale) from the previous hour.  
- `roll24h` – 24‑hour rolling mean of global active power (log scale).  

These features let the model learn from **recent trends and daily patterns**.

### 4️⃣ Leakage Prevention

The column `Global_intensity` is physically derived from the same quantities as the target and introduces **data leakage** if used.

- It is explicitly removed from the feature matrix and from the saved feature list.
- The final model and the Streamlit app **do not use** `Global_intensity` at inference time.

### 5️⃣ Scaling & Artifacts

- Input features are standardized using `StandardScaler` (fitted on training data).
- Saved artifacts in `data/processed/`:
  - `scaler.pkl` – fitted scaler
  - `features.pkl` – list of **final 10 features**
  - `X_train.npy.7z`, `X_test.npy.7z`, `df_cleaned.csv.7z` – compressed arrays/CSV for GitHub
  - `y_train.npy`, `y_test.npy` – target arrays

**Final feature set used by the champion model:**

```text
['Voltage',
 'Submetering1',
 'Submetering2',
 'Submetering3',
 'hoursin',
 'hourcos',
 'dayofweek',
 'isweekend',
 'lag1h',
 'roll24h']
```

---

## 🤖 Modeling & Results

Modeling and evaluation are implemented in  
`notebooks/03_modeling_evaluation.ipynb`.

### Models Trained

| Model               | Description                         |
|---------------------|-------------------------------------|
| Linear Regression   | Simple baseline regression          |
| Random Forest       | Tree‑based ensemble                 |
| XGBoost Regressor   | Gradient boosting, champion model   |

### Time‑Series Aware Validation

- Train/test split respects **time order** (no shuffling).  
- Internal cross‑validation uses **TimeSeriesSplit** to avoid future data leaking into training.  

### Performance Summary (Log‑Transformed Target)

On the large held‑out test set:

| Model             | R²     | RMSE (log) | MAE (log) |
|-------------------|--------|-----------:|----------:|
| Linear Regression | 0.9614 | 0.0747     | 0.0359    |
| Random Forest     | 0.9759 | 0.0590     | 0.0243    |
| **XGBoost**       | **0.9760** | **0.0589** | **0.0252** |

- RMSE / MAE above are in **log‑kW space**.  
- When converted back to kW, the XGBoost model’s errors are on the order of **a few tenths of a kW** (approximate).

**Champion model:** XGBoost Regressor saved as:

- `outputs/models/best_model.pkl`  

Additional evaluation artifacts:

- `outputs/models/results_summary.pkl`  
- `outputs/plots/model_predictions.png`  
- `outputs/plots/feature_importance.png`  
- `outputs/plots/residuals_analysis.png`  

---

## 🔍 Feature Importance — What Drives Usage?

From the XGBoost feature importance analysis:

**Most influential features:**

1. **`lag1h`** – Previous hour’s consumption (log).  
2. **`roll24h`** – 24‑hour rolling average (log).  
3. **`Submetering3`** – Heavy loads like water heater / HVAC.  
4. **`Voltage`** – Reflects loading conditions on the grid.  
5. Time features: `hoursin`, `hourcos`, `dayofweek`, `isweekend`.  

**Takeaway:**  
Short‑term electricity demand is strongly driven by **recent usage patterns** plus specific high‑consumption appliances, with clear daily/weekly structure.

---

## 💡 Business & Energy Insights

From the modeling and feature analysis:

- **Recent history is critical:**  
  The previous hour and last 24‑hours of usage are the strongest predictors → short‑term forecasting is highly reliable.

- **Appliance groups matter:**  
  The `Submetering3` channel (e.g., water heater / HVAC) is a major contributor to peaks, more so than kitchen or laundry loads.

- **Temporal patterns are stable:**  
  The model benefits from hour‑of‑day and weekday/weekend features, reflecting consistent human activity patterns.

**Applications:**

- Energy providers can use such a model as a **short‑term demand forecasting tool** for load planning and pricing.
- Households can use forecasts to schedule heavy appliances during cheaper/off‑peak periods.

---

## 📊 Visualizations

Key visual outputs (in `outputs/plots/`):

- **Model predictions vs actual** – `model_predictions.png`  
- **Residual analysis** – `residuals_analysis.png`  
  - Distribution of errors  
  - Residuals vs predictions  
- **Feature importance** – `feature_importance.png`  

These plots support both **technical evaluation** and **storytelling**.

---

## 🏗 Final Project Structure

```text
PowerPulse-Energy-Forecast/
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── processed/
│       ├── X_test.npy.7z
│       ├── X_train.npy.7z
│       ├── df_cleaned.csv.7z
│       ├── features.pkl
│       ├── scaler.pkl
│       ├── y_test.npy
│       ├── y_train.npy
│       └── raw/
│           └── household_power_consumption... (original / compressed file)
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_preprocessing_feature_engineering.ipynb
│   └── 03_modeling_evaluation.ipynb
│
└── outputs/
    ├── models/
    │   ├── best_model.pkl
    │   └── results_summary.pkl
    └── plots/
        ├── feature_importance.png
        ├── model_predictions.png
        └── residuals_analysis.png
```

---

## 🖥️ Streamlit App (Deployment)

The Streamlit app (`app.py`) provides an interactive interface for live prediction using the saved artifacts.

**Core behavior:**

1. **Load artifacts (cached):**
   ```python
   model, scaler, feature_names = load_model()
   # best_model.pkl, scaler.pkl, features.pkl
   ```

2. **Collect user inputs via sliders:**
   - Voltage  
   - Sub‑meter 1 (Kitchen)  
   - Sub‑meter 2 (Laundry)  
   - Sub‑meter 3 (Water heater / HVAC)  
   - Hour of day (0–23)

3. **Rebuild exact 10‑feature vector (NO `Global_intensity`):**
   ```python
   hour_sin = np.sin(2 * np.pi * hour / 24.0)
   hour_cos = np.cos(2 * np.pi * hour / 24.0)

   input_data = np.array([[
       voltage,
       sub1, sub2, sub3,
       hour_sin, hour_cos,
       2, 0,       # example: Wednesday, not weekend
       1.0, 1.2    # default lag1h, roll24h
   ]])
   ```

4. **Scale + predict:**
   ```python
   input_scaled = scaler.transform(input_data)
   pred_log = model.predict(input_scaled)
   pred_kwh = np.expm1(pred_log)
   ```

5. **Display:**
   - Predicted next‑hour power (kW).  
   - Reference to model accuracy (R² ≈ 0.976).  
   - Optional: model comparison table and feature importance image.  

---

## 🛠 Tech Stack

- **Language:** Python  
- **Core Libraries:**
  - Data: `pandas`, `numpy`
  - Modeling: `scikit-learn`, `xgboost`
  - Serialization: `joblib`
  - Visualization: `matplotlib`, `seaborn`
  - App: `streamlit`
- **Concepts:**
  - Time‑series regression
  - Feature engineering (lags, rolling windows, cyclic encodings)
  - Data leakage prevention
  - TimeSeriesSplit cross‑validation
  - Model deployment with Streamlit

---

## 🚀 How to Run

1️⃣ **Clone the repo**
```bash
git clone https://github.com/<your-username>/PowerPulse-Energy-Forecast.git
cd PowerPulse-Energy-Forecast
```

2️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

3️⃣ **Ensure processed data & models are present**

You should have (either from the repo or by re‑running notebooks):

- `data/processed/scaler.pkl`
- `data/processed/features.pkl`
- `outputs/models/best_model.pkl`
- `outputs/plots/feature_importance.png` (and other plots)
- Optionally the compressed arrays: `X_train.npy.7z`, `X_test.npy.7z`, `df_cleaned.csv.7z`

4️⃣ **Run the Streamlit app**
```bash
streamlit run app.py
```

Then open the URL shown in your terminal (usually `http://localhost:8501`).

---

## 📌 Business Value Summary

This project demonstrates how ML can be applied in the **Energy & Utilities** domain to:

- Forecast short‑term household electricity demand  
- Identify and quantify peak periods  
- Reveal which appliances and behaviors drive usage  
- Provide a baseline for smart‑grid, dynamic pricing, and home automation solutions  

It combines solid modeling results (**R² ≈ 0.976**) with an interactive app, making it suitable for both technical evaluation and real‑world demonstration.

---

## 👤 Author

**David John Raj D**  
Business Process Lead – Operations, TCS  

**Skills:** Python, SQL, Machine Learning, Data Analysis, Power BI
```

You can paste this directly into your `README.md`.
