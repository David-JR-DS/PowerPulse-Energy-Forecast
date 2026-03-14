import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PowerPulse ⚡",
    page_icon="⚡",
    layout="wide"
)


# ─────────────────────────────────────────────
# Path resolver (works locally & in deployment)
# ─────────────────────────────────────────────
def get_base():
    if os.path.exists(os.path.join("outputs", "models")):
        return ""
    elif os.path.exists(os.path.join("PowerPulse-Energy-Forecast", "outputs", "models")):
        return "PowerPulse-Energy-Forecast"
    else:
        return ""


BASE = get_base()


# ─────────────────────────────────────────────
# Load Model Artifacts (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path    = os.path.join(BASE, "outputs", "models", "best_model.pkl")
    scaler_path   = os.path.join(BASE, "data", "processed", "scaler.pkl")
    features_path = os.path.join(BASE, "data", "processed", "features.pkl")

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at: {model_path}")
        st.stop()
    if not os.path.exists(scaler_path):
        st.error(f"❌ Scaler not found at: {scaler_path}")
        st.stop()
    if not os.path.exists(features_path):
        st.error(f"❌ Features not found at: {features_path}")
        st.stop()

    model         = joblib.load(model_path)
    scaler        = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)

    # Find Global_intensity column so we can drop it AFTER scaling
    gi_name = None
    for cand in ["Global_intensity", "Globalintensity"]:
        if cand in feature_names:
            gi_name = cand
            break
    if gi_name is None:
        st.error(
            "Global_intensity (or Globalintensity) not found in features.pkl. "
            f"Features: {feature_names}"
        )
        st.stop()

    gi_idx = feature_names.index(gi_name)

    return model, scaler, feature_names, gi_name, gi_idx


model, scaler, feature_names, GI_NAME, GI_IDX = load_model()


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("⚡ PowerPulse — Household Energy Forecast")
st.markdown(
    "Predict **next‑hour household electricity consumption (kW)** using an "
    "XGBoost model trained on 2M+ real‑world measurements."
)
st.markdown("---")


# ─────────────────────────────────────────────
# Sidebar – About
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        **Model:** XGBoost Regressor  
        **R² Score:** 0.9760  
        **MAE (log):** 0.0252  
        **Training samples:** 1,660,206  
        **Test samples:** 415,052  

        **Top Features:**
        1. lag1h (prev hour usage)
        2. roll24h (24h rolling avg)
        3. Submetering3 (water heater/HVAC)
        4. Voltage

        **Dataset:**  
        UCI – Individual Household  
        Electric Power Consumption
        """
    )
    st.markdown("---")
    st.markdown("**Author:** David Raj")


# ─────────────────────────────────────────────
# Input Section
# ─────────────────────────────────────────────
st.subheader("⚙️ Enter Current Household Readings")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🔌 Electrical Readings**")
    voltage = st.slider("Voltage (V)", min_value=220.0, max_value=245.0, value=235.0, step=0.1)
    sub1    = st.slider("Sub‑meter 1 – Kitchen (Wh)", min_value=0.0, max_value=40.0, value=0.0, step=0.1)
    sub2    = st.slider("Sub‑meter 2 – Laundry (Wh)", min_value=0.0, max_value=25.0, value=0.0, step=0.1)
    sub3    = st.slider("Sub‑meter 3 – Water Heater/HVAC (Wh)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)

with col2:
    st.markdown("**⏰ Time Settings**")
    hour = st.slider("Hour of Day (0 = midnight)", min_value=0, max_value=23, value=18)
    dayofweek = st.selectbox(
        "Day of Week",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
        index=2
    )

    # Compute weekend internally (no info box shown)
    is_weekend = 1 if dayofweek >= 5 else 0

    # Hidden defaults for lag features (no inputs shown)
    lag1h   = 1.0   # typical log usage
    roll24h = 1.2   # typical log rolling avg

st.markdown("---")


# ─────────────────────────────────────────────
# Predict Button
# ─────────────────────────────────────────────
if st.button("🔮 Predict Next Hour Power Usage", type="primary", use_container_width=True):

    # Build time encodings
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)

            # Map from feature name → value (matches training exactly)
    feature_values = {
        "Voltage":        voltage,

        "Sub_metering_1": sub1,
        "Sub_metering_2": sub2,
        "Sub_metering_3": sub3,

        "hour_sin":       hour_sin,
        "hour_cos":       hour_cos,

        "dayofweek":      dayofweek,
        "is_weekend":     is_weekend,

        "lag_1h":         lag1h,
        "roll_24h":       roll24h,
    }



    # Rebuild the FULL 11‑feature vector in the original training order
    full_input = []
    for fname in feature_names:
        if fname == GI_NAME:
            # Dummy value – scaler needs 11 columns, but model never sees this one
            full_input.append(0.0)
        else:
            if fname not in feature_values:
                st.error(f"Feature '{fname}' is not provided by the app inputs.")
                st.stop()
            full_input.append(feature_values[fname])

    full_input = np.array(full_input, dtype=float).reshape(1, -1)

    # Scale all 11 features with the original scaler
    full_scaled = scaler.transform(full_input)

    # Drop Global_intensity column to match how the model was trained (10 features, no leakage)
    input_scaled_for_model = np.delete(full_scaled, GI_IDX, axis=1)

    # Predict (log scale → kW)
    pred_log = model.predict(input_scaled_for_model)[0]
    pred_kwh = np.expm1(pred_log)

    # ── Result Display ──
    st.markdown("### 🎯 Prediction Result")
    r1, r2, r3 = st.columns(3)
    r1.metric("⚡ Predicted Power", f"{pred_kwh:.3f} kW")
    r2.metric("📊 Model R² Score",  "0.9760")
    r3.metric("📉 Model MAE (log)", "0.0252")

    # ── Interpretation ──
    if pred_kwh < 0.5:
        level, color = "🟢 Low", "green"
    elif pred_kwh < 1.5:
        level, color = "🟡 Moderate", "orange"
    else:
        level, color = "🔴 High", "red"

    st.markdown(f"**Consumption Level:** {level}")

    # ── Simple bar chart ──
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.barh(["Predicted kW"], [pred_kwh], color=color, height=0.4)
    ax.set_xlim(0, max(3.0, pred_kwh * 1.3))
    ax.set_xlabel("kW")
    ax.set_title("Next‑Hour Power Consumption Estimate")
    ax.axvline(x=1.5, color="gray", linestyle="--", linewidth=0.8, label="High threshold (1.5 kW)")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

    st.balloons()

st.markdown("---")


# ─────────────────────────────────────────────
# Model Comparison Table
# ─────────────────────────────────────────────
with st.expander("📊 Full Model Comparison Results"):
    results = {
        "Model":        ["Linear Regression", "Random Forest", "XGBoost (Champion)"],
        "R² Score":     [0.9614, 0.9759, 0.9760],
        "RMSE (log)":   [0.0747, 0.0590, 0.0589],
        "MAE (log)":    [0.0359, 0.0243, 0.0252],
    }
    df_results = pd.DataFrame(results).set_index("Model")
    st.dataframe(
        df_results.style
        .format({"R² Score": "{:.4f}", "RMSE (log)": "{:.4f}", "MAE (log)": "{:.4f}"})
        .highlight_max(subset=["R² Score"], color="#d4edda")
        .highlight_min(subset=["RMSE (log)", "MAE (log)"], color="#d4edda"),
        use_container_width=True
    )


# ─────────────────────────────────────────────
# Feature Importance Plot
# ─────────────────────────────────────────────
with st.expander("📈 Feature Importance (XGBoost)"):
    feat_path = os.path.join(BASE, "outputs", "plots", "feature_importance.png")
    if os.path.exists(feat_path):
        st.image(feat_path, caption="Top Features – XGBoost", use_column_width=True)
    else:
        st.warning(f"Feature importance plot not found at: {feat_path}")


# ─────────────────────────────────────────────
# Model Predictions Plot
# ─────────────────────────────────────────────
with st.expander("📉 Model Predictions vs Actual"):
    pred_path = os.path.join(BASE, "outputs", "plots", "model_predictions.png")
    if os.path.exists(pred_path):
        st.image(pred_path, caption="Actual vs Predicted – All Models", use_column_width=True)
    else:
        st.warning(f"Predictions plot not found at: {pred_path}")


# ─────────────────────────────────────────────
# Residuals Plot
# ─────────────────────────────────────────────
with st.expander("🔍 Residuals Analysis"):
    res_path = os.path.join(BASE, "outputs", "plots", "residuals_analysis.png")
    if os.path.exists(res_path):
        st.image(res_path, caption="Residuals Analysis – XGBoost", use_column_width=True)
    else:
        st.warning(f"Residuals plot not found at: {res_path}")


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "**⚡ PowerPulse** &nbsp;|&nbsp; XGBoost R²=0.976 &nbsp;|&nbsp; "
    "2M+ Samples &nbsp;|&nbsp; No Data Leakage &nbsp;|&nbsp; "
    "**David Raj**"
)
