# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# ---------- PAGE SETUP ----------
st.set_page_config(
    page_title="Hydroponic Cultivation Predictor",
    page_icon="🌱",
    layout="wide"
)

# ---------- LOAD MODEL SAFELY ----------
@st.cache_resource
def load_models():
    model = joblib.load("models/hydroponics_model.pkl")
    le_plant = joblib.load("models/plant_encoder.pkl")
    le_stage = joblib.load("models/stage_encoder.pkl")
    return model, le_plant, le_stage

if not os.path.exists("models/hydroponics_model.pkl"):
    st.error("⚠️ Model files not found! Please run `train_model.py` first.")
    st.stop()

model, le_plant, le_stage = load_models()

# ---------- HEADER ----------
st.title("🌿 Smart Hydroponic Nutrient & Cultivation Predictor")
st.caption("Predict optimal cultivation days and nutrient needs based on environmental and crop parameters.")

# ---------- INPUT SECTION ----------
st.header("🌾 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    plant_type = st.selectbox("Plant Type", le_plant.classes_)
    growth_stage = st.selectbox("Growth Stage", le_stage.classes_)
    plant_count = st.number_input("Plant Count", min_value=10, max_value=100, step=1)

with col2:
    temperature = st.slider("Temperature (°C)", 22, 38, 28)
    humidity = st.slider("Humidity (%)", 40, 90, 65)
    light_intensity = st.number_input("Light Intensity (lux)", min_value=5000, max_value=40000, step=1000)

st.markdown("---")

# ---------- PREDICTION ----------
if st.button("🔍 Predict Results"):
    input_data = pd.DataFrame([[
        temperature,
        humidity,
        light_intensity,
        plant_count,
        le_stage.transform([growth_stage])[0],
        le_plant.transform([plant_type])[0]
    ]], columns=["Temperature", "Humidity", "Light_Intensity", "Plant_Count", "Growth_Stage", "Plant_Type"])

    prediction = model.predict(input_data)[0]

    # ---------- RESULT DISPLAY ----------
    st.header("📊 Prediction Summary")

    col_a, col_b = st.columns(2)

    with col_a:
        st.metric(label="🕓 Predicted Cultivation Days", value=f"{prediction[0]:.0f} days")
        st.metric(label="🌾 Total Nitrogen (N)", value=f"{prediction[1]:.2f}")
        st.metric(label="🌻 Total Phosphorus (P)", value=f"{prediction[2]:.2f}")

    with col_b:
        st.metric(label="🍃 Total Potassium (K)", value=f"{prediction[3]:.2f}")
        st.metric(label="💧 Total Calcium (Ca)", value=f"{prediction[4]:.2f}")
        st.metric(label="🌼 Total Magnesium (Mg)", value=f"{prediction[5]:.2f}")

    # ---------- NUTRIENT BAR CHART ----------
    nutrients = ["N", "P", "K", "Ca", "Mg"]
    values = prediction[1:]

    st.subheader("📈 Nutrient Requirement Chart")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(nutrients, values, color=["#81C784", "#AED581", "#FFD54F", "#4FC3F7", "#BA68C8"])
    ax.set_title("Predicted Nutrient Requirements", fontsize=13)
    ax.set_xlabel("Nutrients")
    ax.set_ylabel("Total Amount (relative units)")

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + 0.1, yval + 0.05, f"{yval:.2f}", fontsize=9)

    st.pyplot(fig)

    # ---------- SMART RECOMMENDATIONS ----------
    st.header("💡 Smart Recommendations")

    if "tomato" in plant_type.lower():
        if "vegetative" in growth_stage.lower():
            rec = "Increase nitrogen slightly for leaf growth and maintain temperature around 28°C."
        elif "flowering" in growth_stage.lower():
            rec = "Increase potassium and calcium for better fruit set and firmness."
        else:
            rec = "Maintain balanced NPK and avoid excess humidity."
    elif "chilli" in plant_type.lower():
        rec = "Keep humidity near 60% and ensure high potassium during flowering."
    elif "cucumber" in plant_type.lower():
        rec = "Maintain high humidity (70–80%) and balanced NPK levels."
    elif "brinjal" in plant_type.lower():
        rec = "Slightly higher nitrogen in early stage; increase potassium during fruiting."
    elif "bitter" in plant_type.lower():
        rec = "Ensure ample nitrogen and train vines properly to improve airflow."
    else:
        rec = "Maintain optimal nutrient balance and adjust light intensity to 20k–25k lux."

    st.success(rec)

# ---------- FOOTER ----------
st.markdown("---")
st.caption("AI-Powered Smart Hydroponics 🌱")
