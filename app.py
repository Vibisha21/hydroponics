# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from train_model import train_model

# ---------- PAGE SETUP ----------
st.set_page_config(
    page_title="Hydroponic Cultivation Predictor",
    page_icon="🌱",
    layout="wide"
)

# ---------- LOAD / TRAIN MODEL ----------
@st.cache_resource
def load_models():
    return train_model()

model, le_plant, le_stage = load_models()

# ---------- HEADER ----------
st.title("🌿 Smart Hydroponic Nutrient & Cultivation Predictor")
st.caption(
    "Predict optimal cultivation days and nutrient needs based on environmental and crop parameters."
)

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
    light_intensity = st.number_input(
        "Light Intensity (lux)", min_value=5000, max_value=40000, step=1000
    )

st.markdown("---")

# ---------- PREDICTION ----------
if st.button("🔍 Predict Results"):
    input_data = pd.DataFrame(
        [[
            temperature,
            humidity,
            light_intensity,
            plant_count,
            le_stage.transform([growth_stage])[0],
            le_plant.transform([plant_type])[0]
        ]],
        columns=[
            "Temperature",
            "Humidity",
            "Light_Intensity",
            "Plant_Count",
            "Growth_Stage",
            "Plant_Type"
        ]
    )

    prediction = model.predict(input_data)[0]

    # ---------- RESULT DISPLAY ----------
    st.header("📊 Prediction Summary")

    col_a, col_b = st.columns(2)

    with col_a:
        st.metric("🕓 Predicted Cultivation Days", f"{prediction[0]:.0f} days")
        st.metric("🌾 Total Nitrogen (N)", f"{prediction[1]:.2f}")
        st.metric("🌻 Total Phosphorus (P)", f"{prediction[2]:.2f}")

    with col_b:
        st.metric("🍃 Total Potassium (K)", f"{prediction[3]:.2f}")
        st.metric("💧 Total Calcium (Ca)", f"{prediction[4]:.2f}")
        st.metric("🌼 Total Magnesium (Mg)", f"{prediction[5]:.2f}")

    # ---------- NUTRIENT BAR CHART ----------
    st.subheader("📈 Nutrient Requirement Chart")

    nutrients = ["N", "P", "K", "Ca", "Mg"]
    values = prediction[1:]

    fig, ax = plt.subplots()
    ax.bar(nutrients, values)
    ax.set_xlabel("Nutrients")
    ax.set_ylabel("Total Amount (relative units)")
    ax.set_title("Predicted Nutrient Requirements")

    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")

    st.pyplot(fig)

    # ---------- SMART RECOMMENDATIONS ----------
    st.header("💡 Smart Recommendations")

    plant = plant_type.lower()
    stage = growth_stage.lower()

    if "tomato" in plant:
        if "vegetative" in stage:
            rec = "Increase nitrogen slightly for leaf growth and maintain temperature around 28°C."
        elif "flowering" in stage:
            rec = "Increase potassium and calcium for better fruit set and firmness."
        else:
            rec = "Maintain balanced NPK and avoid excess humidity."
    elif "chilli" in plant:
        rec = "Keep humidity near 60% and ensure high potassium during flowering."
    elif "cucumber" in plant:
        rec = "Maintain high humidity (70–80%) and balanced NPK levels."
    elif "brinjal" in plant:
        rec = "Use higher nitrogen in early stages; increase potassium during fruiting."
    elif "bitter" in plant:
        rec = "Ensure ample nitrogen and train vines properly to improve airflow."
    else:
        rec = "Maintain optimal nutrient balance and adjust light intensity to 20k–25k lux."

    st.success(rec)

# ---------- FOOTER ----------
st.markdown("---")
st.caption("AI-Powered Smart Hydroponics 🌱")

