# train_model.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


def train_model():
    # Load dataset (relative path, Streamlit-safe)
    data = pd.read_csv("Hydroponics_Indian_Plants_1000.csv")

    # Encode categorical columns
    le_plant = LabelEncoder()
    le_stage = LabelEncoder()

    data["Plant_Type"] = le_plant.fit_transform(data["Plant_Type"])
    data["Growth_Stage"] = le_stage.fit_transform(data["Growth_Stage"])

    # Features and targets
    X = data[
        ["Temperature", "Humidity", "Light_Intensity", "Plant_Count", "Growth_Stage", "Plant_Type"]
    ]

    Y = data[
        ["Predicted_Cultivation_Days", "Total_N", "Total_P", "Total_K", "Total_Ca", "Total_Mg"]
    ]

    # Train model
    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    )

    model.fit(X, Y)

    return model, le_plant, le_stage

