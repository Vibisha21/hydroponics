# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv("data/Hydroponics_Indian_Plants_1000.csv")

# Encode categorical columns
le_plant = LabelEncoder()
le_stage = LabelEncoder()
data["Plant_Type"] = le_plant.fit_transform(data["Plant_Type"])
data["Growth_Stage"] = le_stage.fit_transform(data["Growth_Stage"])

# Features and Targets
X = data[["Temperature", "Humidity", "Light_Intensity", "Plant_Count", "Growth_Stage", "Plant_Type"]]
Y = data[["Predicted_Cultivation_Days", "Total_N", "Total_P", "Total_K", "Total_Ca", "Total_Mg"]]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
model.fit(X_train, Y_train)

# Predictions
Y_pred = model.predict(X_test)

# Evaluate model for each target
metrics = {}
for i, col in enumerate(Y.columns):
    r2 = r2_score(Y_test[col], Y_pred[:, i])
    mae = mean_absolute_error(Y_test[col], Y_pred[:, i])
    mse = mean_squared_error(Y_test[col], Y_pred[:, i])
    metrics[col] = {"R2": r2, "MAE": mae, "MSE": mse}

# Print performance metrics
print("\n🌱 HYDROPONICS MODEL PERFORMANCE REPORT 🌱")
print("=" * 60)
for col, vals in metrics.items():
    print(f"\nTarget: {col}")
    print(f"  R² Score: {vals['R2']:.4f}")
    print(f"  Mean Absolute Error (MAE): {vals['MAE']:.4f}")
    print(f"  Mean Squared Error (MSE): {vals['MSE']:.4f}")

# Calculate average scores
avg_r2 = np.mean([v["R2"] for v in metrics.values()])
avg_mae = np.mean([v["MAE"] for v in metrics.values()])
avg_mse = np.mean([v["MSE"] for v in metrics.values()])

print("\n📊 AVERAGE MODEL PERFORMANCE")
print("=" * 60)
print(f"Average R² Score: {avg_r2:.4f}")
print(f"Average MAE: {avg_mae:.4f}")
print(f"Average MSE: {avg_mse:.4f}")

# Feature importances (from one of the regressors)
importances = model.estimators_[0].feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n🌿 FEATURE IMPORTANCE (Sample Estimator)")
print("=" * 60)
for _, row in importance_df.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")

# Save model and encoders
joblib.dump(model, "models/hydroponics_model.pkl")
joblib.dump(le_plant, "models/plant_encoder.pkl")
joblib.dump(le_stage, "models/stage_encoder.pkl")

print("\n✅ Model training complete and saved in 'models/' folder.")
