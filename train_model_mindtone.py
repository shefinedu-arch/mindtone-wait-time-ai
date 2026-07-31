"""
AI Wait-Time Prediction Model — Mindtone Mental Health Clinic
"""
import os
import json
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
# Disable font fallback checks that trigger FreeType FT_Open_Face errors on Windows
matplotlib.rcParams['font.enable_last_resort'] = False
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Ensure necessary output directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 1. Load Dataset
data_path = "data/mindtone_wait_times.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found at {data_path}. Please place 'mindtone_wait_times.csv' in the 'data/' folder.")

df = pd.read_csv(data_path)

# 2. Define Features & Target
target = "actual_wait_time_min"
categorical = ["department", "day_of_week", "appointment_type", "booking_channel"]
numeric = [
    "doctor_experience_years", "is_weekend", "appointment_hour", "patient_age",
    "is_first_visit", "staff_on_duty", "doctors_on_duty", "patients_ahead_in_queue",
    "walkin_load", "avg_service_time_doctor_min"
]

X = df[categorical + numeric]
y = df[target]

# 3. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Preprocessing Pipeline
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
], remainder="passthrough")

# 5. Define Candidate Models
models = {
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42),
}

results = {}
best_name, best_pipe, best_r2 = None, None, -np.inf

# 6. Model Evaluation Loop
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    
    results[name] = {
        "MAE_min": round(mae, 2),
        "RMSE_min": round(rmse, 2),
        "R2": round(r2, 3)
    }
    
    if r2 > best_r2:
        best_name, best_pipe, best_r2 = name, pipe, r2

print("Model comparison:", json.dumps(results, indent=2))
print("Best model:", best_name)

# 7. Save Metrics & Best Model
with open("results/metrics.json", "w") as f:
    json.dump({"results": results, "best_model": best_name}, f, indent=2)

joblib.dump(best_pipe, "model/wait_time_model.joblib")

# ---------------------------------------------------------
# Plot 1: Actual vs Predicted Scatter Plot
# ---------------------------------------------------------
preds_best = best_pipe.predict(X_test)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds_best, alpha=0.35, s=15, color="#0F6E56")
lims = [0, max(y_test.max(), preds_best.max())]
plt.plot(lims, lims, "--", color="#D85A30", linewidth=1.5)
plt.xlabel("Actual Wait Time (min)")
plt.ylabel("Predicted Wait Time (min)")
plt.title(f"{best_name}: Actual vs Predicted Wait Time")
plt.savefig("results/actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# Plot 2: Top 12 Feature Importances
# ---------------------------------------------------------
ohe = best_pipe.named_steps["prep"].named_transformers_["cat"]
cat_feature_names = list(ohe.get_feature_names_out(categorical))
all_feature_names = cat_feature_names + numeric
importances = best_pipe.named_steps["model"].feature_importances_

imp_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False).head(12)

plt.figure(figsize=(8, 6))
plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#0F6E56")
plt.xlabel("Importance")
plt.title(f"{best_name}: Top 12 Feature Importances")
plt.savefig("results/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# Plot 3: Model Comparison Bar Chart
# ---------------------------------------------------------
names = list(results.keys())
maes = [results[n]["MAE_min"] for n in names]
r2s = [results[n]["R2"] for n in names]
x = np.arange(len(names))

fig, ax1 = plt.subplots(figsize=(6, 4.5))
ax1.bar(x - 0.15, maes, width=0.3, label="MAE (min)", color="#0F6E56")
ax1.set_ylabel("MAE (minutes)")
ax1.set_xticks(x)
ax1.set_xticklabels(names)

ax2 = ax1.twinx()
ax2.bar(x + 0.15, r2s, width=0.3, label="R²", color="#D85A30")
ax2.set_ylabel("R² score")

fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))
plt.title("Model Comparison: MAE vs R²")
plt.savefig("results/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved model, metrics, and charts successfully.")