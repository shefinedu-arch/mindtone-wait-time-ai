"""
AI Wait-Time Prediction Model — Mindtone Mental Health Clinic
Compares Gradient Boosting, Random Forest, and XGBoost; selects the best
performer (by R²) as the production model.
"""
import pandas as pd
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import json

df = pd.read_csv("data/mindtone_wait_times.csv")

target = "actual_wait_time_min"
categorical = ["department", "day_of_week", "appointment_type", "booking_channel"]
numeric = ["doctor_experience_years", "is_weekend", "appointment_hour", "patient_age",
           "is_first_visit", "staff_on_duty", "doctors_on_duty", "patients_ahead_in_queue",
           "walkin_load", "avg_service_time_doctor_min"]

X = df[categorical + numeric]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
], remainder="passthrough")

models = {
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbosity=0),
}

results = {}
best_name, best_pipe, best_r2 = None, None, -np.inf

for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    train_time = time.time() - t0
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    results[name] = {"MAE_min": round(mae, 2), "RMSE_min": round(rmse, 2), "R2": round(r2, 3), "train_time_sec": round(train_time, 3)}
    if r2 > best_r2:
        best_name, best_pipe, best_r2 = name, pipe, r2

print("Model comparison:", json.dumps(results, indent=2))
print("Best model:", best_name)

with open("results/metrics.json", "w") as f:
    json.dump({"results": results, "best_model": best_name}, f, indent=2)

joblib.dump(best_pipe, "model/wait_time_model.joblib")

# ---- Chart 1: Actual vs Predicted ----
preds_best = best_pipe.predict(X_test)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds_best, alpha=0.35, s=15, color="#0F6E56")
lims = [0, max(y_test.max(), preds_best.max())]
plt.plot(lims, lims, "--", color="#D85A30", linewidth=1.5)
plt.xlabel("Actual Wait Time (min)")
plt.ylabel("Predicted Wait Time (min)")
plt.title(f"{best_name}: Actual vs Predicted Wait Time")
plt.tight_layout()
plt.savefig("results/actual_vs_predicted.png", dpi=150)
plt.close()

# ---- Chart 2: Feature importance (best model) ----
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
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=150)
plt.close()

# ---- Chart 3: Model comparison bar chart (MAE + R^2 across all 3 models) ----
names = list(results.keys())
maes = [results[n]["MAE_min"] for n in names]
r2s = [results[n]["R2"] for n in names]
x = np.arange(len(names))
fig, ax1 = plt.subplots(figsize=(7, 4.8))
ax1.bar(x - 0.15, maes, width=0.3, label="MAE (min)", color="#0F6E56")
ax1.set_ylabel("MAE (minutes)")
ax1.set_xticks(x)
ax1.set_xticklabels(names)
ax2 = ax1.twinx()
ax2.bar(x + 0.15, r2s, width=0.3, label="R\u00b2", color="#D85A30")
ax2.set_ylabel("R\u00b2 score")
fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))
plt.title("Model Comparison: MAE vs R\u00b2")
plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=150)
plt.close()

print("Saved model, metrics, and charts.")
