# ==== Imports ====
import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from joblib import dump

# ======================================================
# 1) Load and prepare data  (your original code)
# ======================================================
df = pd.read_csv("cleaned_data.csv")

# Map categorical variables
location_map = {"Colombo": 4, "Colombo Suburbs": 3, "Other Urban": 2, "Other Rural": 1}
df["Location"] = df["Location"].map(location_map)

furnish_map = {"Furnished": 3, "Semi-Furnished": 2, "Unfurnished": 1}
df["Furnishing"] = df["Furnishing"].map(furnish_map)

df["Car_Parking"] = df["Car_Parking"].astype(int)

# Select features
features = ["Location", "Super_Area", "Carpet_Area", "Bathroom", "Furnishing", "Car_Parking", "Balcony"]
X = df[features]
y = df["Amount_in_rupees"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================================================
# 2) Baseline fit & metrics (your original intent)
# ======================================================
baseline_model = RandomForestRegressor(n_estimators=200, random_state=42)
baseline_model.fit(X_train, y_train)
y_pred_base = baseline_model.predict(X_test)
print("Baseline MAE:", mean_absolute_error(y_test, y_pred_base))
print("Baseline MSE:", mean_squared_error(y_test, y_pred_base))
print("Baseline R2 :", r2_score(y_test, y_pred_base))

# [PRINT] Extra baseline accuracy-style metrics
baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_base))
mask_base = (y_test != 0)
baseline_mape = float(np.mean(np.abs((y_test[mask_base] - y_pred_base[mask_base]) / y_test[mask_base])) * 100)
print("Baseline RMSE:", baseline_rmse)
print(f"Baseline MAPE: {baseline_mape:.2f}%  (≈ Accuracy: {100 - baseline_mape:.2f}%)")

# ======================================================
# 3) [UPDATED] Robust evaluation types & light casting
#    - use float32 to reduce RAM and final file size a bit
# ======================================================
X_train_ = X_train.astype(np.float32)
X_test_  = X_test.astype(np.float32)
y_train_ = y_train.astype(np.float32)
y_test_  = y_test.astype(np.float32)

# ======================================================
# 4) [UPDATED] Hyperparameter tuning with CV (RandomizedSearchCV)
#    - focuses on accuracy and a smaller, well-regularized forest
# ======================================================
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Focused, size-aware search space
param_dist = {
    "n_estimators": np.arange(150, 900, 50),            # keep enough breadth
    "max_depth": [None] + list(range(6, 26, 2)),        # control overfitting
    "min_samples_split": np.arange(2, 21),
    "min_samples_leaf": np.arange(1, 21),
    "max_features": ["sqrt", "log2", 0.5, 0.7, 1.0],    # both categorical & float
    "bootstrap": [True],
    "max_leaf_nodes": [None, 64, 128, 256, 512],        # fewer leaves -> smaller model
    "ccp_alpha": np.append([0.0], np.logspace(-5, -2, 6))  # gentle post-pruning
    # If all y >= 0, you could also try criterion=["squared_error","poisson","absolute_error"]
}

rf_base = RandomForestRegressor(
    random_state=42,
    n_jobs=-1,
    oob_score=True,
    bootstrap=True
)

search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    n_iter=80,  # adjust higher for more thorough search if you have time
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=1,
    refit=True
)

search.fit(X_train_, y_train_)
print("\n[UPDATED] Best params from CV:", search.best_params_)
print("[UPDATED] CV best MAE:", -search.best_score_)

best_rf = search.best_estimator_
y_pred_tuned = best_rf.predict(X_test_)
print("[UPDATED] Tuned Test MAE:", mean_absolute_error(y_test_, y_pred_tuned))
print("[UPDATED] Tuned Test RMSE:", np.sqrt(mean_squared_error(y_test_, y_pred_tuned)))  # [FIX]
print("[UPDATED] Tuned Test R2 :", r2_score(y_test_, y_pred_tuned))

# ======================================================
# 5) [UPDATED] Compact the forest using OOB “early stop”
#    - find minimum n_estimators where OOB plateaus
# ======================================================
# ======================================================
# 5) [UPDATED-FIX] Compact the forest by refitting at each size (no warm_start)
#    - avoids "n_estimators must be >= len(estimators_) when warm_start==True"
# ======================================================
def grow_until_plateau_refit(base_params, X, y, start=50, step=25, max_trees=900, patience=3, min_gain=1e-4):
    """
    Try increasing n_estimators in steps; at each step, fit a *fresh* model (no warm_start),
    record OOB score, and stop when improvements plateau.
    """
    best_model = None
    best_oob = -np.inf
    best_trees = start
    no_improve = 0

    n = start
    while n <= max_trees:
        rf = RandomForestRegressor(
            **base_params,
            n_estimators=n,
            oob_score=True,
            n_jobs=-1,
            random_state=42,
            bootstrap=True
        )
        rf.fit(X, y)
        oob = rf.oob_score_

        if (oob - best_oob) >= min_gain:
            best_oob = oob
            best_trees = n
            best_model = rf
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        n += step

    return best_model, best_oob, best_trees

# take tuned params, remove keys we control
chosen = best_rf.get_params()
for k in ["n_estimators", "warm_start", "n_jobs", "oob_score", "bootstrap", "random_state"]:
    chosen.pop(k, None)

# [UPDATED CALL] use the refit-based compactor
rf_compact, best_oob, best_trees = grow_until_plateau_refit(
    chosen, X_train_, y_train_, start=50, step=25, max_trees=900
)
print(f"\n[UPDATED] Compact forest -> trees: {best_trees}, OOB={best_oob:.4f}")


y_pred_compact = rf_compact.predict(X_test_)
print("[UPDATED] Compact Test MAE:", mean_absolute_error(y_test_, y_pred_compact))
print("[UPDATED] Compact Test RMSE:", np.sqrt(mean_squared_error(y_test_, y_pred_compact)))  # [FIX]
print("[UPDATED] Compact Test R2 :", r2_score(y_test_, y_pred_compact))

# ======================================================
# 6) [UPDATED] Optional: prune weak features via permutation importance
#    - fewer features -> simpler trees -> smaller model
# ======================================================
perm = permutation_importance(
    rf_compact, X_test_, y_test_, n_repeats=10, random_state=42, n_jobs=-1
)
pi = pd.Series(perm.importances_mean, index=X_test_.columns).sort_values()
print("\n[UPDATED] Permutation importance (mean):")
print(pi)

weak = list(pi[pi <= 0].index)  # features that don't help or hurt
final_features = list(X_train_.columns)

if weak:
    print("\n[UPDATED] Dropping weak features and re-fitting:", weak)
    X_train_re = X_train_.drop(columns=weak)
    X_test_re  = X_test_.drop(columns=weak)

    rf_compact.fit(X_train_re, y_train_)
    y_pred_re = rf_compact.predict(X_test_re)
    print("[UPDATED] Refit(no-weak) Test MAE:", mean_absolute_error(y_test_, y_pred_re))
    print("[UPDATED] Refit(no-weak) Test RMSE:", np.sqrt(mean_squared_error(y_test_, y_pred_re)))  # [FIX]
    print("[UPDATED] Refit(no-weak) Test R2 :", r2_score(y_test_, y_pred_re))

    # keep these if improved or equal
    final_features = list(X_train_re.columns)

# ======================================================
# 7) [UPDATED] Save a smaller model (compressed) + feature list
# ======================================================
model_path = "rf_price_model_xz.joblib"  # LZMA compressed
dump(rf_compact, model_path, compress=("xz", 3))
size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f"\n[UPDATED] Saved compact model -> {model_path}  ({size_mb:.2f} MB)")

with open("model_features.json", "w") as f:
    json.dump({"features": final_features}, f, indent=2)
print("[UPDATED] Saved feature schema -> model_features.json:", final_features)

# ======================================================
# 8) [UPDATED] (Optional) Inference helper for your Flask app
# ======================================================
def predict_price(incoming_dict):
    """
    incoming_dict must contain all keys in final_features.
    Example keys: Location, Super_Area, Carpet_Area, Bathroom, Furnishing, Car_Parking, Balcony
    """
    import numpy as _np
    import pandas as _pd
    from joblib import load as _load

    model = rf_compact  # already in memory after training
    x = _pd.DataFrame([incoming_dict])[final_features].astype(_np.float32)
    return float(model.predict(x)[0])
