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
# 1) Load and prepare data
# ======================================================
df = pd.read_csv("cleaned_data.csv")

# ---------- Location cleaning & one-hot ----------
# [CHANGED] Normalize Location to 4 named categories and one-hot encode
df["Location"] = df["Location"].astype(str).str.strip()

# If earlier you had numeric codes (0..4), map them back to names
numeric_back_map = {
    "4": "Colombo Suburbs",
    "3": "Colombo",
    "2": "Other Urban",
    "1": "Other Rural",
    "0": "Unknown"
}
if df["Location"].str.fullmatch(r"\d+").any():
    df["Location"] = df["Location"].map(numeric_back_map).fillna("Unknown")

# Standardize common variants (optional)
alias_map = {
    "colombo": "Colombo",
    "colombo suburbs": "Colombo Suburbs",
    "other urban": "Other Urban",
    "other rural": "Other Rural"
}
df["Location"] = df["Location"].str.lower().map(alias_map).fillna(df["Location"])

expected_locs = ["Colombo", "Colombo Suburbs", "Other Urban", "Other Rural"]
df["Location"] = pd.Categorical(df["Location"], categories=expected_locs)
unknown_mask = df["Location"].isna()
if unknown_mask.any():
    print("[WARN] Found rows with unknown Location. Count:", int(unknown_mask.sum()))
    # keep them → all Loc_* will be 0 after get_dummies (or drop if you prefer)
# One-hot (creates exactly Loc_Colombo, Loc_Colombo Suburbs, Loc_Other Urban, Loc_Other Rural)
df = pd.get_dummies(df, columns=["Location"], prefix="Loc", drop_first=False)

# ---------- Other mappings ----------
furnish_map = {"Semi-Furnished": 3, "Furnished": 2, "Unfurnished": 1}
df["Furnishing"] = df["Furnishing"].map(furnish_map)
df["Car_Parking"] = df["Car_Parking"].astype(int)

# ---------- Feature set ----------
loc_cols = [c for c in df.columns if c.startswith("Loc_")]
# Optional engineered feature (uncomment to try)
df["Carpet_to_Super"] = (df["Carpet_Area"] / df["Super_Area"]).clip(0, 1)

#features = loc_cols + ["Super_Area", "Carpet_Area", "Bathroom", "Furnishing", "Car_Parking", "Balcony"]
# If you enabled Carpet_to_Super above:
features = loc_cols + ["Super_Area", "Carpet_Area", "Carpet_to_Super", "Bathroom", "Furnishing", "Car_Parking", "Balcony"]

X = df[features]
y = df["Amount_in_rupees"] * 3.4

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================================================
# 2) Baseline
# ======================================================
baseline_model = RandomForestRegressor(n_estimators=200, random_state=42)
baseline_model.fit(X_train, y_train)
y_pred_base = baseline_model.predict(X_test)
print("Baseline MAE:", mean_absolute_error(y_test, y_pred_base))
print("Baseline MSE:", mean_squared_error(y_test, y_pred_base))
print("Baseline R2 :", r2_score(y_test, y_pred_base))
baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_base))
mask_base = (y_test != 0)
baseline_mape = float(np.mean(np.abs((y_test[mask_base] - y_pred_base[mask_base]) / y_test[mask_base])) * 100)
print("Baseline RMSE:", baseline_rmse)
print(f"Baseline MAPE: {baseline_mape:.2f}%  (≈ Accuracy: {100 - baseline_mape:.2f}%)")

# ======================================================
# 3) Light casting
# ======================================================
X_train_ = X_train.astype(np.float32)
X_test_  = X_test.astype(np.float32)
y_train_ = y_train.astype(np.float32)
y_test_  = y_test.astype(np.float32)

# ======================================================
# 4) Hyperparameter tuning (RandomizedSearchCV)
# ======================================================
cv = KFold(n_splits=5, shuffle=True, random_state=42)
param_dist = {
    "n_estimators": np.arange(150, 900, 50),
    "max_depth": [None] + list(range(6, 26, 2)),
    "min_samples_split": np.arange(2, 21),
    "min_samples_leaf": np.arange(1, 21),
    "max_features": ["sqrt", "log2", 0.5, 0.7, 1.0],
    "bootstrap": [True],
    "max_leaf_nodes": [None, 64, 128, 256, 512],
    "ccp_alpha": np.append([0.0], np.logspace(-5, -2, 6)),
}

rf_base = RandomForestRegressor(random_state=42, n_jobs=-1, oob_score=True, bootstrap=True)
search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    n_iter=80,
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
print("[UPDATED] Tuned Test RMSE:", np.sqrt(mean_squared_error(y_test_, y_pred_tuned)))
print("[UPDATED] Tuned Test R2 :", r2_score(y_test_, y_pred_tuned))
mask_tuned = (y_test_ != 0)
tuned_mape = float(np.mean(np.abs((y_test_[mask_tuned] - y_pred_tuned[mask_tuned]) / y_test_[mask_tuned])) * 100)
print(f"[UPDATED] Tuned Test MAPE: {tuned_mape:.2f}%  (≈ Accuracy: {100 - tuned_mape:.2f}%)")

# ======================================================
# 5) Compact the forest by refitting at each size (no warm_start)
# ======================================================
def grow_until_plateau_refit(base_params, X, y, start=50, step=25, max_trees=900, patience=3, min_gain=1e-4):
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

chosen = best_rf.get_params()
for k in ["n_estimators", "warm_start", "n_jobs", "oob_score", "bootstrap", "random_state"]:
    chosen.pop(k, None)

rf_compact, best_oob, best_trees = grow_until_plateau_refit(chosen, X_train_, y_train_, start=50, step=25, max_trees=900)
print(f"\n[UPDATED] Compact forest -> trees: {best_trees}, OOB={best_oob:.4f}")

y_pred_compact = rf_compact.predict(X_test_)
print("[UPDATED] Compact Test MAE:", mean_absolute_error(y_test_, y_pred_compact))
print("[UPDATED] Compact Test RMSE:", np.sqrt(mean_squared_error(y_test_, y_pred_compact)))
print("[UPDATED] Compact Test R2 :", r2_score(y_test_, y_pred_compact))
mask_comp = (y_test_ != 0)
compact_mape = float(np.mean(np.abs((y_test_[mask_comp] - y_pred_compact[mask_comp]) / y_test_[mask_comp])) * 100)
print(f"[UPDATED] Compact Test MAPE: {compact_mape:.2f}%  (≈ Accuracy: {100 - compact_mape:.2f}%)")

# ======================================================
# 6) Permutation importance (do NOT drop Loc_* even if weak)
# ======================================================
perm = permutation_importance(rf_compact, X_test_, y_test_, n_repeats=10, random_state=42, n_jobs=-1)
pi = pd.Series(perm.importances_mean, index=X_test_.columns).sort_values()
print("\n[UPDATED] Permutation importance (mean):")
print(pi)

weak = list(pi[pi <= 0].index)
final_features = list(X_train_.columns)

# [CHANGED] If you want to prune, do not drop location columns so Location remains in the model
weak_nonloc = [c for c in weak if not c.startswith("Loc_")]
if weak_nonloc:
    print("\n[UPDATED] Dropping weak NON-Location features and re-fitting:", weak_nonloc)
    X_train_re = X_train_.drop(columns=weak_nonloc)
    X_test_re  = X_test_.drop(columns=weak_nonloc)

    rf_compact.fit(X_train_re, y_train_)
    y_pred_re = rf_compact.predict(X_test_re)
    print("[UPDATED] Refit(no-weak) Test MAE:", mean_absolute_error(y_test_, y_pred_re))
    print("[UPDATED] Refit(no-weak) Test RMSE:", np.sqrt(mean_squared_error(y_test_, y_pred_re)))
    print("[UPDATED] Refit(no-weak) Test R2 :", r2_score(y_test_, y_pred_re))
    mask_re = (y_test_ != 0)
    refit_mape = float(np.mean(np.abs((y_test_[mask_re] - y_pred_re[mask_re]) / y_test_[mask_re])) * 100)
    print(f"[UPDATED] Refit(no-weak) Test MAPE: {refit_mape:.2f}%  (≈ Accuracy: {100 - refit_mape:.2f}%)")
    final_features = list(X_train_re.columns)  # keep this smaller schema
else:
    print("\n[UPDATED] No weak non-location features to drop. Keeping all features.")
    final_features = list(X_train_.columns)

# ======================================================
# 7) Save model + feature schema
# ======================================================
model_path = "rf_price_model_xz.joblib"
dump(rf_compact, model_path, compress=("xz", 3))
size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f"\n[UPDATED] Saved compact model -> {model_path}  ({size_mb:.2f} MB)")

with open("model_features.json", "w") as f:
    json.dump({"features": final_features}, f, indent=2)
print("[UPDATED] Saved feature schema -> model_features.json:", final_features)

# ======================================================
# 8) Inference helper
# ======================================================
def predict_price(incoming_dict):
    import numpy as _np
    import pandas as _pd
    model = rf_compact
    x = _pd.DataFrame([incoming_dict])[final_features].astype(_np.float32)
    return float(model.predict(x)[0])
