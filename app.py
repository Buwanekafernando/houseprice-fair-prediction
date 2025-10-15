from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import pandas as pd
from joblib import load

# [CHANGED] point Flask to "frontend" folder (your index.html lives here)
app = Flask(__name__, template_folder="frontend")

MODEL_PATH = "rf_price_model_xz.joblib"
FEATS_PATH = "model_features.json"

model = None
final_features = None

try:
    model = load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")

try:
    with open(FEATS_PATH, "r") as f:
        final_features = json.load(f)["features"]
    print("✅ Feature schema loaded:", final_features)
except Exception as e:
    print(f"❌ Failed to load feature schema: {e}")

# Furnishing numeric map (same as training)
furnish_map  = {"Furnished": 3, "Semi-Furnished": 2, "Unfurnished": 1}

def make_model_input(payload: dict) -> pd.DataFrame:
    """
    Build a 1-row DataFrame with the SAME encoding used in training:
    - one-hot for Location (Loc_*)
    - numeric features cast to float32
    - Carpet_Area ≤ Super_Area validation
    """
    required = ["location","super_area","carpet_area","bathroom","furnishing","car_parking","balcony","price"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Parse numerics
    try:
        super_area  = float(payload["super_area"])
        carpet_area = float(payload["carpet_area"])
        bathroom    = int(payload["bathroom"])
        car_parking = int(payload["car_parking"])
        balcony     = int(payload["balcony"])
        price       = float(payload["price"])
    except Exception:
        raise ValueError("One or more numeric fields are invalid.")

    if super_area <= 0 or carpet_area <= 0:
        raise ValueError("Areas must be greater than zero.")
    if carpet_area > super_area:  # [CHANGED] strong validation
        raise ValueError("Carpet Area cannot be greater than Super Area.")
    if price <= 0:
        raise ValueError("Price must be greater than zero.")

    # Furnishing -> numeric
    try:
        furn_val = furnish_map[payload["furnishing"]]
    except KeyError:
        raise ValueError(f"Unknown furnishing: {payload['furnishing']}")

    location_str = str(payload["location"]).strip()

    # Base row (non-location)
    row = {
        "Super_Area":  super_area,
        "Carpet_Area": carpet_area,
        "Bathroom":    bathroom,
        "Furnishing":  furn_val,
        "Car_Parking": car_parking,
        "Balcony":     balcony,
    }

    # [CHANGED] one-hot Location based on the trained schema
    if not final_features:
        raise RuntimeError("Feature schema not loaded.")
    loc_cols = [c for c in final_features if c.startswith("Loc_")]
    for c in loc_cols:
        row[c] = 0.0
    loc_key = f"Loc_{location_str}"
    if loc_key in loc_cols:
        row[loc_key] = 1.0
    else:
        # unseen location -> keep all zeros, or raise error if you prefer
        pass

    # If you added Carpet_to_Super in training, also compute it here:
    # row["Carpet_to_Super"] = max(0.0, min(1.0, carpet_area / super_area))
    if "Carpet_to_Super" in final_features:
        row["Carpet_to_Super"] = max(0.0, min(1.0, carpet_area / super_area))


    # Align to exact order and dtype
    df = pd.DataFrame([row], columns=final_features).astype(np.float32)
    return df

def classify_fairness(user_df: pd.DataFrame, entered_price: float, tolerance: float = 0.30):
    fair_price = float(model.predict(user_df)[0])
    if fair_price <= 0:
        return {
            "status": "Check Data",
            "message": "Predicted fair price is non-positive; verify inputs.",
            "fair_price": round(fair_price, 2),
            "entered_price": entered_price,
            "difference": round(entered_price - fair_price, 2),
            "percentage_diff": None
        }
    pct_diff = ((entered_price - fair_price) / fair_price) * 100.0
    if entered_price > fair_price * (1 + tolerance):
        status = "Overpriced"
    elif entered_price < fair_price * (1 - tolerance):
        status = "Underpriced"
    else:
        status = "Fair"
    return {
        "status": status,
        "fair_price": round(fair_price, 2),
        "entered_price": round(entered_price, 2),
        "difference": round(entered_price - fair_price, 2),
        "percentage_diff": round(pct_diff, 2)
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "model_loaded": model is not None,
        "features_loaded": final_features is not None
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if model is None or final_features is None:
            return jsonify({"error": "Model or feature schema not loaded."}), 500

        data = request.get_json(force=True)

        # Build model input (performs validation inside)
        user_df = make_model_input(data)
        entered_price = float(data["price"])

        result = classify_fairness(user_df, entered_price, tolerance=0.30)
        # [CHANGED] echo features used (handy for debugging)
        result["used_features"] = user_df.iloc[0].astype(float).to_dict()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
