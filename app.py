# app.py  (Flask API for price fairness)  — UPDATED to use tuned & compact model
from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import pandas as pd

# [UPDATED] use joblib instead of pickle (compressed .joblib model)
from joblib import load

app = Flask(__name__ , template_folder="frontend")

# -----------------------------
# Model + feature schema loading
# -----------------------------
MODEL_PATH = "rf_price_model_xz.joblib"       # [UPDATED]
FEATS_PATH = "model_features.json"            # [UPDATED]

model = None
final_features = None

try:
    model = load(MODEL_PATH)  # tuned + compact RF with compression
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")

try:
    with open(FEATS_PATH, "r") as f:
        final_features = json.load(f)["features"]
    print("✅ Feature schema loaded:", final_features)
except Exception as e:
    print(f"❌ Failed to load feature schema: {e}")

# -----------------------------
# Mappings (must match training)
# -----------------------------
location_map = {"Colombo": 4, "Colombo Suburbs": 3, "Other Urban": 2, "Other Rural": 1}
furnish_map  = {"Furnished": 3, "Semi-Furnished": 2, "Unfurnished": 1}

# -----------------------------
# Helpers
# -----------------------------
def make_model_input(payload: dict) -> pd.DataFrame:
    """
    Convert incoming JSON to the model-ready DataFrame:
    - Apply the SAME mappings used in training
    - Keep EXACT column order expected by the model (final_features)
    - Cast to float32 to match training dtype
    """
    # Validate required keys early for clearer messages
    required_keys = [
        "location", "super_area", "carpet_area", "bathroom",
        "furnishing", "car_parking", "balcony", "price"
    ]
    missing = [k for k in required_keys if k not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Map categorical to numeric (raise if value not recognized)
    try:
        loc_val = location_map[payload["location"]]
    except KeyError:
        raise ValueError(f"Unknown location: {payload['location']}")

    try:
        furn_val = furnish_map[payload["furnishing"]]
    except KeyError:
        raise ValueError(f"Unknown furnishing: {payload['furnishing']}")

    # Build a single-row dict in *training feature names*
    row = {
        "Location":      loc_val,
        "Super_Area":    float(payload["super_area"]),
        "Carpet_Area":   float(payload["carpet_area"]),
        "Bathroom":      int(payload["bathroom"]),
        "Furnishing":    furn_val,
        "Car_Parking":   int(payload["car_parking"]),
        "Balcony":       int(payload["balcony"]),
    }

    # If you dropped weak features during training,
    # final_features may be a subset of the above; select/align accordingly:
    if not final_features:
        raise RuntimeError("Feature schema not loaded.")

    # Create DF with exact column order expected by the model
    df = pd.DataFrame([row])[final_features].astype(np.float32)  # [UPDATED] dtype match
    return df

def classify_fairness(user_df: pd.DataFrame, entered_price: float, tolerance: float = 0.30):
    """
    Classify price as Fair/Overpriced/Underpriced using model prediction and tolerance band.
    - tolerance=0.30 => ±30% band around predicted fair price
    """
    fair_price = float(model.predict(user_df)[0])

    # Guard against edge cases
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

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    """Simple health/ready check."""
    return jsonify({
        "model_loaded": model is not None,
        "features_loaded": final_features is not None
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze property price fairness."""
    try:
        if model is None or final_features is None:
            return jsonify({"error": "Model or feature schema not loaded. Train first or place files correctly."}), 500

        data = request.get_json(force=True)

        # Basic input validation for entered price
        try:
            entered_price = float(data["price"])
        except Exception:
            return jsonify({"error": "Invalid 'price' value."}), 400

        if entered_price <= 0:
            return jsonify({"error": "Price must be greater than zero."}), 400

        # Build model input in correct order & dtype
        user_df = make_model_input(data)

        # Classify fairness
        result = classify_fairness(user_df, entered_price, tolerance=0.30)  # keep same 30% band
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))  # Railway sets PORT
    app.run(host="0.0.0.0", port=port, debug=False)
