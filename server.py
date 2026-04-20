import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# ============================================================
# LOAD FILES (ONCE AT STARTUP)
# ============================================================

BASE_DIR = os.path.dirname(__file__)

model   = joblib.load(os.path.join(BASE_DIR, "air_quality_model.pkl"))
scaler  = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

FEATURE_COLS = ['dust', 'humidity', 'mq135', 'temperature']

# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "model": "air_quality_model.pkl"
    })

# ============================================================
# PREDICT API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if data is None:
            return jsonify({"error": "Invalid JSON body"}), 400

        # ---- Validate input ----
        for field in FEATURE_COLS:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # ---- Prepare features ----
        features = [[
            float(data["dust"]),
            float(data["humidity"]),
            float(data["mq135"]),
            float(data["temperature"]),
        ]]

        # ---- Apply scaling (IMPORTANT) ----
        features_scaled = scaler.transform(features)

        # ---- Prediction ----
        code = int(model.predict(features_scaled)[0])
        label = encoder.inverse_transform([code])[0]

        # ---- Probabilities ----
        probabilities = model.predict_proba(features_scaled)[0]
        classes = encoder.classes_

        prob_dict = {
            cls.lower(): round(prob * 100, 1)
            for cls, prob in zip(classes, probabilities)
        }

        confidence = round(max(probabilities) * 100, 1)

        return jsonify({
            "prediction": label,
            "code": code,
            "confidence": confidence,
            "probabilities": prob_dict
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
