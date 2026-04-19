from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load model once at startup
model = joblib.load(os.path.join(os.path.dirname(__file__), "air_quality_model.pkl"))

LABELS = {0: "Good", 1: "Moderate", 2: "Bad"}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "online", "model": "air_quality_model.pkl"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON body"}), 400

        # Feature order must match training: dust, humidity, mq135, temperature
        features = [[
            float(data["dust"]),
            float(data["humidity"]),
            float(data["mq135"]),
            float(data["temperature"]),
        ]]

        code          = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0].tolist()
        confidence    = round(max(probabilities) * 100, 1)

        return jsonify({
            "prediction":  LABELS[code],
            "code":        code,
            "confidence":  confidence,
            "probabilities": {
                "good":     round(probabilities[0] * 100, 1),
                "moderate": round(probabilities[1] * 100, 1),
                "bad":      round(probabilities[2] * 100, 1),
            }
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
