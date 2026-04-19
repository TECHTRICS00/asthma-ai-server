import joblib
import numpy as np
from firebase_functions import https_fn
from firebase_admin import initialize_app

initialize_app()

# Load model once at cold start (not on every request)
model = joblib.load("air_quality_model.pkl")

LABELS = {0: "Good", 1: "Moderate", 2: "Bad"}


@https_fn.on_request()
def predict(req: https_fn.Request) -> https_fn.Response:
    # Handle CORS preflight
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }

    if req.method == "OPTIONS":
        return https_fn.Response("", status=204, headers=headers)

    if req.method != "POST":
        return https_fn.Response(
            '{"error": "POST only"}',
            status=405,
            headers={**headers, "Content-Type": "application/json"},
        )

    try:
        data = req.get_json(silent=True)
        if data is None:
            return https_fn.Response(
                '{"error": "Invalid JSON body"}',
                status=400,
                headers={**headers, "Content-Type": "application/json"},
            )

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

        result = {
            "prediction":  LABELS[code],
            "code":        code,
            "confidence":  confidence,
            "probabilities": {
                "good":     round(probabilities[0] * 100, 1),
                "moderate": round(probabilities[1] * 100, 1),
                "bad":      round(probabilities[2] * 100, 1),
            }
        }

        import json
        return https_fn.Response(
            json.dumps(result),
            status=200,
            headers={**headers, "Content-Type": "application/json"},
        )

    except KeyError as e:
        import json
        return https_fn.Response(
            json.dumps({"error": f"Missing field: {str(e)}"}),
            status=400,
            headers={**headers, "Content-Type": "application/json"},
        )
    except Exception as e:
        import json
        return https_fn.Response(
            json.dumps({"error": str(e)}),
            status=500,
            headers={**headers, "Content-Type": "application/json"},
        )
