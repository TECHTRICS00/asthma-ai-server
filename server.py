from flask import Flask, request, jsonify
import joblib
import os
import time
from datetime import datetime

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
model = joblib.load(os.path.join(os.path.dirname(__file__), "air_quality_model.pkl"))
LABELS = {0: "Bad", 1: "Good", 2: "Moderate"}

# ─────────────────────────────────────────────────────────────
# TWILIO CONFIG
# ─────────────────────────────────────────────────────────────
TWILIO_SID = "ACc1e2b46bc01b0e39424c24be385f2b9d"
TWILIO_TOKEN = "8cf5e3949dcf4ce926f3fff902844297"
MESSAGING_SERVICE_SID = "MG638f81a0c40567ce80aeb26f649160cf"
ALERT_TO = "+918078020512"

# ─────────────────────────────────────────────────────────────
# SMS COOLDOWN — send alert at most once every 30 seconds
# ─────────────────────────────────────────────────────────────
last_sms_time = 0
SMS_COOLDOWN_SEC = 30


def send_sms_alert(dust, humidity, mq135, temperature):
    """Send SMS alert via Twilio Messaging Service when air quality is BAD."""
    global last_sms_time

    now = time.time()
    if now - last_sms_time < SMS_COOLDOWN_SEC:
        seconds_left = int(SMS_COOLDOWN_SEC - (now - last_sms_time))
        print(f"[SMS] Cooldown active — {seconds_left}s remaining, skipping")
        return {"sent": False, "reason": f"cooldown_{seconds_left}s"}

    try:
        from twilio.rest import Client
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        message_body = (
            f"⚠️ ASTHMA ALERT — BAD Air Quality Detected!\n\n"
            f"There is a bad air situation — please move to a safe area immediately!\n\n"
            f"📊 Sensor Readings:\n"
            f" • Dust    : {dust:.2f} µg/m³\n"
            f" • MQ-135  : {mq135} ppm\n"
            f" • Humidity: {humidity:.1f}%\n"
            f" • Temp    : {temperature:.1f}°C\n\n"
            f"🕐 Time: {timestamp}\n"
            f"— Asthma Gas Monitor"
        )

        msg = client.messages.create(
            messaging_service_sid=MESSAGING_SERVICE_SID,
            body=message_body,
            to=ALERT_TO
        )

        last_sms_time = now
        print(f"[SMS] Alert sent! SID={msg.sid} To={ALERT_TO}")
        return {"sent": True, "sid": msg.sid}

    except Exception as e:
        print(f"[SMS] Failed to send: {str(e)}")
        return {"sent": False, "reason": str(e)}


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "model": "air_quality_model.pkl",
        "sms_enabled": True,
        "sms_cooldown": SMS_COOLDOWN_SEC,
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON body"}), 400

        dust        = float(data["dust"])
        humidity    = float(data["humidity"])
        mq135       = float(data["mq135"])
        temperature = float(data["temperature"])

        features = [[dust, humidity, mq135, temperature]]
        code = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0].tolist()
        confidence = round(max(probabilities) * 100, 1)
        prediction = LABELS[code]

        # ── SMS ALERT — only when prediction is BAD (code=0) ──
        sms_result = {"sent": False, "reason": "not_bad"}
        if code == 0:
            sms_result = send_sms_alert(dust, humidity, mq135, temperature)

        return jsonify({
            "prediction": prediction,
            "code": code,
            "confidence": confidence,
            "probabilities": {
                "bad":      round(probabilities[0] * 100, 1),
                "good":     round(probabilities[1] * 100, 1),
                "moderate": round(probabilities[2] * 100, 1),
            },
            "sms_alert": sms_result,
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
