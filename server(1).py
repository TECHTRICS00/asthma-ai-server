from flask import Flask, request, jsonify
import joblib
import os
import time
from datetime import datetime

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────────────────────
model = joblib.load(os.path.join(os.path.dirname(__file__), "air_quality_model.pkl"))
LABELS = {0: "Bad", 1: "Good", 2: "Moderate"}

# ─────────────────────────────────────────────────────────────
#  TWILIO CONFIG
#  Set these as Environment Variables on Render — never hardcode
#  Render Dashboard → Your Service → Environment → Add variables:
#    TWILIO_ACCOUNT_SID  = ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#    TWILIO_AUTH_TOKEN   = your_auth_token
#    TWILIO_FROM_NUMBER  = +1xxxxxxxxxx   (your Twilio number)
#    ALERT_TO_NUMBER     = +91xxxxxxxxxx  (your personal number)
# ─────────────────────────────────────────────────────────────
TWILIO_SID    = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN  = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM   = os.environ.get("TWILIO_FROM_NUMBER")
ALERT_TO      = os.environ.get("ALERT_TO_NUMBER")

# ─────────────────────────────────────────────────────────────
#  SMS COOLDOWN — send alert at most once every 30 seconds
#  Prevents spam if sensor keeps reading BAD continuously
# ─────────────────────────────────────────────────────────────
last_sms_time   = 0        # Unix timestamp of last SMS sent
SMS_COOLDOWN_SEC = 30      # minimum seconds between SMS alerts

def send_sms_alert(dust, humidity, mq135, temperature):
    """Send SMS alert via Twilio when air quality is BAD."""
    global last_sms_time

    # Check cooldown — don't spam SMS every 10 seconds
    now = time.time()
    if now - last_sms_time < SMS_COOLDOWN_SEC:
        seconds_left = int(SMS_COOLDOWN_SEC - (now - last_sms_time))
        print(f"[SMS] Cooldown active — {seconds_left}s remaining, skipping")
        return {"sent": False, "reason": f"cooldown_{seconds_left}s"}

    # Check Twilio credentials are configured
    if not all([TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, ALERT_TO]):
        print("[SMS] Twilio credentials not configured in environment variables")
        return {"sent": False, "reason": "credentials_missing"}

    try:
        from twilio.rest import Client
        client = Client(TWILIO_SID, TWILIO_TOKEN)

        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        message_body = (
            f"⚠️ ASTHMA ALERT — BAD Air Quality Detected!\n\n"
            f"There is a bad air situation — please move to a safe area immediately!\n\n"
            f"📊 Sensor Readings:\n"
            f"  • Dust    : {dust:.2f} µg/m³\n"
            f"  • MQ-135  : {mq135} ppm\n"
            f"  • Humidity: {humidity:.1f}%\n"
            f"  • Temp    : {temperature:.1f}°C\n\n"
            f"🕐 Time: {timestamp}\n"
            f"— Asthma Gas Monitor"
        )

        msg = client.messages.create(
            body=message_body,
            from_=TWILIO_FROM,
            to=ALERT_TO
        )

        last_sms_time = now
        print(f"[SMS] Alert sent! SID={msg.sid}  To={ALERT_TO}")
        return {"sent": True, "sid": msg.sid}

    except Exception as e:
        print(f"[SMS] Failed to send: {str(e)}")
        return {"sent": False, "reason": str(e)}


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "online",
        "model":        "air_quality_model.pkl",
        "sms_enabled":  all([TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, ALERT_TO]),
        "sms_cooldown": SMS_COOLDOWN_SEC,
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON body"}), 400

        # Feature order: dust, humidity, mq135, temperature
        dust        = float(data["dust"])
        humidity    = float(data["humidity"])
        mq135       = float(data["mq135"])
        temperature = float(data["temperature"])

        features = [[dust, humidity, mq135, temperature]]

        code          = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0].tolist()
        confidence    = round(max(probabilities) * 100, 1)

        prediction = LABELS[code]

        # ── SMS ALERT — only when prediction is BAD (code=0) ──
        sms_result = {"sent": False, "reason": "not_bad"}
        if code == 0:  # BAD
            sms_result = send_sms_alert(dust, humidity, mq135, temperature)

        return jsonify({
            "prediction":  prediction,
            "code":        code,
            "confidence":  confidence,
            "probabilities": {
                "bad":      round(probabilities[0] * 100, 1),
                "good":     round(probabilities[1] * 100, 1),
                "moderate": round(probabilities[2] * 100, 1),
            },
            "sms_alert": sms_result,   # Flutter can show this in UI
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
