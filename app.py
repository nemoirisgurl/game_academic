from flask import Flask, jsonify, request, render_template
import joblib
import pandas as pd
import os


app = Flask(__name__)
model = joblib.load("grade_prediction_model.pkl")
model_columns = joblib.load("model_features.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()

    try:
        data["gaming_hours"] = float(data["gaming_hours"])
        data["study_hours"] = float(data["study_hours"])
        data["sleep_hours"] = float(data["sleep_hours"])
        data["device_usage"] = float(data["device_usage"])
        data["addiction_score"] = float(data["addiction_score"])
        data["reaction_time_ms"] = float(data["reaction_time_ms"])
        data["attendance"] = float(data["attendance"])
        data["stress_level"] = str(data["stress_level"])
    except (KeyError, TypeError, ValueError):
        return "<h1>Invalid input: All fields must be numeric.</h1>", 400

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return (
        "<h1>Predicted Grade: {:.2f}</h1>"
        "<a href='/'>Make Another Prediction</a>".format(prediction)
    )


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5050"))
    app.run(host=host, port=port)
