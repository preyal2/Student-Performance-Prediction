"""
Student Performance Prediction - Production REST API & Web App Serving
Author: Preyal Modi <modipreyal@gmail.com>
"""

import pickle
import os
from flask import Flask, request, jsonify

# Load the trained ML pipeline relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "model.bin")
MODEL_FILE = os.environ.get("MODEL_FILE", DEFAULT_MODEL)

# Fallback to parent directory if model not in current directory
if not os.path.exists(MODEL_FILE):
    parent_model = os.path.join(SCRIPT_DIR, "..", "model.bin")
    if os.path.exists(parent_model):
        MODEL_FILE = parent_model

with open(MODEL_FILE, "rb") as f_in:
    model = pickle.load(f_in)

app = Flask("Student_Performance_Prediction")


@app.after_request
def add_cors_headers(response):
    """Enable Cross-Origin Resource Sharing (CORS) for web app clients."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/", methods=["GET"])
def index():
    """Serve the interactive web application frontend or API metadata."""
    possible_paths = [
        os.path.join(SCRIPT_DIR, "index.html"),
        os.path.join(SCRIPT_DIR, "..", "frontend", "index.html"),
        os.path.join(SCRIPT_DIR, "..", "index.html")
    ]
    for p in possible_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}

    return jsonify({
        "status": "healthy",
        "service": "Student Performance Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health (GET)",
            "predict": "/predict (POST)"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for container orchestrators."""
    return jsonify({"status": "UP", "model_loaded": model is not None}), 200


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """Predict math score based on demographic and academic features."""
    if request.method == "OPTIONS":
        return "", 204

    try:
        individual = request.get_json(force=True)
        if individual is None:
            return jsonify({"error": "No JSON payload provided"}), 400

        # Handle single record dict vs batch list of dicts
        if isinstance(individual, dict):
            records = [individual]
            is_batch = False
        elif isinstance(individual, list):
            records = individual
            is_batch = True
        else:
            return jsonify({"error": "Payload must be a JSON object or array of objects"}), 400

        predictions = model.predict(records)

        if not is_batch:
            raw_score = float(predictions[0])
            rounded_score = int(round(raw_score))
            output = {
                "status": "success",
                "predicted_math_score": round(raw_score, 2),
                "The student has scored in Math": rounded_score
            }
            return jsonify(output)
        else:
            results = [
                {
                    "predicted_math_score": round(float(p), 2),
                    "The student has scored in Math": int(round(float(p)))
                }
                for p in predictions
            ]
            return jsonify({"status": "success", "predictions": results})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=9696)
