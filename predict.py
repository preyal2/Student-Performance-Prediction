"""
Student Performance Prediction - Production REST API
Author: Preyal Modi
"""

import pickle
import os
from flask import Flask, request, jsonify

# Load the trained ML pipeline
MODEL_FILE = os.environ.get("MODEL_FILE", "model.bin")

with open(MODEL_FILE, "rb") as f_in:
    model = pickle.load(f_in)

app = Flask("Student_Performance_Prediction")


@app.route("/", methods=["GET"])
def index():
    """Service health and info endpoint."""
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


@app.route("/predict", methods=["POST"])
def predict():
    """Predict math score based on demographic and academic features."""
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
