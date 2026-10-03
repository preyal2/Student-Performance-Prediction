"""
Sample test script for Student Performance Prediction API.
Author: Preyal Modi
"""

import requests
import json

URL = "http://localhost:9696/predict"

# Sample student profile
individual = {
    "gender": "male",
    "race_ethnicity": "group E",
    "parental_level_of_education": "some college",
    "lunch": "standard",
    "test_preparation_course": "completed",
    "total score": 245,
    "average": 81.66666666666667
}

print("=" * 60)
print("Sending prediction request for student profile:")
print(json.dumps(individual, indent=2))
print("=" * 60)

try:
    response = requests.post(URL, json=individual)
    response.raise_for_status()
    result = response.json()
    print("Prediction API Response:")
    print(json.dumps(result, indent=2))
except requests.exceptions.ConnectionError:
    print(f"[!] Could not connect to API at {URL}.")
    print("    Ensure the service is running via:")
    print("    python predict.py OR waitress-serve --listen=0.0.0.0:9696 predict:app")
except Exception as e:
    print(f"[!] Error making request: {e}")
