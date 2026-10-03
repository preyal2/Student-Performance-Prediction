# 🛠️ Development Workspace

Welcome to the development workspace for the **Student Performance Prediction System**. This directory organizes the codebase into dedicated **Frontend** and **Backend** subsystems for modular development and deployment.

---

## 📁 Directory Structure

```
Development/
│
├── frontend/
│   └── index.html             # Interactive Web Application (Single-File)
│                              # - In-browser mathematical regression engine
│                              # - Responsive glassmorphic UI
│                              # - Real-time score analytics & presets
│                              # - Dual engine toggle (In-Browser / Flask API)
│
└── backend/
    ├── predict.py             # Production Flask REST API & WSGI serving
    ├── predict_test.py        # Automated endpoint test client
    ├── train.py               # ML training, cross-validation & serialization
    ├── model.bin              # Trained Scikit-Learn / XGBoost pipeline artifact
    ├── Dockerfile             # Multi-stage container specification
    ├── requirements.txt       # Standard pip package dependencies
    ├── pyproject.toml         # Poetry packaging configuration
    └── poetry.lock            # Deterministic dependency lockfile
```

---

## 🚀 Quickstart Guides

### 1. Frontend Development (`Development/frontend/`)
The frontend is completely self-contained within `index.html`. You can open it directly in any browser:
```bash
# Simply double-click index.html or open via local server
cd Development/frontend
python -m http.server 8000
```
Then navigate to `http://localhost:8000`.

### 2. Backend Development (`Development/backend/`)
Run the Flask API or multi-threaded Waitress WSGI server:
```bash
cd Development/backend

# Option A: With Poetry
poetry install
poetry run python -m waitress --listen=0.0.0.0:9696 predict:app

# Option B: With Standard Pip
pip install -r requirements.txt
python predict.py
```

### 3. Verify Endpoints
In a separate terminal, test the running backend:
```bash
cd Development/backend
python predict_test.py
```

---

## 🔗 Key Links
- **Live Deployment:** [https://preyal2.github.io/Student-Performance-Prediction/](https://preyal2.github.io/Student-Performance-Prediction/)
- **Model Prediction & EDA Notebook:** [notebook.ipynb](https://github.com/preyal2/Student-Performance-Prediction/blob/main/notebook.ipynb)
- **Author:** Preyal Modi ([modipreyal@gmail.com](mailto:modipreyal@gmail.com))
