# 🛠️ Development Workspace & Engineering Guide
### *Comprehensive Subsystem Documentation for Student Performance Prediction*

Welcome to the development workspace. This directory separates and organizes the project into modular **Frontend** and **Backend** subsystems for clean development, testing, and deployment.

---

## 📁 Subsystem Structure

```
Development/
│
├── frontend/
│   └── index.html             # Interactive Web Application (Single-File)
│                              # - In-browser mathematical regression engine
│                              # - Responsive glassmorphic UI (CSS3 + Flexbox/Grid)
│                              # - Real-time score analytics & 4 persona presets
│                              # - Dual engine toggle (In-Browser / Flask REST API)
│
└── backend/
    ├── predict.py             # Production Flask REST API & Waitress WSGI serving
    ├── predict_test.py        # Automated endpoint test client
    ├── train.py               # ML training, cross-validation & Bayesian tuning
    ├── model.bin              # Serialized Scikit-Learn / XGBoost pipeline artifact
    ├── Dockerfile             # Multi-stage production container specification
    ├── requirements.txt       # Standard pip package dependencies
    ├── pyproject.toml         # Poetry packaging configuration
    └── poetry.lock            # Deterministic dependency lockfile
```

---

## 🚀 Quickstart Guides

### 1. 🎨 Frontend Subsystem (`Development/frontend/`)
The frontend is completely self-contained within a single [`index.html`](frontend/index.html).

#### Local Preview:
```bash
# Option A: Open directly in your browser
start Development/frontend/index.html   # Windows
open Development/frontend/index.html    # macOS

# Option B: Run via lightweight HTTP server
cd Development/frontend
python -m http.server 8000
```
Navigate to: `http://localhost:8000`

#### In-Browser ML Engine:
The frontend contains an embedded regression engine aligned with the trained pipeline:
$$\hat{y} = (0.9771 	imes 	ext{average} - 0.1306) + \Delta_{	ext{gender}} + \Delta_{	ext{ethnicity}} + \Delta_{	ext{lunch}} + \Delta_{	ext{parent}} + \Delta_{	ext{prep}}$$
This enables instant client-side predictions on static hosts like GitHub Pages without needing a live backend server.

---

### 2. ⚙️ Backend Subsystem (`Development/backend/`)

#### Environment Setup:
```bash
cd Development/backend

# Option A: Poetry (Recommended)
poetry install
poetry run python -m waitress --listen=0.0.0.0:9696 predict:app

# Option B: Standard Virtual Environment & Pip
python -m venv .venv
.venv\Scripts\activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python predict.py
```

#### API Endpoints:
- `GET /` — Serves the interactive web UI or API metadata.
- `GET /health` — Orchestrator health check probe (`{"status": "UP"}`).
- `POST /predict` — Inference endpoint supporting single objects and batch arrays.

#### Automated Client Verification:
```bash
python predict_test.py
```

---

### 3. 🐳 Docker Containerization
```bash
cd Development/backend

# Build image
docker build -t student-performance-backend .

# Run container
docker run -d -p 9696:9696 --name student-predictor student-performance-backend

# Check health
curl http://localhost:9696/health
```

---

### 4. 🧠 Model Retraining & Bayesian Search (`train.py`)
```bash
cd Development/backend
python train.py
```
- Performs 5-fold cross-validation across 8+ regression algorithms.
- Runs Bayesian hyperparameter search via `skopt.BayesSearchCV`.
- Re-serializes the optimal model pipeline into `model.bin`.

---

## 🔗 Key Links
- **🌐 Live Web Deployment:** [https://preyal2.github.io/Student-Performance-Prediction/](https://preyal2.github.io/Student-Performance-Prediction/)
- **📊 Model Prediction & EDA Notebook:** [notebook.ipynb](https://github.com/preyal2/Student-Performance-Prediction/blob/main/notebook.ipynb)
- **👨‍💻 Author:** Preyal Modi ([modipreyal@gmail.com](mailto:modipreyal@gmail.com))
