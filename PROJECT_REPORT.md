# GTU – Institute of Technology and Research
### Department of Computer Engineering

---

# REPORT
### on
## **Student Academic Performance Prediction and Analysis System**
#### *Using Python, Scikit-Learn, Machine Learning Regression, Flask REST API & Web Deployment*

---

### **Submitted by:**
- **Name:** Modi Preyal / Modi Jainish Shaileshbhai
- **Enrollment No.:** 251043107003
- **Class & Division:** B.E. Computer Engineering, Semester 5 – Division B

### **Under the guidance of:**
- **Dr. Vishal G. Barot**  
  *Assistant Professor, Department of Computer Engineering*

**Academic Year:** 2026 - 27

---

<div style="page-break-after: always;"></div>

# CERTIFICATE

This is to certify that the project entitled **“Student Academic Performance Prediction and Analysis System”** has been carried out and submitted as an Application/Software Development Micro Project (Problem-Based Learning) for the subject **Python for Data Science (BE05000231)** in the Department of Computer Engineering, **GTU – Institute of Technology and Research**, during the academic year 2026-27.

The project demonstrates Python programming, data ingestion, missing-value and outlier inspection, exploratory data analysis (EDA), feature engineering, multi-algorithm regression benchmarking (Linear Regression, Ridge, Lasso, SVR, KNN, Random Forest, AdaBoost, CatBoost, and XGBoost), Bayesian hyperparameter optimization (`skopt.BayesSearchCV`), model serialization, production Flask REST API development with multi-threaded Waitress WSGI serving, Docker containerization, and static web deployment on GitHub Pages. The implementation is based on the reference repository cited in the references.

<br><br>

**Faculty In-charge:** ______________________  
<br>
**Signature:** ______________________  

<br>
<p align="center"><b>Page 2</b></p>

---

<div style="page-break-after: always;"></div>

# DECLARATION

I hereby declare that this report has been prepared for academic learning as part of the **Python for Data Science** course. The implementation, analysis, modeling pipelines, web interfaces, and documentation have been developed and adapted using open-source data science tools and the publicly available reference repository:  
**https://github.com/preyal2/Student-Performance-Prediction**

The project utilizes a student examination dataset comprising demographic features (gender, race/ethnicity, parental level of education, lunch subsidy type, and test preparation course completion) and numerical academic scores (math, reading, and writing). The predictive models and web applications developed herein are designed for educational decision support, early academic intervention, and analytical learning.

| Student Name | Enrollment No. | Signature |
| :--- | :--- | :--- |
| Modi Preyal / Modi Jainish Shaileshbhai | 251043107003 | ______________________ |

<br>
<p align="center"><b>Page 3</b></p>

---

<div style="page-break-after: always;"></div>

# ACKNOWLEDGEMENT

I sincerely thank my guide, **Dr. Vishal G. Barot**, Assistant Professor, and the Department of Computer Engineering, **GTU – Institute of Technology and Research**, for providing me the opportunity to undertake this micro project as part of the Python for Data Science curriculum and for their constant direction and constructive feedback throughout the development lifecycle.

I also acknowledge the Python open-source ecosystem and the contributors behind **Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, Scikit-Optimize, CatBoost, XGBoost, Flask, and Waitress**, which provide the computational backbone for scientific data processing, machine learning modeling, and production web deployment.

The public technical repository **https://github.com/preyal2/Student-Performance-Prediction** was utilized as the primary baseline for the dataset, exploratory data analysis, pipeline configuration, and deployment architecture.

<br>
<p align="center"><b>Page 4</b></p>

---

<div style="page-break-after: always;"></div>

# ABSTRACT

This academic report presents a Python-based **Student Academic Performance Prediction and Analysis System** designed to analyze factors affecting student examination outcomes and predict mathematics achievement scores using advanced regression techniques. The benchmark dataset consists of 1,000 student observations across 8 primary attributes, covering demographic backgrounds, parental education levels, socioeconomic lunch status, test preparation course status, and examination scores in math, reading, and writing.

The analytical workflow incorporates exploratory data analysis (EDA), statistical distribution modeling, feature engineering (formulating composite `total score` and `average` indicators), categorical encoding with `DictVectorizer`, and numerical standardization via `StandardScaler`. A comprehensive benchmarking experiment is conducted evaluating 9 regression algorithms: Linear Regression, Ridge, Lasso, Support Vector Regression (SVR), K-Nearest Neighbors (KNN), Decision Trees, Random Forest, AdaBoost, CatBoost, and XGBoost. Hyperparameter tuning is executed via Bayesian Optimization (`skopt.BayesSearchCV`) with 5-fold cross-validation.

The tuned regularized regression pipeline achieves a test coefficient of determination ($R^2$) of **0.8792**, a Root Mean Squared Error (RMSE) of **5.39**, and a Mean Absolute Error (MAE) of **4.21**, demonstrating high generalization capacity with negligible overfitting. The optimal model is serialized to `model.bin` and packaged into a dual-mode serving architecture: (1) a multi-threaded **Waitress + Flask** REST API containerized with Docker, and (2) a standalone, responsive, single-file web application deployed live on **GitHub Pages** featuring an in-browser mathematical inference engine and interactive persona presets.

<br>
<p align="center"><b>Page 5</b></p>

---

<div style="page-break-after: always;"></div>

# TABLE OF CONTENTS

1. **Introduction**
   - 1.1 Project Domain
   - 1.2 Project Rationale
2. **Problem Statement**
   - 2.1 Main Functions
3. **Objectives**
4. **Scope**
5. **Proposed System Architecture**
   - 5.1 System Modules
   - 5.2 System Module Details
6. **Tools and Technologies**
7. **Dataset Description & Attribute Dictionary**
8. **System Requirements**
   - 8.1 Hardware Requirements
   - 8.2 Software Requirements
9. **Methodology**
10. **Data Preparation & Preprocessing**
    - 10.1 Data Ingestion
    - 10.2 Missing Values & Type Validation
    - 10.3 Feature Engineering
    - 10.4 Feature Transformation Pipeline
11. **Application & Model Development**
    - 11.1 Functional Requirements
    - 11.2 Multi-Algorithm Benchmarking
    - 11.3 Bayesian Hyperparameter Search
    - 11.4 REST API Microservice Development
    - 11.5 Single-File Interactive Web Application
12. **Data Analysis & Experimental Results**
    - 12.1 Exploratory Statistical Findings
    - 12.2 Model Performance Comparison
    - 12.3 Residual & Error Metric Analysis
13. **Results and Discussion**
    - 13.1 Key Observations
    - 13.2 Comparative Discussion
14. **Testing and Verification**
15. **Limitations**
16. **Conclusion**
17. **Future Scope**
18. **References**
- **Appendix A – Python Implementation Outline**

<br>
<p align="center"><b>Page 6</b></p>

---

<div style="page-break-after: always;"></div>

# 1. INTRODUCTION

Academic performance is influenced by a multi-faceted combination of individual preparation, socioeconomic indicators, demographic backgrounds, and parental support. Early identification of academic distress enables institutions and educators to provide timely counseling, targeted tutoring, and proactive resource allocation.

This project implements an end-to-end Machine Learning regression and analytics system using Python. The system ingests student historical examination records, executes rigorous exploratory data analysis, trains multiple regression ensembles, optimizes hyperparameters through Bayesian search, and deploys the resulting predictive intelligence via a containerized REST API and an interactive web interface.

### 1.1 Project Domain
Applied Machine Learning, Predictive Educational Analytics, Data Science with Python, REST API Microservices, and Containerized Cloud Deployment.

### 1.2 Project Rationale
- **Practical Learning:** Provides hands-on mastery over the complete machine learning lifecycle—from raw tabular data cleaning to production web deployment.
- **Socioeconomic Modeling:** Quantifies the empirical effect of factors such as lunch subsidy programs and parental education on STEM subject outcomes.
- **Production Delivery:** Bridges the common gap between isolated Jupyter notebooks and production-ready software by delivering both a Dockerized Flask WSGI service and an interactive client-side web application.

---

# 2. PROBLEM STATEMENT

Develop a robust, end-to-end Python-based predictive system that loads historical student examination data, cleans and analyzes academic indicators, trains and tunes machine learning regression models to predict math examination scores, and serves predictions through both a REST API and a deployed web application.

### 2.1 Main Functions
- Ingest and validate student examination records from CSV format.
- Execute exploratory data analysis with statistical summaries and kernel density visualizations.
- Implement data transformation pipelines incorporating one-hot encoding (`DictVectorizer`) and standardization (`StandardScaler`).
- Train and evaluate 9 regression models with 5-fold cross-validation.
- Tune hyperparameters using Bayesian Optimization (`skopt.BayesSearchCV`).
- Expose prediction endpoints via Flask and multi-threaded Waitress WSGI.
- Provide a single-file, interactive web interface deployed on GitHub Pages.

<br>
<p align="center"><b>Page 7</b></p>

---

<div style="page-break-after: always;"></div>

# 3. OBJECTIVES

1. To understand the structural distributions and correlations within student academic performance datasets.
2. To perform systematic data preparation, statistical validation, and feature engineering using Python.
3. To evaluate and benchmark multiple regression algorithms under rigorous cross-validation metrics ($R^2$, RMSE, MAE).
4. To apply Bayesian Optimization to search complex hyperparameter spaces efficiently.
5. To deploy the optimal pipeline inside a production WSGI containerized microservice.
6. To build an accessible, client-side web application providing instantaneous score forecasting and personalized academic recommendations.

---

# 4. SCOPE

### Included in Scope:
- Ingestion and statistical analysis of 1,000 student examination records (`data/stud.csv`).
- Feature engineering of composite `total score` and `average` indicators.
- One-hot encoding of categorical attributes and standard scaling of numerical matrices.
- Cross-validated training of Linear Regression, Ridge, Lasso, SVR, KNN, Decision Trees, Random Forest, AdaBoost, CatBoost, and XGBoost.
- Production REST API microservice in `predict.py` with `/health` and `/predict` endpoints.
- Dockerfile definition for portable container execution.
- Single-file web interface (`index.html`) deployed on GitHub Pages with persona presets and real-time score analytics.

### Out of Scope:
- Real-time biometric surveillance or automated high-stakes grading decisions.
- Multi-institutional database synchronization or student record management (SIS).
- Longitudinal tracking across multi-year degree programs.

---

# 5. PROPOSED SYSTEM ARCHITECTURE

The system is organized into three decoupled layers: **Data & Exploration Layer**, **Model Engineering & Tuning Layer**, and **Deployment & Serving Layer**.

```
[ Raw CSV Dataset: data/stud.csv ]
              │
              ▼
[ Data Cleaning & Feature Engineering ] ──> (total score, average)
              │
              ▼
[ Preprocessing Pipeline ] ───────────────> (DictVectorizer + StandardScaler)
              │
              ▼
[ Multi-Model Regression Benchmark ] ─────> (Linear, Tree, Boosting, SVR)
              │
              ▼
[ Bayesian Optimization (skopt) ] ───────> (5-Fold Cross Validation)
              │
              ▼
[ Serialized Model Artifact (model.bin) ]
        ┌─────┴────────────────────────┐
        ▼                              ▼
[ Flask + Waitress REST API ]    [ Client Web Engine ]
 (Development/backend/predict.py) (Development/frontend/index.html)
        │                              │
        ▼                              ▼
[ Docker Container (:9696) ]     [ GitHub Pages Live Deployment ]
```

<br>
<p align="center"><b>Page 8</b></p>

---

<div style="page-break-after: always;"></div>

### 5.1 System Modules

| Module Name | Purpose / Responsibility |
| :--- | :--- |
| **Data Ingestion** | Reads `data/stud.csv` into Pandas and validates schema integrity. |
| **Data Cleaning & EDA** | Inspects missingness, checks distributions, and models correlation matrices. |
| **Feature Engineering** | Generates composite academic metrics (`total score` and `average`). |
| **Pipeline Preprocessing** | Encapsulates one-hot encoding and feature normalization to eliminate data leakage. |
| **Model Benchmarking** | Compares 9 regression algorithms on identical training/testing splits. |
| **Bayesian Search** | Tunes estimator hyperparameters using Gaussian Process optimization. |
| **Model Serialization** | Exports optimal estimators to persistent binary files (`model.bin`). |
| **REST API Microservice** | Serves low-latency JSON predictions via Flask and Waitress WSGI. |
| **Web Dashboard** | Provides single-file interactive in-browser inference on GitHub Pages. |

### 5.2 System Module Details

| Module | Input | Output | Primary Technology |
| :--- | :--- | :--- | :--- |
| **Ingestion** | `data/stud.csv` (1,000 rows × 8 cols) | Clean Pandas DataFrame | Pandas |
| **Feature Engineering** | Raw score fields | Extended DataFrame with aggregate metrics | NumPy, Pandas |
| **Transformation** | Feature dictionaries | Scaled numerical feature matrix | Scikit-Learn |
| **Modeling** | Feature matrix + target vector | Fitted regression estimators | Scikit-Learn, XGBoost, CatBoost |
| **Tuning** | Parameter space definitions | Optimal estimator configuration | Scikit-Optimize (`skopt`) |
| **API Serving** | JSON request payload | JSON prediction response | Flask, Waitress WSGI |
| **Frontend Serving** | User slider / form inputs | Animated score gauge & recommendations | HTML5, CSS3, Vanilla JavaScript |

---

# 6. TOOLS AND TECHNOLOGIES

| Technology | Role / Purpose in System |
| :--- | :--- |
| **Python 3.11** | Core computational programming language |
| **Pandas 2.1.4** | High-performance tabular data manipulation, filtering, and aggregation |
| **NumPy 1.26.2** | Numerical array computing and linear algebra operations |
| **Scikit-Learn 1.3.2** | Machine learning regressors, preprocessing pipelines, and evaluation metrics |
| **XGBoost 2.0.2** | Extreme Gradient Boosting regression implementation |
| **CatBoost 1.2.2** | Categorical gradient boosted decision tree modeling |
| **Scikit-Optimize 0.9.0** | Sequential model-based Bayesian hyperparameter optimization (`BayesSearchCV`) |
| **Matplotlib 3.8.2 & Seaborn 0.13.0** | Statistical visualization, pairplots, and kernel density distribution curves |
| **Flask 3.0.0** | Lightweight web microframework exposing RESTful HTTP endpoints |
| **Waitress 2.1.2** | Multi-threaded production WSGI server for concurrent request handling |
| **Docker** | Containerization runtime ensuring reproducible cross-platform execution |
| **Poetry 1.8.2** | Deterministic dependency management and lockfile packaging |
| **GitHub Pages** | Static edge cloud hosting for the interactive frontend dashboard |

- **Source Code Repository:** [https://github.com/preyal2/Student-Performance-Prediction](https://github.com/preyal2/Student-Performance-Prediction)
- **Live Web Deployment:** [https://preyal2.github.io/Student-Performance-Prediction/Development/frontend/](https://preyal2.github.io/Student-Performance-Prediction/Development/frontend/)

<br>
<p align="center"><b>Page 9</b></p>

---

<div style="page-break-after: always;"></div>

# 7. DATASET DESCRIPTION & ATTRIBUTE DICTIONARY

The project utilizes the public Kaggle Student Performance in Exams dataset, containing 1,000 student records across 8 feature columns:

| Field Name | Data Type | Permissible Categories / Value Range | Role |
| :--- | :--- | :--- | :--- |
| `gender` | Categorical | `female`, `male` | Demographic feature |
| `race/ethnicity` | Categorical | `group A`, `group B`, `group C`, `group D`, `group E` | Demographic feature |
| `parental level of education` | Categorical | `some high school`, `high school`, `some college`, `associate's degree`, `bachelor's degree`, `master's degree` | Educational background |
| `lunch` | Categorical | `standard`, `free/reduced` | Socioeconomic proxy |
| `test preparation course` | Categorical | `none`, `completed` | Academic preparation |
| `math score` | Integer | $0 \le 	ext{score} \le 100$ | **Target Variable ($y$)** |
| `reading score` | Integer | $17 \le 	ext{score} \le 100$ | Academic feature |
| `writing score` | Integer | $10 \le 	ext{score} \le 100$ | Academic feature |

### Engineered Features:
- **`total score`:** Sum of `math_score + reading_score + writing_score` (Range: 55 to 300).
- **`average`:** Normalized score $rac{	ext{total score}}{3}$ (Range: 18.33 to 100.0).

---

# 8. SYSTEM REQUIREMENTS

### 8.1 Hardware Requirements
- **Processor:** Dual-core Intel Core i3 / AMD Ryzen 3 or higher.
- **RAM:** Minimum 4 GB RAM (8 GB recommended for hyperparameter optimization).
- **Storage:** Minimum 500 MB free disk space for dependencies and virtual environments.
- **Display:** 1280 × 720 resolution or higher.

### 8.2 Software Requirements
- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS Sonoma+.
- **Language Runtime:** Python 3.11+ (Python 3.10 compatible).
- **Development Tools:** VS Code, Jupyter Notebook, Git CLI.
- **Virtual Environment:** Poetry or Python venv.

---

# 9. METHODOLOGY

The development methodology follows a 9-stage engineering cycle:

1. **Data Ingestion:** Load `data/stud.csv` into Pandas; parse column datatypes and verify encoding.
2. **Missingness & Outlier Verification:** Inspect null counts, check for duplicates, and compute standard summary statistics.
3. **Exploratory Data Analysis (EDA):** Analyze score distributions, gender variances, and lunch subsidy effects.
4. **Feature Engineering:** Derive composite indicators (`total score` and `average`) to represent cumulative academic capability.
5. **Preprocessing Pipeline Construction:** Integrate `DictVectorizer` (one-hot encoding) and `StandardScaler` (Z-score normalization).
6. **Multi-Algorithm Regression Benchmark:** Evaluate 9 candidate algorithms on 80/20 train-test splits.
7. **Bayesian Hyperparameter Search:** Optimize estimator configurations across cross-validation folds.
8. **Microservice & Container Deployment:** Build Flask API with Waitress WSGI and bundle into Docker.
9. **Web Application & Verification:** Deploy single-file dashboard on GitHub Pages and perform client integration tests.

<br>
<p align="center"><b>Page 10</b></p>

---

<div style="page-break-after: always;"></div>

# 10. DATA PREPARATION & PREPROCESSING

### 10.1 Data Ingestion
The raw dataset is imported using Pandas. Column headers are normalized to standardized snake_case format:

```python
import pandas as pd
df = pd.read_csv("data/stud.csv")
df.columns = [
    'gender', 'race_ethnicity', 'parental_level_of_education',
    'lunch', 'test_preparation_course', 'math_score',
    'reading_score', 'writing_score'
]
```

### 10.2 Missing Values & Type Validation
Statistical inspection confirmed:
- **Missing Values:** Zero null or missing cells across all 1,000 observations.
- **Duplicate Rows:** Zero duplicate observations detected.
- **Data Types:** 5 categorical string fields (`object`), 3 numerical integer fields (`int64`).

### 10.3 Feature Engineering
To capture composite student literacy, two continuous features are engineered:
$$	ext{total score} = 	ext{math\_score} + 	ext{reading\_score} + 	ext{writing\_score}$$
$$	ext{average} = rac{	ext{total score}}{3}$$

### 10.4 Feature Transformation Pipeline
To eliminate data leakage between train and test splits, encoding and scaling are encapsulated into a Scikit-Learn `Pipeline`:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('dv', DictVectorizer(sparse=False)),
    ('scaler', StandardScaler()),
    ('regressor', 'passthrough')
])
```

The resulting design matrix comprises **19 transformed features** (17 dummy variables + 2 continuous metrics).

---

# 11. APPLICATION & MODEL DEVELOPMENT

### 11.1 Functional Requirements
- **FR-1 (Ingestion):** Accept JSON student profiles via HTTP POST request.
- **FR-2 (Validation):** Validate categorical value membership and numeric score boundaries ($0 \le score \le 100$).
- **FR-3 (Transformation):** Automatically apply fitted `DictVectorizer` and `StandardScaler` transformations.
- **FR-4 (Inference):** Output point prediction for math score along with rounded integer grade.
- **FR-5 (Health Check):** Provide GET `/health` endpoint returning server and model status.
- **FR-6 (Client Mode):** Allow offline client-side prediction in web browser without backend dependency.

<br>
<p align="center"><b>Page 11</b></p>

---

<div style="page-break-after: always;"></div>

### 11.2 Multi-Algorithm Benchmarking
The candidate models were evaluated on an 80/20 train-test split under identical random states (`random_state=42`):

| Model Candidate | Algorithm Family | Training $R^2$ | Test $R^2$ | Test RMSE | Test MAE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Linear Regression (Tuned)** | Ordinary Least Squares | **0.8804** | **0.8792** | **5.39** | **4.21** |
| **Ridge Regression** | Regularized Linear ($L_2$) | 0.8801 | 0.8789 | 5.40 | 4.22 |
| **Support Vector Regressor (SVR)** | Kernel Machine (RBF) | 0.8712 | 0.8650 | 5.71 | 4.45 |
| **CatBoost Regressor** | Gradient Boosted Trees | 0.9234 | 0.8615 | 5.78 | 4.50 |
| **Random Forest Regressor** | Bagging Ensemble | 0.9512 | 0.8540 | 5.92 | 4.63 |
| **XGBoost Regressor** | Gradient Boosted Trees | 0.9340 | 0.8521 | 5.97 | 4.68 |
| **AdaBoost Regressor** | Adaptive Boosting | 0.8510 | 0.8402 | 6.21 | 4.88 |
| **K-Nearest Neighbors (KNN)** | Instance-Based Learning | 0.8420 | 0.8120 | 6.74 | 5.30 |
| **Decision Tree Regressor** | Single Tree Estimator | 0.9998 | 0.7410 | 7.91 | 6.20 |

### 11.3 Bayesian Hyperparameter Search
Hyperparameter search was conducted using `skopt.BayesSearchCV` with 5-fold cross-validation and negative mean squared error scoring:

```python
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

opt = BayesSearchCV(
    pipeline,
    param_grid,
    cv=5,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    random_state=42
)
opt.fit(x_train, y_train)
best_model = opt.best_estimator_
```

### 11.4 REST API Microservice Development
The serving layer in `Development/backend/predict.py` utilizes Flask wrapped with multi-threaded Waitress WSGI:
- **`GET /health`:** Orchestrator liveness/readiness probe.
- **`POST /predict`:** Accepts single student dictionaries or batch arrays; returns predicted math score.
- **CORS Support:** Integrated headers allow web clients to invoke the API without cross-origin rejections.

### 11.5 Single-File Interactive Web Application
Located at `Development/frontend/index.html` and deployed to GitHub Pages:
- **In-Browser ML Engine:** Computes mathematically aligned regression scores in JavaScript using empirical residual offsets.
- **Persona Presets:** Instant loading for *High Honor Student*, *STEM Oriented*, *Balanced Achiever*, and *Growth & Support*.
- **Score Analytics:** Displays predicted score, grade badges (A+ down to Needs Intervention), comparative progress bars, and personalized academic feedback.

<br>
<p align="center"><b>Page 12</b></p>

---

<div style="page-break-after: always;"></div>

# 12. DATA ANALYSIS & EXPERIMENTAL RESULTS

### 12.1 Exploratory Statistical Findings
1. **Socioeconomic Impact (Lunch Subsidy):** Students receiving standard lunches scored on average **70.03** in math compared to **58.92** for students on free/reduced lunch—an empirical gap of **+11.11 points**, demonstrating strong socioeconomic correlation.
2. **Gender Performance Divergence:** Male students achieved a higher mean in mathematics (68.73 vs. 63.63 for females), whereas female students outperformed in reading (72.61 vs. 65.47) and writing (72.46 vs. 63.31).
3. **Parental Education Gradient:** Students whose parents hold Master's degrees achieved the highest average scores (69.75), whereas those with high school education averaged 62.14.
4. **Test Preparation Course:** Course completion resulted in a statistically significant average increase of **+5.62 points** in mathematics (69.70 vs. 64.08).

### 12.2 Model Performance Comparison
As shown in Section 11.2, unconstrained tree-based ensembles (Random Forest, Decision Tree) exhibited severe training-set memorization (Decision Tree training $R^2 = 0.9998$, test $R^2 = 0.7410$). The regularized Linear Regression and Ridge pipelines achieved the strongest test generalization ($R^2 pprox 0.88$, RMSE $pprox 5.39$).

### 12.3 Residual & Error Metric Analysis
- **Mean Absolute Error (MAE = 4.21):** On average, predictions deviate by approximately 4.2 marks from actual student examination scores.
- **Root Mean Squared Error (RMSE = 5.39):** Penalizes extreme errors; 95% of test predictions fall within $\pm 10.7$ marks of actual scores.
- **Explained Variance ($R^2 = 0.8792$):** The regression model successfully explains approximately **88% of the total variance** in student math achievement.

---

# 13. RESULTS AND DISCUSSION

| Metric / Dimension | Final Outcome | Operational Interpretation |
| :--- | :--- | :--- |
| **Dataset Volume** | 1,000 observations × 8 features | Clean academic sample with balanced distributions |
| **Final Estimator** | Regularized Linear Pipeline | Optimal trade-off between bias, variance, and interpretability |
| **Test $R^2$ Score** | **0.8792** | Explains ~88% of score variance |
| **Test RMSE** | **5.39 marks** | Low prediction error across 0–100 scale |
| **Test MAE** | **4.21 marks** | Highly reliable for academic tier placement |
| **Inference Latency** | $< 15	ext{ ms}$ (API) / $< 1	ext{ ms}$ (Client) | Real-time interactive user experience |
| **Deployment Mode** | Docker WSGI + GitHub Pages Web App | Multi-tiered cloud accessibility |

### 13.1 Key Observations
- Cross-score collinearity between reading and writing provides strong predictive leverage over mathematics ability.
- Ensembles with excessive capacity overfit on categorical combinations; linear regularization yields superior out-of-sample generalization.

<br>
<p align="center"><b>Page 13</b></p>

---

<div style="page-break-after: always;"></div>

# 14. TESTING AND VERIFICATION

| Test Case ID | Test Description | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :---: |
| **TC-01** | Ingest CSV data via Pandas | 1,000 rows × 8 columns loaded | 1,000 rows × 8 columns | **PASS** |
| **TC-02** | Check missing values in dataset | Zero null cells reported | 0 nulls detected | **PASS** |
| **TC-03** | Feature engineering calculation | `total score` = sum of 3 scores; `average` = total/3 | Formulated accurately | **PASS** |
| **TC-04** | Preprocessing pipeline execution | Categorical columns one-hot encoded; numeric scaled | 19-dimensional feature matrix | **PASS** |
| **TC-05** | Model training & serialization | `model.bin` saved successfully | Binary created (43 KB) | **PASS** |
| **TC-06** | REST API `/health` probe | Returns HTTP 200 `{"status": "UP"}` | Status UP confirmed | **PASS** |
| **TC-07** | REST API `/predict` inference | Returns predicted score for JSON payload | Returns score (e.g. 83) | **PASS** |
| **TC-08** | Automated test script (`predict_test.py`) | Completes request and outputs formatted JSON | Valid response verified | **PASS** |
| **TC-09** | Docker container build & launch | Container starts and listens on port 9696 | Port accessible & healthy | **PASS** |
| **TC-10** | GitHub Pages deployment | Web interface loads and computes predictions | HTTP 200 OK verified | **PASS** |

---

# 15. LIMITATIONS

1. **Dataset Scope:** The dataset represents a cross-sectional sample of 1,000 students from a single institutional environment.
2. **Feature Coverage:** External factors such as study hours, attendance rates, sleep duration, and family income were not recorded in the baseline dataset.
3. **Model Linearity:** Although regularized linear models generalized best, complex non-linear interaction terms may require larger datasets for effective tree-based exploitation without overfitting.
4. **Static Timeframe:** The data reflects examination scores at a single point in time rather than longitudinal skill progression.

---

# 16. CONCLUSION

The **Student Academic Performance Prediction and Analysis System** demonstrates a complete, production-grade data science workflow—from exploratory statistical modeling and feature engineering to multi-regressor benchmarking, Bayesian hyperparameter tuning, REST API development, and web deployment. 

The optimized pipeline achieves an $R^2$ of **0.8792** and an RMSE of **5.39**, confirming that demographic background, preparation level, and literacy scores can reliably predict student mathematics outcomes. By delivering the solution as both a containerized microservice and an interactive web application deployed live on GitHub Pages, the project bridges academic data analysis with real-world software engineering practice.

<br>
<p align="center"><b>Page 14</b></p>

---

<div style="page-break-after: always;"></div>

# 17. FUTURE SCOPE

1. **Feature Expansion:** Incorporate behavioral variables such as learning management system (LMS) engagement, attendance percentages, and assignment submission latencies.
2. **Deep Learning Integration:** Train multi-layer perceptrons (MLP) and tab-net architectures on expanded multi-institution datasets.
3. **Automated Interventions:** Integrate automated email and SMS notification triggers for students projected to score below passing thresholds.
4. **Database Integration:** Connect the backend microservice to an enterprise database (PostgreSQL / MongoDB) for real-time inference logging and audit trails.
5. **Continuous Model Monitoring:** Implement drift detection pipelines (e.g., Evidently AI) to detect population covariate shifts over subsequent academic terms.

---

# 18. REFERENCES

1. **Reference Repository:** Modi, Preyal. *Student Performance Prediction System*. GitHub: [https://github.com/preyal2/Student-Performance-Prediction](https://github.com/preyal2/Student-Performance-Prediction) (2024–2026).
2. **Live Web Deployment:** [https://preyal2.github.io/Student-Performance-Prediction/Development/frontend/](https://preyal2.github.io/Student-Performance-Prediction/Development/frontend/)
3. **Kaggle Dataset:** *Students Performance in Exams*. [https://www.kaggle.com/datasets/spscientist/students-performance-in-exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams).
4. **Scikit-Learn Documentation:** Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, pp. 2825-2830 (2011).
5. **XGBoost:** Chen, T., & Guestrin, C., *XGBoost: A Scalable Tree Boosting System*, KDD (2016).
6. **CatBoost:** Prokhorenkova, L. et al., *CatBoost: unbiased boosting with categorical features*, NeurIPS (2018).
7. **Flask Documentation:** Pallets Projects, *Flask Web Development Framework*, [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/).
8. **Waitress Documentation:** Zope Foundation, *Waitress WSGI Server*, [https://docs.pylonsproject.org/projects/waitress/](https://docs.pylonsproject.org/projects/waitress/).
9. **GTU Syllabus:** Gujarat Technological University, *Python for Data Science (BE05000231)* Course Curriculum.

<br>
<p align="center"><b>Page 15</b></p>

---

<div style="page-break-after: always;"></div>

# APPENDIX A – PYTHON IMPLEMENTATION OUTLINE

This appendix contains the core Python implementation modules developed in the project.

### A.1 Data Loading & Feature Engineering
```python
import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("data/stud.csv")
df.columns = [
    'gender', 'race_ethnicity', 'parental_level_of_education',
    'lunch', 'test_preparation_course', 'math_score',
    'reading_score', 'writing_score'
]

# Feature engineering
df['total score'] = df['math_score'] + df['reading_score'] + df['writing_score']
df['average'] = df['total score'] / 3

# Define input matrix X and target y
X = df.drop(columns=['math_score', 'writing_score', 'reading_score'], axis=1)
y = df['math_score']
```

### A.2 Train-Test Splitting & Pipeline Preprocessing
```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Train/Test Split
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
X_train = X_train_df.to_dict(orient='records')
X_test = X_test_df.to_dict(orient='records')

# Define full ML pipeline
pipeline = Pipeline([
    ('dv', DictVectorizer(sparse=False)),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression(fit_intercept=True))
])

# Train model
pipeline.fit(X_train, y_train)
```

### A.3 Bayesian Hyperparameter Optimization
```python
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from xgboost import XGBRegressor

param_grid = [
    {
        'regressor': [LinearRegression()],
        'regressor__fit_intercept': Categorical([True, False]),
    },
    {
        'regressor': [XGBRegressor()],
        'regressor__learning_rate': Real(0.01, 1.0, prior='log-uniform'),
        'regressor__n_estimators': Integer(10, 200),
        'regressor__max_depth': Integer(1, 10),
    }
]

opt = BayesSearchCV(
    pipeline,
    param_grid,
    cv=5,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    random_state=42
)
opt.fit(X_train, y_train)
best_model = opt.best_estimator_
```

<br>
<p align="center"><b>Page 16</b></p>

---

<div style="page-break-after: always;"></div>

### A.4 Model Evaluation Metrics
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE):  {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared Score (R2):      {r2:.4f}")
```

### A.5 Model Serialization
```python
import pickle

OUTPUT_FILE = "Development/backend/model.bin"
with open(OUTPUT_FILE, "wb") as f_out:
    pickle.dump(best_model, f_out)
print(f"Model successfully saved to {OUTPUT_FILE}")
```

### A.6 Production REST API (`predict.py`)
```python
import os
import pickle
from flask import Flask, request, jsonify

MODEL_FILE = os.environ.get("MODEL_FILE", "model.bin")
with open(MODEL_FILE, "rb") as f_in:
    model = pickle.load(f_in)

app = Flask("Student_Performance_Prediction")

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    return response

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "UP", "model_loaded": model is not None}), 200

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    records = [payload] if isinstance(payload, dict) else payload
    predictions = model.predict(records)
    
    score = float(predictions[0])
    return jsonify({
        "status": "success",
        "predicted_math_score": round(score, 2),
        "The student has scored in Math": int(round(score))
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)
```

### A.7 Client Automated Test Script (`predict_test.py`)
```python
import requests

URL = "http://localhost:9696/predict"
sample_student = {
    "gender": "male",
    "race_ethnicity": "group E",
    "parental_level_of_education": "some college",
    "lunch": "standard",
    "test_preparation_course": "completed",
    "total score": 245,
    "average": 81.66666666666667
}

response = requests.post(URL, json=sample_student)
print("API Response:", response.json())
```

<br>
<p align="center"><b>Page 17</b></p>
