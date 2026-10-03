# GTU – Institute of Technology and Research
### Department of Computer Engineering

---

# REPORT
### On
## **Student Performance Prediction System**
#### *Using Python, Data Science and Machine-Learning Regression*

---

### **Submitted by:**
- **Name:** Modi Preyal Ashishkumar
- **Enrollment No.:** 251043107004
- **Class & Division:** B.E. Computer Engineering, Semester 5 – Division B

### **Under the guidance of:**
- **Dr. Vishal G. Barot**  
  *Assistant Professor, Department of Computer Engineering*

**Academic Year:** 2026 - 27

---

<div style="page-break-after: always;"></div>

# CERTIFICATE

This is to certify that the micro project entitled **“Student Performance Prediction System”** has been carried out and submitted as an Application/Software Development Micro Project (Problem-Based Learning) for the subject **Python for Data Science (BE05000231)** in the Department of Computer Engineering, **GTU – Institute of Technology and Research**, during the academic year 2026-27.

The project demonstrates Python programming, data cleaning, exploratory data analysis, visualization, feature engineering and machine-learning regression using Pandas, NumPy, Matplotlib, Seaborn and scikit-learn. The project uses the publicly available “Students Performance in Exams” dataset and the supplied Jupyter notebook, and has been adapted for academic study.

<br><br>

**Faculty In-charge:** ______________________  
<br>
**Signature:** ______________________  

<br>
<p align="center"><b>Page 2</b></p>

---

<div style="page-break-after: always;"></div>

# DECLARATION

I hereby declare that this report has been prepared for academic learning as part of the Python for Data Science course. The implementation and explanation have been adapted from the supplied Jupyter notebook and the public Kaggle dataset “Students Performance in Exams”, and the external data source is acknowledged in this report. The report does not claim ownership of the original dataset.

The project uses a student-performance dataset containing demographic and background attributes together with math, reading and writing scores. The analysis and prediction workflow described here is intended for educational demonstration only. It must not be used to make real decisions about individual students.

| Student Name | Enrollment No. | Signature |
| :--- | :--- | :--- |
| Modi Preyal Ashishkumar | 251043107004 | ______________________ |

<br>
<p align="center"><b>Page 3</b></p>

---

<div style="page-break-after: always;"></div>

# ACKNOWLEDGEMENT

I sincerely thank my guide, **Dr. Vishal G. Barot**, Assistant Professor, and the Department of Computer Engineering, **GTU – Institute of Technology and Research**, for providing me the opportunity to undertake this micro project as part of the Python for Data Science course and for their valuable guidance.

I also acknowledge the Python open-source community and the developers of Pandas, NumPy, Matplotlib, Seaborn and scikit-learn, which provide the tools used for data handling, analysis, visualization, model training and evaluation.

The “Students Performance in Exams” dataset published on Kaggle was used as the data source for this project, and the supplied Jupyter notebook (Student Performance Indicator) was used as the main technical reference for the machine-learning workflow.

<br>
<p align="center"><b>Page 4</b></p>

---

<div style="page-break-after: always;"></div>

# ABSTRACT

This micro project presents a Python-based Student Performance Prediction System that studies how background factors such as gender, race/ethnicity, parental level of education, lunch type and test preparation course relate to exam scores. The dataset contains 1,000 student records and 8 columns, with no missing values and no duplicate rows. The project performs data checks, feature engineering (total score and average), exploratory data analysis and regression modelling to predict the math score.

The notebook builds a scikit-learn pipeline (DictVectorizer for one-hot encoding, StandardScaler and a regressor) and uses RandomizedSearchCV with 5-fold cross-validation over eight model families. Linear Regression was selected as the best model and the notebook reports a test MAE of 2.8200, RMSE of 3.5927 and R² of 0.9470, with the trained model saved as model.bin.

While reviewing the notebook for this report, a data-leakage issue was found: the input features include “total score” and “average”, both of which are calculated from the math score that is being predicted. The report therefore also presents an independent re-run of the experiment, performed for this report on the same data and the same 80/20 split. Using only the background attributes, test R² is 0.176; adding the reading and writing scores gives R² of 0.880 (RMSE 5.39). These corrected figures give a more honest picture of what the data can predict.

The project is designed as an academic micro project and is not a production system for assessing real students.

<br>
<p align="center"><b>Page 5</b></p>

---

<div style="page-break-after: always;"></div>

# TABLE OF CONTENTS

| Sr. No. | Content | Page No. |
| :---: | :--- | :---: |
| 1 | Introduction | 7 |
| 2 | Problem Statement | 7 |
| 3 | Objectives | 8 |
| 4 | Scope | 8 |
| 5 | Proposed System | 8 |
| 6 | Tools and Technologies | 10 |
| 7 | Dataset Description | 10 |
| 8 | System Requirements | 11 |
| 9 | Methodology | 11 |
| 10 | Data Preparation | 12 |
| 11 | Application Development | 13 |
| 12 | Data Analysis | 14 |
| 13 | Results and Discussion | 17 |
| 14 | Testing | 18 |
| 15 | Limitations | 18 |
| 16 | Conclusion | 19 |
| 17 | Future Scope | 19 |
| 18 | References | 19 |
| — | Appendix A – Python Implementation | 20 |

<br>
<p align="center"><b>Page 6</b></p>

---

<div style="page-break-after: always;"></div>

# 1. INTRODUCTION

Academic performance is influenced by many factors outside the classroom, such as the learning environment at home, nutrition, parental education and access to preparation resources. Educational data can be analyzed to understand these relationships and, with machine learning, to build models that estimate a student’s expected score from known attributes.

This project develops a Python-based student-performance analysis and prediction workflow using the public “Students Performance in Exams” dataset. The notebook loads the data with Pandas, checks data quality, engineers total and average score features, visualizes how each background factor relates to scores, trains regression models in a scikit-learn pipeline and saves the selected model.

The supplied notebook is titled “Student Performance Indicator” and the dataset file is stud.csv, which contains 1,000 student records with 8 columns.

### 1.1 Project Domain
Educational data analytics, Python for Data Science, exploratory data analysis, machine-learning regression and model evaluation.

### 1.2 Project Rationale
- A clean real-world style dataset is practical for learning the complete data-science life cycle.
- The project combines data cleaning, visualization, feature engineering and modelling rather than only displaying raw data.
- Mixed categorical and numerical features demonstrate one-hot encoding and scaling inside a pipeline.
- Reviewing the workflow critically, for example checking for data leakage, is an important data-science skill.

---

# 2. PROBLEM STATEMENT

Develop a Python-based system that loads student exam data, cleans and analyzes it, studies how gender, ethnicity group, parental education, lunch type and test-preparation course are associated with exam scores, and trains a regression model to predict a student’s math score.

### 2.1 Main Functions
- Load the student dataset into a Pandas DataFrame.
- Inspect data shape, types, unique values, missing values and duplicates.
- Create the total score and average features.
- Visualize score distributions and compare groups with plots and summaries.
- Build a pipeline of one-hot encoding, scaling and a regressor and select the best model by cross-validation.
- Evaluate predictions using MAE, MSE, RMSE and R², then save the model.

<br>
<p align="center"><b>Page 7</b></p>

---

<div style="page-break-after: always;"></div>

# 3. OBJECTIVES

- To understand the structure of a real-world student-performance dataset.
- To perform data preparation and exploratory analysis using Python.
- To study the effect of lunch type, parental education, ethnicity group, gender and test preparation on scores.
- To build a regression model that predicts the math score.
- To evaluate model performance with standard regression metrics and verify the setup for data leakage.
- To present the workflow in a structured academic report.

---

# 4. SCOPE

- Analysis of the supplied 1,000-record student-performance dataset.
- Data quality checks and feature engineering.
- Feature-wise exploratory visualization.
- Regression modelling of the math score with cross-validated model selection.
- Model evaluation, model saving and single-record testing.
- Academic demonstration using a Jupyter/Google Colab style Python workflow.

**Out of scope:** live student data, real-time grading, decisions about real students, scholarship or admission screening, production deployment and guaranteed predictions. The supplied notebook is also not a complete web/mobile application.

**Important implementations note:** the notebook uses total score and average as input features, and both are derived from the math score being predicted. This report keeps the notebook’s reported results but also presents a leakage-free re-run (Section 12.7), clearly labelled as computed for this report.

---

# 5. PROPOSED SYSTEM

The proposed system follows a data-science pipeline. Raw student records are loaded, inspected and prepared. Exploratory analysis is used to understand how each factor relates to scores. The prepared data is split into training and test sets, passed through a pipeline with a candidate regressor, and the best model is evaluated and saved. The workflow is shown in Figure 1.

```
                  ┌────────────────────────┐
                  │    Student Dataset     │
                  │       (stud.csv)       │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │     Data Cleaning      │
                  │        & Checks        │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │  Feature Engineering   │
                  │ (total score, average) │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │         EDA &          │
                  │     Visualization      │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │     Pre-processing     │
                  │   (One-hot + Scaling)  │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │     Model Training     │
                  │        & Tuning        │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │       Evaluation       │
                  │   (MAE, RMSE, R^2)     │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │      Model Saving      │
                  │       & Testing        │
                  └────────────────────────┘
```
<p align="center"><b>Figure 1: Proposed Student Performance Prediction Workflow</b></p>

### 5.1 Modules

| Module | Purpose |
| :--- | :--- |
| **Data Loading** | Read stud.csv into Pandas. |
| **Data Cleaning** | Check missing values, duplicates and types; rename columns. |
| **Feature Engineering** | Create total score and average. |
| **EDA** | Study distributions and compare groups. |
| **Pre-processing** | One-hot encode categorical features and scale features. |
| **Model Training** | Randomized search over regressors with 5-fold cross-validation. |
| **Evaluation** | Calculate MAE, MSE, RMSE and R². |

<br>
<p align="center"><b>Page 8</b></p>

---

<div style="page-break-after: always;"></div>

### 5.2 System Module Details

| Module | Input | Output |
| :--- | :--- | :--- |
| **Data Loading** | `stud.csv` | DataFrame with 1,000 rows × 8 columns |
| **Cleaning** | Raw DataFrame | Verified data: 0 missing, 0 duplicates |
| **Feature Engineering** | Three score columns | total score and average |
| **EDA** | Prepared dataset | Plots, grouped summaries and insights |
| **Data Splitting** | Features X, target math_score | 800 training and 200 test records |
| **Pipeline** | Training records | Encoded and scaled feature matrix |
| **Model Selection** | Pipeline + parameter space | Best estimator |
| **Evaluation** | Predicted + actual math score | Error metrics |

The system is primarily a notebook-driven analytical application. Its main result is a reproducible sequence of Python operations rather than a GUI. The saved model can later be placed behind a Streamlit or Flask interface.

---

# 6. TOOLS AND TECHNOLOGIES

| Technology | Purpose |
| :--- | :--- |
| **Python 3** | Core programming language |
| **Pandas** | Data loading, grouping and summaries |
| **NumPy** | Numerical operations |
| **Matplotlib** | Plots and visualization |
| **Seaborn** | Statistical visualization used by the notebook |
| **scikit-learn** | Pipeline, DictVectorizer, StandardScaler, regressors, search and metrics |
| **XGBoost, CatBoost** | Boosting regressors imported by the notebook |
| **scikit-optimize** | Search-space definitions (Real, Integer, Categorical) |
| **pickle** | Saving the trained model (`model.bin`) |
| **Jupyter Notebook / Colab** | Interactive development and experiment execution |
| **CSV** | Student dataset format |

- **Development Environment:** The supplied notebook is written as an interactive Python notebook. It can be adapted for VS Code, Jupyter Notebook or Google Colab.
- **Data Source:** Kaggle – “Students Performance in Exams”, [https://www.kaggle.com/datasets/spscientist/students-performance-in-exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

---

# 7. DATASET DESCRIPTION

The dataset `stud.csv` is shown by the executed notebook as 1,000 rows and 8 columns. The fields represent five background attributes and three exam scores. There are no missing values and no duplicate rows.

| Field | Description | Type |
| :--- | :--- | :--- |
| `gender` | Sex of the student (female 518, male 482) | Text |
| `race/ethnicity` | Anonymized group A–E | Text |
| `parental level of education` | Parents’ highest education (6 levels) | Text |
| `lunch` | Standard (645) or free/reduced (355) | Text |
| `test preparation course` | None (642) or completed (358) | Text |
| `math score` | Exam score, 0–100 | Numeric |
| `reading score` | Exam score, 17–100 | Numeric |
| `writing score` | Exam score, 10–100 | Numeric |

The mean scores are close together (66.09 math, 69.17 reading, 68.05 writing) and the standard deviations are similar (15.16, 14.60, 15.20). Math has the lowest minimum score (0), while the lowest reading and writing scores are 17 and 10.

<br>
<p align="center"><b>Page 9 & 10</b></p>

---

<div style="page-break-after: always;"></div>

# 8. SYSTEM REQUIREMENTS

### 8.1 Hardware
- Dual-core processor or better
- 4 GB RAM or more
- At least 1 GB free storage
- Standard display, keyboard and mouse

### 8.2 Software
- Python 3.x
- VS Code, Jupyter Notebook or Google Colab
- Pandas, NumPy, Matplotlib and Seaborn
- scikit-learn, scikit-optimize, XGBoost and CatBoost

---

# 9. METHODOLOGY

| Step | Activity |
| :---: | :--- |
| **1** | Obtain the stud.csv dataset (Kaggle “Students Performance in Exams”). |
| **2** | Load the dataset using Pandas. |
| **3** | Inspect shape, data types, missing values, duplicates and statistics. |
| **4** | Rename columns and create total score and average. |
| **5** | Perform exploratory analysis and visualization. |
| **6** | Split data 80/20 with random_state = 42 (800 training, 200 test records). |
| **7** | Build the pipeline: DictVectorizer → StandardScaler → regressor. |
| **8** | Run RandomizedSearchCV (5-fold CV) and select the best model. |
| **9** | Evaluate predictions, verify the setup, save the model and prepare the report. |

---

# 10. DATA PREPARATION

### 10.1 Load Data
The notebook loads the CSV file using Pandas. The resulting DataFrame contains five categorical and three numerical fields.
```python
import pandas as pd
import numpy as np

df = pd.read_csv("data/stud.csv")
print(df.shape)  # (1000, 8)
print(df.head())
```

### 10.2 Check Missing Values and Duplicates
```python
print(df.isna().sum())       # 0 in every column
print(df.duplicated().sum()) # 0
df.info()
print(df.nunique())
```
The executed notebook reports zero missing values and zero duplicates. The number of unique values is 2 (gender), 5 (race/ethnicity), 6 (parental education), 2 (lunch), 2 (test preparation), 81 (math), 72 (reading) and 77 (writing). No imputation or row removal was needed.

### 10.3 Column Renaming
```python
df.columns = [
    "gender", "race_ethnicity", "parental_level_of_education",
    "lunch", "test_preparation_course",
    "math_score", "reading_score", "writing_score"
]
```

### 10.4 Feature Engineering
```python
df["total score"] = df["math_score"] + df["reading_score"] + df["writing_score"]
df["average"] = df["total score"] / 3
```

| Measure | Math | Writing | Reading |
| :--- | :---: | :---: | :---: |
| Students with full marks (100) | 7 | 14 | 17 |
| Students with score $\le$ 20 | 4 | 3 | 1 |

*Students performed worst in math and best in reading.*

### 10.5 Why Cleaning Matters
- Missing or duplicate records can bias statistics and model training.
- Consistent column names avoid errors in later code.
- Knowing the data types decides how each feature is encoded.

<br>
<p align="center"><b>Page 11 & 12</b></p>

---

<div style="page-break-after: always;"></div>

# 11. APPLICATION DEVELOPMENT

The analytical application is organized as a notebook workflow rather than a menu-driven console program. Each stage produces an intermediate result that can be inspected visually.

### 11.1 Functional Requirements

| No. | Function | Description |
| :---: | :--- | :--- |
| **1** | Load Dataset | Read and validate stud.csv. |
| **2** | Check Quality | Check missing values, duplicates and types. |
| **3** | Engineer Features | Create total score and average. |
| **4** | Explore Data | Plot distributions and group comparisons. |
| **5** | Split Data | 80% training, 20% test records. |
| **6** | Build Pipeline | One-hot encoding, scaling and regressor. |
| **7** | Select Model | Randomized search with 5-fold cross-validation. |
| **8** | Evaluate | Calculate MAE, MSE, RMSE and R². |
| **9** | Save & Test | Pickle the model and predict for one student. |

### 11.2 User Workflow
- Open the notebook in Jupyter/Colab.
- Place stud.csv in the data folder.
- Run cells in order from data loading to testing.
- Inspect generated graphs and evaluation metrics.

### 11.3 Pipeline Design
A single pipeline applies exactly the same transformations to training data, test data and every cross-validation fold.

| Pipeline step | Component | Role |
| :--- | :--- | :--- |
| **dv** | `DictVectorizer(sparse=False)` | One-hot encodes categorical fields |
| **scaler** | `StandardScaler()` | Standardizes all columns |
| **regressor** | Candidate model | Predicts the math score |

### 11.4 Candidate Models
The randomized search covers Linear Regression, Support Vector Regression, K-Nearest Neighbors, AdaBoost, Decision Tree, XGBoost, CatBoost and Random Forest, each with its own hyperparameter space.

### 11.5 Data Splitting and Model Search
```python
x = df.drop(columns=["math_score", "writing_score", "reading_score"])
y = df["math_score"]

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)
x_train = x_tr.to_dict(orient="records")
x_test = x_te.to_dict(orient="records")

opt = RandomizedSearchCV(
    pipeline, param_grid, cv=5, n_jobs=-1, scoring="neg_mean_squared_error"
)
opt.fit(x_train, y_tr.values)
```

<br>
<p align="center"><b>Page 13</b></p>

---

<div style="page-break-after: always;"></div>

# 12. DATA ANALYSIS

### 12.1 Group-wise Average Scores
The notebook compares scores across every categorical feature, verified from `stud.csv`:

| Feature | Group | Math | Reading | Writing | Average |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Gender** | female | 63.63 | 72.61 | 72.47 | 69.57 |
| | male | 68.73 | 65.47 | 63.31 | 65.84 |
| **Lunch** | standard | 70.03 | 71.65 | 70.82 | 70.84 |
| | free/reduced | 58.92 | 64.65 | 63.02 | 62.20 |
| **Test prep.** | completed | 69.70 | 73.89 | 74.42 | 72.67 |
| | none | 64.08 | 66.53 | 64.50 | 65.04 |
| **Race/ethnicity** | group A | 61.63 | 64.67 | 62.67 | 62.99 |
| | group E | 73.82 | 73.03 | 71.41 | 72.75 |
| **Parental edu.** | high school | 62.14 | 64.70 | 62.45 | 63.10 |
| | master's degree | 69.75 | 75.37 | 75.68 | 73.60 |

### 12.2 Exploratory Observations
- **Gender:** Females score higher overall and in reading and writing; males score higher in math (68.73 vs 63.63).
- **Lunch:** Standard lunch students average about 8.6 points higher (70.84 vs 62.20), for both genders.
- **Parental education:** Master’s and bachelor’s degree parents are associated with the highest scores; high school with the lowest.
- **Race/ethnicity:** Group E scores highest and group A lowest. Labels are anonymized and differences are not causal.
- **Test preparation:** Completing the course is associated with higher scores in all subjects (about 5.6 points in math, 7.4 in reading, 9.9 in writing).
- Most students score between 60 and 80 in math and between 50 and 80 in reading and writing. The three scores are strongly correlated; reading and writing are almost perfectly related ($r = 0.955$).

### 12.3 Model Selection and Fitting

| Parameter | Value |
| :--- | :--- |
| **Target variable** | `math_score` |
| **Input features (notebook)** | 5 categorical + `total score` + `average` |
| **Training / test records** | 800 / 200 |
| **Search method** | `RandomizedSearchCV`, 5-fold CV, negative MSE |
| **Best model** | Linear Regression (`fit_intercept = True`) |
| **Best CV score (negative MSE)** | -13.6670 |

### 12.4 Prediction Generation
```python
y_test_pred = best_model.predict(x_test)

with open("model.bin", "wb") as f_out:
    pickle.dump(best_model, f_out)

best_model.predict([x_test[4]])[0]  # 87.37 (actual = 84)
```

<br>
<p align="center"><b>Page 14 & 15</b></p>

---

<div style="page-break-after: always;"></div>

### 12.5 Evaluation Measures

| Metric | Purpose |
| :--- | :--- |
| **MAE** | Average absolute prediction error in score points. |
| **MSE** | Mean squared error; penalizes larger errors. |
| **RMSE** | Square root of MSE; on the score scale. |
| **R²** | Proportion of variance in math score explained by the model. |

### 12.6 Model Evaluation

| Metric | Training set | Test set |
| :--- | :---: | :---: |
| **RMSE** | 3.6252 | 3.5927 |
| **MAE** | 2.8984 | 2.8200 |
| **R² score** | 0.9417 | 0.9470 |
| **MSE** | 13.1420 | 12.9073 |

Training and test metrics are close, so the model does not appear to overfit. These values are reproduced from the executed notebook output. The sample predictions were 81.96 (actual 91), 55.93 (53), 78.17 (80), 76.87 (74) and 87.37 (84).

### 12.7 Leakage Check and Corrected Experiment
**Computed for this report:** the notebook’s features total score and average contain the math score itself, which is the target. This target leakage inflates R². The experiment was re-run on stud.csv with the same split and pipeline structure.

| Setup | Input features | Test R² | MAE | RMSE |
| :--- | :--- | :---: | :---: | :---: |
| **A – Notebook** | 5 categorical + total + average | 0.947 | 2.81 | 3.59 |
| **B – Background only** | 5 categorical features | 0.176 | 11.27 | 14.16 |
| **C – Background + reading + writing** | 5 categorical + reading + writing | 0.880 | 4.22 | 5.39 |

Setup A reproduces the notebook (R² 0.947), confirming the replication. Background attributes alone explain only about 18% of the variation in math score. When reading and writing scores are used instead, R² is 0.880 and Linear Regression remains the best of eight models tested (Gradient Boosting 0.872, Random Forest 0.860, SVR 0.817, KNN 0.487). The notebook search also scored only Linear Regression; the other nine sampled candidates returned NaN.

---

# 13. RESULTS AND DISCUSSION

| Measure | Result | Meaning |
| :--- | :--- | :--- |
| **Dataset size** | 1000 × 8 | Student records and attributes |
| **Missing / duplicate values** | 0 / 0 | Clean dataset |
| **Target variable** | math_score | Predicted using regression |
| **Train / test split** | 800 / 200 | 80/20, random_state = 42 |
| **Selected model** | Linear Regression | Best model in the notebook search |
| **Notebook test MAE** | 2.8200 | Average error in score points |
| **Notebook test RMSE** | 3.5927 | Root mean squared error |
| **Notebook test R²** | 0.9470 | Inflated by data leakage |
| **Background-only test R²** | 0.176 | Computed for this report |
| **Realistic test R² / RMSE** | 0.880 / 5.39 | With reading and writing scores |

The results show that the implemented workflow can reproduce a structured prediction experiment on student data. The evaluation is limited to the specific train/test split used in the notebook.

### 13.1 Observations
- Lunch type, parental education, ethnicity group, gender and test preparation show clear group-level score differences.
- Females lead in reading and writing and in the overall average, while males lead in math.
- Completing the test preparation course is associated with higher scores in every subject.
- Reading and writing scores are by far the best predictors of math score; background attributes alone are weak (R² ≈ 0.18).
- The notebook’s MAE of about 2.8 points is optimistic because of leakage; about 4.2 points is a realistic error with reading and writing scores.
- Linear models perform best on this small, mostly linear dataset.

**Discussion:** A high metric is not automatically a good result. Checking which features enter the model is as important as tuning it. Group-level patterns do not make an individual score predictable.

<br>
<p align="center"><b>Page 16 & 17</b></p>

---

<div style="page-break-after: always;"></div>

# 14. TESTING

Testing for this academic system focuses on data loading, preprocessing, feature engineering, model fitting and metric calculation.

| Test Case | Expected Result | Result |
| :--- | :--- | :---: |
| **Load CSV** | DataFrame created successfully. | **Pass** |
| **Check shape** | 1000 rows and 8 columns. | **Pass** |
| **Check missing values** | No missing values found. | **Pass** |
| **Check duplicates** | No duplicate rows found. | **Pass** |
| **Feature engineering** | total score and average created. | **Pass** |
| **Train/test split** | 800 training and 200 test records. | **Pass** |
| **Fit pipeline** | Pipeline fitted without error. | **Pass** |
| **Calculate metrics** | MAE, MSE, RMSE and R² produced. | **Pass** |
| **Save model** | `model.bin` written with pickle. | **Pass** |
| **Single prediction** | Predicted 87.37 for actual math score 84. | **Pass** |
| **Leakage check** | Target-derived features absent from X. | **Fail in notebook; corrected in 12.7** |

---

# 15. LIMITATIONS

- The dataset has only 1,000 records from one public source, so results may not generalize to other schools.
- The notebook feature set contains target leakage, so its reported accuracy is overstated.
- The hyperparameter search sampled only 10 candidates and most returned NaN scores, so Linear Regression won by default.
- Only one random train/test split was used.
- Only math score is predicted.
- Some notebook comments are imprecise (for example 48% female / 52% male, whereas the data show 51.8% / 48.2%); this report uses recomputed values.
- Attributes such as race/ethnicity and parental education are sensitive; the model must not be used to label, rank or select real students.

---

# 16. CONCLUSION

The Student Performance Prediction System demonstrates a complete Python data-science workflow from raw student records to data checks, feature engineering, exploratory analysis, pipeline-based regression and model saving. The project uses Pandas and NumPy for data handling, Matplotlib and Seaborn for visualization, and scikit-learn for pre-processing, model selection and evaluation.

The executed notebook provides measurable results, including MAE 2.8200, RMSE 3.5927 and R² 0.9470, but these are inflated by data leakage. A leakage-free re-run gave R² 0.176 for background attributes only and 0.880 (RMSE 5.39) with reading and writing scores. These values demonstrate the mechanics of a prediction experiment and should not be interpreted as a guarantee of accuracy for real students.

<br>
<p align="center"><b>Page 18 & 19</b></p>

---

<div style="page-break-after: always;"></div>

# 17. FUTURE SCOPE

- Remove total score and average from the features and retrain without leakage.
- Fix the parameter search and use more iterations or Bayesian search.
- Use repeated cross-validation for more reliable estimates.
- Predict reading and writing scores or build a pass/fail classifier.
- Add model explanation tools such as SHAP.
- Build a Streamlit dashboard or API that loads model.bin.
- Test on larger datasets and examine fairness across groups.

---

# 18. REFERENCES

- Kaggle, “Students Performance in Exams” dataset (spscientist),  
  [https://www.kaggle.com/datasets/spscientist/students-performance-in-exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- Supplied Jupyter notebook “Student Performance Indicator” and dataset file stud.csv.
- Python Software Foundation – Python Documentation.
- Pandas, NumPy, Matplotlib, Seaborn Documentation.
- scikit-learn Documentation – Pipelines, Model Selection and Metrics.
- XGBoost, CatBoost and scikit-optimize Documentation.
- GTU Syllabus – Python for Data Science (BE05000231).

*Source note: Notebook results are taken from the executed notebook output. Group averages, correlations and Section 12.7 were recomputed from stud.csv for this report.*

---

<div style="page-break-after: always;"></div>

# APPENDIX A – PYTHON IMPLEMENTATION

The following pages contain an academically adapted implementation outline. It is intentionally rewritten rather than copied verbatim from the supplied notebook.

### A.1 Data Loading
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("data/stud.csv")
df.columns = [
    "gender", "race_ethnicity", "parental_level_of_education",
    "lunch", "test_preparation_course",
    "math_score", "reading_score", "writing_score"
]
print(df.shape)
print(df.head())
```

### A.2 Basic Inspection
```python
print(df.isna().sum())
print(df.duplicated().sum())
df.info()
print(df.nunique())
print(df.describe())
```

### A.3 Feature Engineering
```python
df["total score"] = df["math_score"] + df["reading_score"] + df["writing_score"]
df["average"] = df["total score"] / 3
print((df.math_score == 100).sum(), (df.writing_score == 100).sum(), (df.reading_score == 100).sum())
```

### A.4 Exploratory Analysis
```python
sns.histplot(data=df, x="average", bins=30, kde=True)
plt.show()

num = ["math_score", "reading_score", "writing_score", "average"]
for col in ["gender", "lunch", "test_preparation_course",
            "parental_level_of_education", "race_ethnicity"]:
    print(df.groupby(col)[num].mean())

sns.boxplot(data=df[num]); plt.show()
sns.pairplot(df, hue="gender"); plt.show()
```

<br>
<p align="center"><b>Page 20</b></p>

---

<div style="page-break-after: always;"></div>

### A.5 Pipeline and Model Search
```python
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.pipeline import Pipeline

x = df.drop(columns=["math_score", "reading_score", "writing_score"])
y = df["math_score"]

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)
x_train = x_tr.to_dict(orient="records")
x_test = x_te.to_dict(orient="records")

pipeline = Pipeline([
    ("dv", DictVectorizer(sparse=False)),
    ("scaler", StandardScaler()),
    ("regressor", "passthrough")
])

opt = RandomizedSearchCV(
    pipeline, param_grid, cv=5, n_jobs=-1, scoring="neg_mean_squared_error"
)
opt.fit(x_train, y_tr.values)
best_model = opt.best_estimator_
```

### A.6 Model Evaluation
```python
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

pred = best_model.predict(x_test)
mae = mean_absolute_error(y_te, pred)
mse = mean_squared_error(y_te, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_te, pred)

print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)
# Reported notebook values: MAE 2.8200, RMSE 3.5927, R2 0.9470
```

### A.7 Model Saving and Testing
```python
with open("model.bin", "wb") as f_out:
    pickle.dump(best_model, f_out)

print(best_model.predict([x_test[4]])[0], y_te.values[4]) # 87.37 vs 84
```

### A.8 Leakage-Free Version (added for this report)
```python
from sklearn.linear_model import LinearRegression

base = ["gender", "race_ethnicity", "parental_level_of_education",
        "lunch", "test_preparation_course"]
features = base + ["reading_score", "writing_score"]  # no total/average
x = df[features]
y = df["math_score"]

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)
x_train = x_tr.to_dict(orient="records")
x_test = x_te.to_dict(orient="records")

pipe = Pipeline([
    ("dv", DictVectorizer(sparse=False)),
    ("sc", StandardScaler()),
    ("m", LinearRegression())
])
pipe.fit(x_train, y_tr.values)
pred = pipe.predict(x_test)

print(r2_score(y_te, pred), np.sqrt(mean_squared_error(y_te, pred)))
# Reported in Section 12.7: R2 0.880, RMSE 5.39
```

### A.9 Implementation Note
This appendix is a rewritten academic outline. The supplied notebook remains the source for the dataset structure, feature engineering, pipeline design and reported evaluation results. The leakage-free experiment in A.8 was added for this report.

<br>
<p align="center"><b>Page 21</b></p>
