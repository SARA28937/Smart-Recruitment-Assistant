# 🤖 Smart Recruitment Assistant

An end-to-end Machine Learning recruitment screening project that predicts whether a candidate should advance to the next stage of the hiring process.

## 🎯 Project Objective

Recruitment teams may receive a large number of applications for a single vacancy. This project explores how Machine Learning can support the initial screening stage by estimating a candidate's likelihood of job-change interest and converting that probability into a recruitment recommendation.

> **Important:** This project is an academic/portfolio decision-support system, not a replacement for human hiring decisions.

## 📊 Dataset

**Dataset:** HR Analytics: Job Change of Data Scientists

The dataset contains **19,158 records and 14 original features**.

The target distribution is imbalanced:

- `0` → 75.07%
- `1` → 24.93%

The project handles missing values, feature engineering, encoding, scaling, class imbalance, model comparison, hyperparameter tuning, and threshold selection.

## 🧹 Data Preparation

Main preprocessing steps:

1. Missing categorical values → `Unknown`
2. Missing numerical values → median imputation
3. Experience → numerical `experience_clean`
4. Last new job → numerical `last_new_job_clean`
5. Ordinal encoding for education, relevant experience, and university enrollment
6. One-hot encoding for categorical variables
7. Removed identifier/location columns that were not used by the final model
8. Outlier treatment using the IQR method
9. Train/test split with stratification
10. Standard scaling

## 🤖 Models Evaluated

### Logistic Regression
Baseline linear classification model.

### Random Forest
Non-linear ensemble model used as the main candidate model.

### Class-weighted models
`class_weight="balanced"` was tested to improve performance on the minority class.

### Hyperparameter tuning
RandomizedSearchCV with 5-fold cross-validation was used to tune the Random Forest using **F1 score**.

Best parameters found:

```text
n_estimators = 100
max_depth = 20
min_samples_split = 2
min_samples_leaf = 2
```

Best cross-validation F1 score:

```text
0.6200
```

## 📈 Final Model Performance

The final model is a tuned Random Forest.

Using the selected threshold of **0.40**:

| Metric | Class 0 | Class 1 |
|---|---:|---:|
| Precision | 0.91 | 0.53 |
| Recall | 0.78 | 0.78 |
| F1-score | 0.84 | 0.63 |

Overall:

- **Accuracy:** 77.61%
- **Macro F1:** 0.74
- **Weighted F1:** 0.79
- **ROC-AUC:** 0.7924
- **PR-AUC:** 0.5269

Confusion matrix:

```text
[[2232, 645],
 [ 213, 742]]
```

The threshold was selected to improve the balance between precision and recall for the positive class rather than optimizing accuracy alone.

## 🔍 Exploratory Analysis

The notebook also investigates:

- Class imbalance
- Missing values
- Outliers
- Feature importance
- City Development Index vs. target
- Target rate across City Development Index groups
- Model performance before and after class weighting
- Precision/Recall trade-offs at different thresholds

## 🖥️ Streamlit Application

The project includes an interactive Streamlit app where a user can enter candidate information and receive:

- Predicted recommendation
- Candidate probability
- Decision threshold
- Clear next-stage recommendation

The app uses the saved model, scaler, threshold, and feature names to keep inference consistent with the training pipeline.

## 🗂️ Suggested Repository Structure

```text
Smart-Recruitment-Assistant/
│
├── app.py
├── 01_Data_Understanding_EDA.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── recruitment_model.pkl
│   ├── scaler.pkl
│   ├── threshold.pkl
│   └── feature_names.pkl
│
└── data/
    └── README.md
```

## ▶️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Smart-Recruitment-Assistant.git
cd Smart-Recruitment-Assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Streamlit app

```bash
streamlit run app.py
```

## 📦 Required Model Files

The Streamlit application expects these files:

```text
recruitment_model.pkl
scaler.pkl
threshold.pkl
feature_names.pkl
```

Place them in the same directory as `app.py`, **or update the paths in `app.py`** if you use a `models/` folder.

## 🧠 Key Takeaways

This project demonstrates a complete ML workflow:

**Data Understanding → Cleaning → Feature Engineering → EDA → Model Comparison → Class Imbalance Handling → Hyperparameter Tuning → Threshold Optimization → Deployment**

The main lesson is that for an imbalanced recruitment problem, **accuracy alone is not enough**. Recall, precision, F1-score, ROC-AUC, PR-AUC, and the business meaning of the classification threshold should also be considered.

## 👩‍💻 Author

**Sara Tantawy**

Machine Learning & Data Analysis Portfolio Project

