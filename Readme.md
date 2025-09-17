# Loan Default Prediction Project

## Overview

This project implements a predictive machine learning pipeline to identify loan defaults using structured financial and personal data. The goal is to build a robust, interpretable, and production-ready model to help banking analysts and risk officers proactively identify high-risk borrowers.

---

## Dataset Description

- The dataset consists of applicant demographics, loan details, credit scores, income, employment info, and other relevant features.
- The target variable is binary: `Default` (1 if defaulted, 0 otherwise).
- Data cleaning involved handling missing values with median/mode imputation and removing duplicates.
- Binary categorical variables were encoded as 0/1 and multiclass categorical variables label-encoded.
- Numerical features were standardized using `StandardScaler`.

---

## Methodology

### 1. Data Preprocessing

- Missing value imputation for both numerical and categorical features.
- Feature engineering including the creation of interaction terms (e.g., `Income × CreditScore`).
- Train-test splitting with stratification keeping class balance.
- Class imbalance corrected using SMOTE oversampling technique on the training data.

### 2. Baseline Model Training

- Trained Logistic Regression, Random Forest, and XGBoost models on resampled data.
- Evaluated using metrics focused on class 1 (default): ROC AUC, recall, precision, and F1-score.
- Adjusted prediction threshold from default 0.5 to 0.4 to improve recall on defaulters.

### 3. Model Selection and Hyperparameter Tuning

- Logistic Regression was selected based on a balance of high recall (0.74) of defaulters, interpretability, and strong ROC AUC (~0.79).
- Hyperparameters optimized using `GridSearchCV` over regularization strength `C` and solver options.

### 4. Deep Learning Model

- Developed a feedforward neural network with dropout layers to compare classical ML model performance.
- Achieved ROC AUC of 0.74 and recall 0.62, showing competitive but slightly lower recall than logistic regression.

### 5. Model Explainability

- Employed SHAP (SHapley Additive exPlanations) to interpret predictions of logistic regression at global and local levels.
- This step ensures compliance and trustworthiness for banking risk management.

### 6. Deployment and Monitoring

- Saved the tuned logistic regression and scaler objects using `joblib`.
- Recommended creation of APIs or batch jobs for production inference.
- Suggested monitoring for model drift and scheduled retraining as new data arrives.

---

## Results and Model Comparison

| Model                     | ROC AUC | Recall (Default) | Precision (Default) |
|---------------------------|---------|------------------|---------------------|
| Logistic Regression (Base) | ~0.73   | ~0.65            | 0.19                |
| Random Forest (Base)      | ~0.72   | ~0.44            | 0.27                |
| XGBoost (Base)            | ~0.72   | ~0.29            | 0.34                |
| Logistic Regression (Tuned) | **0.79** | **0.74**         | 0.19                |
| Deep Learning             | 0.74    | 0.62             | 0.23                |

- Logistic Regression with hyperparameter tuning was the strongest model with the best recall and strong ROC AUC.
- Deep Learning provides a tradeoff with slightly better precision but lower recall, making LR preferable when missing defaults is costlier.
  
---

## How to Run

1. Clone the repo and set up Python environment.
2. Install dependencies (e.g., scikit-learn, imbalanced-learn, xgboost, tensorflow, shap).
3. Run `data_preprocessing.py` to clean and prepare the dataset.
4. Run `train_models.py` for baseline training and evaluation.
5. Run `hyperparameter_tuning.py` for fine-tuning logistic regression.
6. Run `deep_learning.py` to train the neural network model.
7. Run `explainability.py` to generate SHAP explanations.
8. Run `save_model.py` to save final models and scalers for deployment.

---

## Future Improvements

- Integration with real-time loan application systems for live scoring.
- Incorporation of additional features (e.g., credit bureau data, transaction history).
- Automate threshold selection based on business objectives.
- Build interactive dashboards for risk officers to explore predictions and explanations.
