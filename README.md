# AI-Powered Customer Churn Intelligence System

## Project Overview
This project implements an end-to-end **Machine Learning solution** to predict customer churn for a telecom company. The system is designed to help businesses identify customers likely to leave, understand the key factors influencing churn, and make data-driven retention decisions.

By analyzing customer demographics, subscription details, service usage, and payment methods, this system provides both predictive insights and actionable recommendations for reducing churn. The project demonstrates practical application of data preprocessing, feature engineering, model training, evaluation, and visualization.

## Key Features
- **Data Preprocessing & Cleaning:** Handles missing values, converts categorical variables into numeric formats, and ensures feature integrity.
- **Feature Engineering:** Encodes binary and multi-category variables for model readiness.
- **Model Training:** Implements Logistic Regression for churn prediction.
- **Baseline Comparison:** Uses a DummyClassifier to compare model performance against a majority-class baseline.
- **Evaluation Metrics:** Provides Accuracy, Confusion Matrix, Classification Report, and ROC-AUC score.
- **Business Insights:** Extracts top features driving churn and visualizes their impact.
- **Visualization:** Includes bar plots for top features and ROC curve for model performance.
- **Actionable Recommendations:** Suggests retention strategies for business stakeholders.

## Technologies Used
- **Language:** Python 
- **Libraries:** pandas, numpy, matplotlib, scikit-learn  
  

## Workflow
1. **Data Loading:** Load and inspect `Telco_customer_churn.xlsx`.
2. **Data Cleaning:** Drop irrelevant columns, handle missing values, and correct data types.
3. **Feature Encoding:** Encode binary and multi-category features.
4. **Train-Test Split:** Split the dataset with stratification to preserve class distribution.
5. **Feature Scaling:** Standardize numeric features for model training.
6. **Model Training:** Train Logistic Regression on scaled features.
7. **Model Evaluation:** Evaluate using accuracy, confusion matrix, classification report, and ROC-AUC score.
8. **Feature Importance:** Identify key drivers of churn for business insights.
9. **Baseline Comparison:** Compare performance with a DummyClassifier.
10. **Visualization:** Generate plots for feature importance and ROC curve.
11. **Business Recommendations:** Provide actionable strategies to reduce churn.
