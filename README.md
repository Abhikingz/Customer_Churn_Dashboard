# Customer Churn Prediction and Explainability Dashboard

A predictive analytics and explainability suite designed to detect customer churn risk across 70,000+ customer accounts. Built with XGBoost and SHAP feature explainability, the dashboard equips business stakeholders with actionable customer risk scores and individual feature attribution charts.

## Project Documentation & Technical Report

* **Download Technical PDF Report**: [Technical_Report_Customer_Churn_Dashboard.pdf](Technical_Report_Customer_Churn_Dashboard.pdf)
* **Primary Dataset**: [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Local Sample Data**: Included in `data/telecom_churn_dataset.csv`

## Key Features

* Predictive churn classification model achieving 0.94 AUC ROC score
* High precision rating of 88% on minority churn cases
* Feature engineering pipeline covering 25+ signals including RFM metrics and usage patterns
* Interactive Streamlit dashboard for real time customer risk simulation and SHAP explanations

## Quickstart Guide

```bash
pip install -r requirements.txt
streamlit run app.py
```
