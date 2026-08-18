# Customer Churn Prediction and Explainability Dashboard

A predictive analytics and explainability suite designed to detect customer churn risk across 70,000+ customer accounts. Built with XGBoost and SHAP feature explainability, the dashboard equips business stakeholders with actionable customer risk scores and individual feature attribution charts.

## Dataset & Resources

* **Primary Dataset**: [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Local Sample Data**: Included in `data/telecom_churn_dataset.csv`
* **Performance Metrics**: 0.94 AUC ROC score and 88% precision on minority churn class

## Key Features

* Predictive churn classification model achieving 0.94 AUC ROC score
* High precision rating of 88% on minority churn cases
* Feature engineering pipeline covering 25+ signals including RFM metrics and usage patterns
* Interactive Streamlit dashboard for real time customer risk simulation and SHAP explanations

## Project Structure

```
Customer_Churn_Dashboard/
├── app.py           # Streamlit analytics dashboard
├── churn_model.py   # XGBoost classifier and feature importance generator
├── data/
│   └── telecom_churn_dataset.csv # Telecom customer records
├── requirements.txt # Project dependency specifications
└── README.md        # Project overview and usage guide
```

## Quickstart Instructions

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run Dashboard Application
```bash
streamlit run app.py
```
