import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

def train_churn_model():
    """
    Trains gradient boosted tree model on customer behavioral dataset
    """
    np.random.seed(42)
    n_samples = 1000
    
    tenure = np.random.randint(1, 72, n_samples)
    monthly_charges = np.random.uniform(20.0, 120.0, n_samples)
    support_tickets = np.random.randint(0, 10, n_samples)
    payment_delays = np.random.randint(0, 5, n_samples)
    
    # Target churn probability synthesis
    score = (120 - monthly_charges)*0.01 - tenure*0.04 + support_tickets*0.35 + payment_delays*0.45
    prob = 1 / (1 + np.exp(-score))
    churn = (prob > 0.45).astype(int)
    
    X = pd.DataFrame({
        'tenure_months': tenure,
        'monthly_charges': monthly_charges,
        'support_tickets': support_tickets,
        'payment_delays': payment_delays
    })
    
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X, churn)
    return model, X, churn

def get_feature_importance(model, feature_names):
    importances = model.feature_importances_
    return pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
