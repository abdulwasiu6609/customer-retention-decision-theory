"""
Machine Learning Model Pipeline & Training Protocol
End-to-End Enterprise ML Architecture - Customer Churn & Retention Engine
Author: Abdul Wasiu Adebayo (BAN6440, Nexford University)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss
)

def build_model_suite(random_state: int = 42):
    """
    Initializes model candidates spanning statistical baseline,
    tree ensembles, gradient boosting, and deep neural architectures.
    """
    models = {
        "Baseline_Logistic_Regression": LogisticRegression(
            penalty='l2', C=1.0, class_weight='balanced', max_iter=1000, random_state=random_state
        ),
        "Random_Forest_Ensemble": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=10, min_samples_leaf=4,
            class_weight='balanced', random_state=random_state, n_jobs=-1
        ),
        "Gradient_Boosting_Classifier": HistGradientBoostingClassifier(
            max_iter=150, max_depth=6, learning_rate=0.08, min_samples_leaf=20,
            class_weight='balanced', random_state=random_state
        ),
        "Deep_Neural_Network_MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=0.001,
            batch_size=64, learning_rate_init=0.001, max_iter=200, early_stopping=True,
            random_state=random_state
        )
    }
    return models

def run_cross_validation(models: dict, X_train: np.ndarray, y_train: np.ndarray, n_splits: int = 5):
    """
    Executes 5-fold Stratified Cross-Validation on training data to assess generalization stability.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'pr_auc': 'average_precision'
    }
    
    cv_results = {}
    for name, model in models.items():
        print(f"Running {n_splits}-fold CV for {name}...")
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_results[name] = {
            "cv_accuracy_mean": float(np.mean(scores['test_accuracy'])),
            "cv_accuracy_std": float(np.std(scores['test_accuracy'])),
            "cv_recall_mean": float(np.mean(scores['test_recall'])),
            "cv_recall_std": float(np.std(scores['test_recall'])),
            "cv_precision_mean": float(np.mean(scores['test_precision'])),
            "cv_precision_std": float(np.std(scores['test_precision'])),
            "cv_f1_mean": float(np.mean(scores['test_f1'])),
            "cv_f1_std": float(np.std(scores['test_f1'])),
            "cv_roc_auc_mean": float(np.mean(scores['test_roc_auc'])),
            "cv_roc_auc_std": float(np.std(scores['test_roc_auc'])),
            "cv_pr_auc_mean": float(np.mean(scores['test_pr_auc'])),
            "cv_pr_auc_std": float(np.std(scores['test_pr_auc']))
        }
    return cv_results

def train_and_calibrate_models(models: dict, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray):
    """
    Fits models on training set and applies Platt scaling probability calibration.
    """
    trained_models = {}
    calibrated_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Calibrate probabilities using sigmoid / Platt scaling via CV
        calibrator = CalibratedClassifierCV(estimator=model, cv=3, method='sigmoid')
        calibrator.fit(X_train, y_train)
        calibrated_models[f"{name}_Calibrated"] = calibrator
        
    return trained_models, calibrated_models

def evaluate_on_holdout(models_dict: dict, X_test: np.ndarray, y_test: np.ndarray):
    """
    Evaluates raw and calibrated models on the quarantined holdout test set (N=1,057).
    """
    records = []
    test_predictions = {}
    
    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)
        
        # Compute Top-Decile Lift (Top 10% and Top 20% predicted risk)
        df_eval = pd.DataFrame({'y_true': y_test, 'y_prob': y_prob})
        df_eval = df_eval.sort_values(by='y_prob', ascending=False).reset_index(drop=True)
        
        base_rate = float(np.mean(y_test))
        top_10_pct_idx = int(0.10 * len(df_eval))
        top_20_pct_idx = int(0.20 * len(df_eval))
        
        precision_at_10 = float(df_eval.iloc[:top_10_pct_idx]['y_true'].mean())
        precision_at_20 = float(df_eval.iloc[:top_20_pct_idx]['y_true'].mean())
        
        lift_10 = precision_at_10 / base_rate if base_rate > 0 else 1.0
        lift_20 = precision_at_20 / base_rate if base_rate > 0 else 1.0
        
        rec = {
            "model_name": name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "pr_auc": float(average_precision_score(y_test, y_prob)),
            "brier_score": float(brier_score_loss(y_test, y_prob)),
            "precision_at_top10": precision_at_10,
            "lift_at_top10": lift_10,
            "precision_at_top20": precision_at_20,
            "lift_at_top20": lift_20
        }
        records.append(rec)
        test_predictions[name] = {
            "y_prob": y_prob,
            "y_pred": y_pred
        }
        
    df_metrics = pd.DataFrame(records)
    return df_metrics, test_predictions

def save_artifacts(models_dict: dict, output_dir: str):
    """Saves serialized model artifacts for deployment readiness."""
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models_dict.items():
        model_file = os.path.join(output_dir, f"{name}.joblib")
        joblib.dump(model, model_file)
    print(f"Serialized {len(models_dict)} model objects into {output_dir}")
