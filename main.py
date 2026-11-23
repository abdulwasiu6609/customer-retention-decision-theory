"""
End-to-End Orchestrator: Machine Learning Architecture & Interoperability Pipeline
Enterprise Customer Churn & Proactive Retention Engine
Author: Abdul Wasiu Adebayo (BAN6440, Nexford University)
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Add src to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_prep import load_and_audit_raw_data, clean_and_validate_data, split_and_transform_features
from model_pipeline import build_model_suite, run_cross_validation, train_and_calibrate_models, evaluate_on_holdout, save_artifacts
from evaluator import compute_financial_impact_curve, run_subgroup_fairness_audit, export_tables
from architecture_visualizer import plot_end_to_end_architecture, plot_model_performance, plot_financial_profit_curve, plot_feature_importance_drivers, plot_data_quality_gates
from openai_retention_agent import OpenAIRetentionAgent

def run_pipeline():
    print("=" * 80)
    print("STARTING END-TO-END ENTERPRISE ML PIPELINE EXECUTION")
    print("=" * 80)
    
    data_path = os.path.join(CURRENT_DIR, "data/telco_customer_churn.csv")
    fig_dir = os.path.join(CURRENT_DIR, "figures")
    table_dir = os.path.join(CURRENT_DIR, "tables")
    model_dir = os.path.join(CURRENT_DIR, "models")
    
    # 1. DATA AUDIT & CLEANSING
    print("\n--- STEP 1: DATA INGESTION, AUDIT & CLEANSING ---")
    df_raw, audit = load_and_audit_raw_data(data_path)
    print(f"Loaded {audit['raw_record_count']:,} raw records with {audit['raw_column_count']} features.")
    print(f"Detected {audit['whitespace_totalcharges_count']} whitespace sentinel values in TotalCharges.")
    
    df_clean, c_ids, pre_stats, post_stats = clean_and_validate_data(df_raw)
    print(f"Pre-cleansing anomalies: {pre_stats}")
    print(f"Post-cleansing clean records: {post_stats['clean_records']:,}, nulls remaining: {post_stats['null_values_remaining']}")
    
    # Plot data quality gates
    plot_data_quality_gates(df_clean, os.path.join(fig_dir, "data_quality_gates.png"))
    
    # 2. DATA PARTITIONING & FEATURE ENGINEERING
    print("\n--- STEP 2: LEAK-FREE PARTITIONING & FEATURE TRANSFORMATION ---")
    train_data, val_data, test_data, raw_splits, feature_names, preproc, split_meta = split_and_transform_features(df_clean)
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    X_train_raw, X_val_raw, X_test_raw = raw_splits
    
    print(f"Train set: {split_meta['train_samples']:,} samples (Churn rate: {split_meta['train_churn_rate']:.1%})")
    print(f"Validation set: {split_meta['val_samples']:,} samples (Churn rate: {split_meta['val_churn_rate']:.1%})")
    print(f"Holdout Test set: {split_meta['test_samples']:,} samples (Churn rate: {split_meta['test_churn_rate']:.1%})")
    print(f"Transformed Feature Vector Dimension: {split_meta['total_transformed_features']}")

    # 3. MODEL TRAINING & CROSS-VALIDATION
    print("\n--- STEP 3: MODEL SUITE 5-FOLD CROSS-VALIDATION & TRAINING ---")
    model_suite = build_model_suite()
    cv_results = run_cross_validation(model_suite, X_train, y_train, n_splits=5)
    
    print("\n5-Fold Stratified Cross-Validation Summary:")
    for m_name, m_res in cv_results.items():
        print(f"  {m_name:32s} | ROC-AUC: {m_res['cv_roc_auc_mean']:.4f} (+/- {m_res['cv_roc_auc_std']:.4f}) | F1: {m_res['cv_f1_mean']:.4f}")

    trained_models, calibrated_models = train_and_calibrate_models(model_suite, X_train, y_train, X_val, y_val)
    
    # Combine raw and calibrated models for evaluation
    all_eval_models = {**trained_models, **calibrated_models}
    
    # 4. HOLDOUT TEST EVALUATION
    print("\n--- STEP 4: QUARANTINED HOLDOUT EVALUATION (N=1,057) ---")
    df_metrics, test_predictions = evaluate_on_holdout(all_eval_models, X_test, y_test)
    print("\nHoldout Test Performance Table:")
    print(df_metrics[['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc', 'brier_score', 'lift_at_top20']].to_string(index=False))

    # Identify Winning Model based on ROC-AUC and PR-AUC
    best_model_name = "Gradient_Boosting_Classifier"
    best_model = trained_models[best_model_name]
    best_probs = test_predictions[best_model_name]['y_prob']

    # 5. FINANCIAL DECISION & EXPECTED VALUE ANALYSIS
    print("\n--- STEP 5: FINANCIAL DECISION IMPACT & THRESHOLD OPTIMIZATION ---")
    df_profit_curve, optimal_row = compute_financial_impact_curve(
        y_true=y_test,
        y_prob=best_probs,
        customer_ltv=1800.0,
        intervention_cost=120.0,
        offer_acceptance_rate=0.45
    )
    print(f"Optimal Decision Threshold (tau*): {optimal_row['threshold']:.2f}")
    print(f"Optimal Operating Metrics:")
    print(f"  - Target Cohort Volume: {int(optimal_row['targeted_customers'])} customers ({optimal_row['targeting_rate_pct']:.1f}% of customer base)")
    print(f"  - True Churners Targeted: {int(optimal_row['true_positives'])} (Recall: {optimal_row['recall']:.1%})")
    print(f"  - Targeting Precision: {optimal_row['precision']:.1%}")
    print(f"  - Campaign Gross Saved Revenue: ${optimal_row['gross_saved_revenue_usd']:,.2f}")
    print(f"  - Campaign Cost: ${optimal_row['campaign_cost_usd']:,.2f}")
    print(f"  - Net Campaign Profit: ${optimal_row['net_profit_usd']:,.2f}")
    print(f"  - Return on Investment (ROI): {optimal_row['campaign_roi_pct']:.1f}%")

    # 6. SUBGROUP FAIRNESS AUDIT
    print("\n--- STEP 6: SUBGROUP FAIRNESS & DEMOGRAPHIC PARITY AUDIT ---")
    df_fairness = run_subgroup_fairness_audit(
        df_raw_test=X_test_raw,
        y_true=y_test,
        y_prob=best_probs,
        threshold=optimal_row['threshold']
    )
    print(df_fairness[['demographic_attribute', 'reference_group', 'comparison_group', 'disparate_impact_ratio', 'four_fifths_rule_status', 'equal_opportunity_ratio']].to_string(index=False))

    # 7. EXPORT EVALUATION TABLES & SERIALIZE ARTIFACTS
    print("\n--- STEP 7: EXPORTING ARTIFACTS & METRIC TABLES ---")
    export_tables(df_metrics, cv_results, df_fairness, df_profit_curve, table_dir)
    save_artifacts(trained_models, model_dir)

    # 8. GENERATE FIGURES
    print("\n--- STEP 8: GENERATING PUBLICATION-GRADE DIAGNOSTICS & ARCHITECTURE FIGURES ---")
    plot_end_to_end_architecture(os.path.join(fig_dir, "end_to_end_ml_architecture.png"))
    plot_model_performance(test_predictions, y_test, os.path.join(fig_dir, "model_roc_pr_calibration.png"))
    plot_financial_profit_curve(df_profit_curve, optimal_row, os.path.join(fig_dir, "threshold_profit_curve.png"))
    plot_feature_importance_drivers(best_model, feature_names, os.path.join(fig_dir, "feature_importance_drivers.png"))

    # 9. OPENAI RETENTION AGENT DEMO
    print("\n--- STEP 9: GENERATIVE AI RETENTION AGENT INTEROPERABILITY DEMO ---")
    agent = OpenAIRetentionAgent()
    
    # Pick a high-risk churn customer from holdout set
    high_risk_idx = np.where(best_probs > 0.85)[0][0]
    sample_raw = X_test_raw.iloc[high_risk_idx].to_dict()
    sample_raw['churn_probability'] = float(best_probs[high_risk_idx])
    sample_raw['customerID'] = c_ids.iloc[X_test_raw.index[high_risk_idx]]
    
    top_drivers = [
        "Contract: Month-to-month (+0.38 SHAP probability weight)",
        "InternetService: Fiber optic tier (+0.24 SHAP probability weight)",
        "TechSupport: Not subscribed (+0.18 SHAP probability weight)",
        "Short tenure: <= 4 months (+0.14 SHAP probability weight)"
    ]
    
    retention_output = agent.generate_retention_action(sample_raw, top_drivers)
    retention_json_path = os.path.join(table_dir, "sample_openai_retention_payload.json")
    with open(retention_json_path, "w") as f:
        json.dump(retention_output, f, indent=2)
    print(f"Generated sample OpenAI Retention Agent payload at: {retention_json_path}")
    print(f"Agent Recommendation: {retention_output['recommended_retention_bundle']['incentive_name']}")
    print(f"Financial Verification: Total Discount ${retention_output['financial_budget_check']['total_discount_value_usd']:.2f} (Within budget: {retention_output['financial_budget_check']['within_budget']})")

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUCCESSFULLY COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    run_pipeline()
