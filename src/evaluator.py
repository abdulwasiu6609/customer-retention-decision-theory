"""
Evaluation, Financial Decision Impact & Fairness Audit Engine
End-to-End Enterprise ML Architecture - Customer Churn & Retention Engine
Author: Abdul Wasiu Adebayo (BAN6440, Nexford University)
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

def compute_financial_impact_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    customer_ltv: float = 1800.0,
    intervention_cost: float = 120.0,
    offer_acceptance_rate: float = 0.45,
    thresholds: np.ndarray = np.linspace(0.05, 0.95, 37)
):
    """
    Simulates expected financial net benefit and ROI across classification decision thresholds.
    
    Financial parameters:
    - Customer Lifetime Value (LTV): $1,800
    - Retention Offer Cost (C_offer): $120 per targeted customer
    - Offer Acceptance Rate (P_accept): 45% of targeted churners accept offer and are retained
    
    Payoff Matrix per Customer:
    - True Positive (TP): Targeted & would have churned.
      Net Value = P_accept * LTV - C_offer = 0.45 * $1,800 - $120 = $810 - $120 = +$690
    - False Positive (FP): Targeted & would NOT have churned.
      Net Value = -C_offer = -$120 (unnecessary discount wasted)
    - False Negative (FN): Not targeted & churns.
      Net Value = -$0 direct intervention cost, but $0 value saved (opportunity loss)
    - True Negative (TN): Not targeted & stays.
      Net Value = $0 (organic retention, zero intervention cost)
    """
    results = []
    n_customers = len(y_true)
    
    for tau in thresholds:
        y_pred = (y_prob >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Financial payoffs
        net_saved_revenue = tp * (offer_acceptance_rate * customer_ltv)
        total_campaign_cost = (tp + fp) * intervention_cost
        net_profit = net_saved_revenue - total_campaign_cost
        
        profit_per_customer = net_profit / n_customers
        roi_pct = (net_profit / total_campaign_cost * 100.0) if total_campaign_cost > 0 else 0.0
        targeted_volume = tp + fp
        targeting_rate = targeted_volume / n_customers
        precision = tp / targeted_volume if targeted_volume > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        results.append({
            "threshold": round(tau, 3),
            "targeting_rate_pct": round(targeting_rate * 100, 2),
            "targeted_customers": int(targeted_volume),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "campaign_cost_usd": round(total_campaign_cost, 2),
            "gross_saved_revenue_usd": round(net_saved_revenue, 2),
            "net_profit_usd": round(net_profit, 2),
            "profit_per_customer_usd": round(profit_per_customer, 2),
            "campaign_roi_pct": round(roi_pct, 2)
        })
        
    df_curve = pd.DataFrame(results)
    optimal_row = df_curve.loc[df_curve['net_profit_usd'].idxmax()]
    return df_curve, optimal_row

def run_subgroup_fairness_audit(
    df_raw_test: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50
):
    """
    Audits algorithmic fairness across sensitive protected demographic features:
    - SeniorCitizen (0: Younger, 1: Senior)
    - gender (Male vs Female)
    - Partner (Yes vs No)
    
    Evaluates Disparate Impact Ratio (Selection Rate Parity) and Equal Opportunity (Recall / TPR Parity).
    """
    y_pred = (y_prob >= threshold).astype(int)
    audit_df = df_raw_test.copy().reset_index(drop=True)
    audit_df['y_true'] = y_true
    audit_df['y_pred'] = y_pred
    audit_df['y_prob'] = y_prob
    
    subgroups = {
        'SeniorCitizen': [0, 1],
        'gender': ['Male', 'Female'],
        'Partner': ['Yes', 'No']
    }
    
    records = []
    for attribute, values in subgroups.items():
        base_group = values[0]
        comp_group = values[1]
        
        sub_base = audit_df[audit_df[attribute] == base_group]
        sub_comp = audit_df[audit_df[attribute] == comp_group]
        
        sr_base = sub_base['y_pred'].mean()
        sr_comp = sub_comp['y_pred'].mean()
        disparate_impact = sr_comp / sr_base if sr_base > 0 else 1.0
        
        # True Positive Rate (Equal Opportunity)
        tp_base = sub_base[(sub_base['y_true'] == 1) & (sub_base['y_pred'] == 1)].shape[0]
        actual_pos_base = sub_base[sub_base['y_true'] == 1].shape[0]
        tpr_base = tp_base / actual_pos_base if actual_pos_base > 0 else 0.0
        
        tp_comp = sub_comp[(sub_comp['y_true'] == 1) & (sub_comp['y_pred'] == 1)].shape[0]
        actual_pos_comp = sub_comp[sub_comp['y_true'] == 1].shape[0]
        tpr_comp = tp_comp / actual_pos_comp if actual_pos_comp > 0 else 0.0
        
        tpr_ratio = tpr_comp / tpr_base if tpr_base > 0 else 1.0
        
        records.append({
            "demographic_attribute": attribute,
            "reference_group": str(base_group),
            "comparison_group": str(comp_group),
            "ref_sample_count": len(sub_base),
            "comp_sample_count": len(sub_comp),
            "ref_selection_rate": round(sr_base, 4),
            "comp_selection_rate": round(sr_comp, 4),
            "disparate_impact_ratio": round(disparate_impact, 4),
            "four_fifths_rule_status": "PASS" if 0.80 <= disparate_impact <= 1.25 else "DISPARATE_SELECTION",
            "ref_tpr_recall": round(tpr_base, 4),
            "comp_tpr_recall": round(tpr_comp, 4),
            "equal_opportunity_ratio": round(tpr_ratio, 4)
        })
        
    df_fairness = pd.DataFrame(records)
    return df_fairness

def export_tables(df_metrics: pd.DataFrame, df_cv: dict, df_fairness: pd.DataFrame,
                  df_profit_curve: pd.DataFrame, output_dir: str):
    """Exports structured evaluation CSV tables for formal submission package."""
    os.makedirs(output_dir, exist_ok=True)
    
    df_metrics.to_csv(os.path.join(output_dir, "model_comparison_holdout_metrics.csv"), index=False)
    
    df_cv_formatted = pd.DataFrame(df_cv).T.reset_index().rename(columns={'index': 'model_name'})
    df_cv_formatted.to_csv(os.path.join(output_dir, "model_cross_validation_results.csv"), index=False)
    
    df_fairness.to_csv(os.path.join(output_dir, "demographic_fairness_audit.csv"), index=False)
    df_profit_curve.to_csv(os.path.join(output_dir, "financial_threshold_profit_curve.csv"), index=False)
    
    print(f"Exported evaluation tables to {output_dir}")
