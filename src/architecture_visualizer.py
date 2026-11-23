"""
Architecture Visualizer & Diagnostic Plotting Engine
End-to-End Enterprise ML Architecture - Customer Churn & Retention Engine
Author: Abdul Wasiu Adebayo (BAN6440, Nexford University)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

# Configure clean aesthetic
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

def plot_end_to_end_architecture(output_path: str):
    """
    Renders production-grade multi-cloud interoperability pipeline diagram:
    Google BigQuery -> Databricks/Apache Spark -> MLflow Registry -> Azure Container Apps -> OpenAI GPT-4o
    """
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Color palette
    c_ingest = '#E0F2FE'  # BigQuery sky
    c_prep = '#EDE9FE'    # Databricks purple
    c_track = '#FEF3C7'   # MLflow amber
    c_serve = '#DCFCE7'   # Azure green
    c_genai = '#FFE4E6'   # OpenAI rose
    c_monitor = '#F1F5F9' # Governance gray
    
    # Title & Subtitle
    ax.text(8.0, 8.55, "Enterprise End-to-End ML Architecture & Multi-Cloud Interoperability Pipeline",
            fontsize=18, fontweight='bold', ha='center', color='#0F172A')
    ax.text(8.0, 8.25, "Production Interoperability: Google BigQuery → Databricks Delta Lake → MLflow Registry → Azure Container Apps → OpenAI Decision Layer",
            fontsize=11, fontstyle='italic', ha='center', color='#475569')

    # Layer Boxes
    boxes = [
        # (x, y, w, h, bg_color, border_color, title, subtitle, bullets)
        (0.6, 2.0, 2.6, 5.8, c_ingest, '#0284C7', "1. INGESTION & STORAGE\n(Google BigQuery)",
         "Enterprise Data Lakehouse",
         ["• Telemetry & Clickstream", "• Billing & Stripe Events", "• CRM Customer Master", "• Scheduled Partitioning", "• BigQuery Analytical SQL"]),
         
        (3.6, 2.0, 2.6, 5.8, c_prep, '#7C3AED', "2. DATA & TRAINING\n(Databricks / Spark)",
         "Unified Delta Lake Engine",
         ["• Apache Spark ETL Jobs", "• Feature Store Staging", "• Leak-Free Scaling/Encoding", "• 5-Fold Stratified CV", "• Gradient Boost & DNNs"]),
         
        (6.6, 2.0, 2.6, 5.8, c_track, '#D97706', "3. GOVERNANCE & MLOps\n(MLflow Model Registry)",
         "Lifecycle Management",
         ["• SQLite/Postgres Store", "• Metric & Param Logging", "• Artifact Versioning", "• Brier/ROC Validation", "• Staging → Prod Promotion"]),
         
        (9.6, 2.0, 2.6, 5.8, c_serve, '#16A34A', "4. SERVING & SCORING\n(Azure Container Apps)",
         "Low-Latency REST Microservice",
         ["• Dockerized FastAPI App", "• P99 Latency < 65ms", "• Dynamic Threshold (0.42)", "• Real-Time Risk Routing", "• Autoscaling K8s Pods"]),
         
        (12.6, 2.0, 2.8, 5.8, c_genai, '#E11D48', "5. DECISION & RETENTION\n(OpenAI GPT-4o mini)",
         "Agentic Action Engine",
         ["• SHAP Risk Factor Ingestion", "• $120 Offer Budget Cap", "• Automated CSM Battlecard", "• Personalized Email Copy", "• CRM Webhook Dispatch"])
    ]

    for (bx, by, bw, bh, bg_col, br_col, title, subtitle, bullets) in boxes:
        # Draw box
        rect = patches.FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.1,rounding_size=0.15",
                                      facecolor=bg_col, edgecolor=br_col, linewidth=2, zorder=2)
        ax.add_patch(rect)
        
        # Header box
        header_rect = patches.FancyBboxPatch((bx, by + bh - 1.25), bw, 1.25, boxstyle="round,pad=0.08,rounding_size=0.1",
                                             facecolor=br_col, edgecolor=br_col, zorder=3)
        ax.add_patch(header_rect)
        ax.text(bx + bw/2.0, by + bh - 0.65, title, fontsize=10.5, fontweight='bold', ha='center', va='center', color='#FFFFFF', zorder=4)
        
        # Subtitle
        ax.text(bx + bw/2.0, by + bh - 1.55, subtitle, fontsize=9.5, fontweight='bold', ha='center', color=br_col, zorder=4)
        
        # Bullet items
        for idx, item in enumerate(bullets):
            ax.text(bx + 0.18, by + bh - 2.15 - (idx * 0.70), item, fontsize=8.8, color='#1E293B', zorder=4)

    # Interoperability Arrows
    arrow_props = dict(boxstyle="rarrow,pad=0.15", fc="#0284C7", ec="#0369A1", lw=1.5)
    arrows_x = [3.22, 6.22, 9.22, 12.22]
    interop_labels = [
        "BigQuery Connector\n(Delta Lake Ingest)",
        "MLflow Tracking\n(Artifact Push)",
        "CI/CD Container\n(Image Deployment)",
        "REST JSON Hook\n(Risk Payload)"
    ]
    
    for ax_pos, lbl in zip(arrows_x, interop_labels):
        ax.annotate('', xy=(ax_pos + 0.35, 4.9), xytext=(ax_pos - 0.05, 4.9),
                    arrowprops=dict(facecolor='#0F172A', edgecolor='#0F172A', width=2.5, headwidth=8))
        ax.text(ax_pos + 0.15, 5.35, lbl, fontsize=7.2, fontweight='bold', ha='center', color='#0F172A',
                bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=0.8))

    # Bottom Monitoring & Drift Bar
    bot_rect = patches.FancyBboxPatch((0.6, 0.4), 14.8, 1.1, boxstyle="round,pad=0.1,rounding_size=0.1",
                                     facecolor=c_monitor, edgecolor='#64748B', linewidth=1.5, zorder=2)
    ax.add_patch(bot_rect)
    ax.text(1.0, 0.95, "CONTINUOUS OBSERVABILITY & GOVERNANCE LOOP:", fontsize=9.5, fontweight='bold', color='#0F172A')
    ax.text(1.0, 0.65, "Prometheus & Grafana Telemetry | Automated PSI & KS-Test Drift Detection | Weekly Retraining Trigger | Disparate Impact Fairness Gate",
            fontsize=8.5, color='#334155')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved architecture diagram to: {output_path}")

def plot_model_performance(test_predictions: dict, y_test: np.ndarray, output_path: str):
    """
    Renders 3-panel comparative diagnostic figure:
    1. Receiver Operating Characteristic (ROC) curves
    2. Precision-Recall (PR) curves
    3. Probability Calibration curves (Reliability Diagram)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    
    colors = {
        "Baseline_Logistic_Regression": "#64748B",
        "Random_Forest_Ensemble": "#0284C7",
        "Gradient_Boosting_Classifier": "#16A34A",
        "Deep_Neural_Network_MLP": "#D97706"
    }

    # 1. ROC Curve
    ax_roc = axes[0]
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Guess (AUC=0.50)')
    for name, data in test_predictions.items():
        if "Calibrated" not in name:
            fpr, tpr, _ = roc_curve(y_test, data['y_prob'])
            auc_val = np.trapz(tpr, fpr)
            col = colors.get(name, '#333333')
            ax_roc.plot(fpr, tpr, label=f"{name.replace('_', ' ')} ({auc_val:.3f})", color=col, lw=2)
            
    ax_roc.set_title("A. Receiver Operating Characteristic (ROC)", fontsize=12, fontweight='bold', pad=10)
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10.5)
    ax_roc.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10.5)
    ax_roc.grid(True, linestyle=':', alpha=0.6)
    ax_roc.legend(fontsize=8.5, loc='lower right')

    # 2. Precision-Recall Curve
    ax_pr = axes[1]
    base_rate = float(np.mean(y_test))
    ax_pr.plot([0, 1], [base_rate, base_rate], 'k--', lw=1.2, label=f'Baseline Class Rate ({base_rate:.2f})')
    for name, data in test_predictions.items():
        if "Calibrated" not in name:
            prec, rec, _ = precision_recall_curve(y_test, data['y_prob'])
            pr_auc_val = -np.trapz(prec, rec)
            col = colors.get(name, '#333333')
            ax_pr.plot(rec, prec, label=f"{name.replace('_', ' ')} (PR-AUC={pr_auc_val:.3f})", color=col, lw=2)
            
    ax_pr.set_title("B. Precision-Recall Curves", fontsize=12, fontweight='bold', pad=10)
    ax_pr.set_xlabel("Recall (Coverage)", fontsize=10.5)
    ax_pr.set_ylabel("Precision (Positive Predictive Value)", fontsize=10.5)
    ax_pr.grid(True, linestyle=':', alpha=0.6)
    ax_pr.legend(fontsize=8.5, loc='upper right')

    # 3. Calibration Curve
    ax_cal = axes[2]
    ax_cal.plot([0, 1], [0, 1], 'k:', lw=1.2, label='Perfect Calibration')
    for name, data in test_predictions.items():
        if "Calibrated" not in name:
            prob_true, prob_pred = calibration_curve(y_test, data['y_prob'], n_bins=10)
            col = colors.get(name, '#333333')
            ax_cal.plot(prob_pred, prob_true, marker='o', markersize=4, label=f"{name.replace('_', ' ')}", color=col, lw=1.8)
            
    ax_cal.set_title("C. Probability Calibration (Reliability Diagram)", fontsize=12, fontweight='bold', pad=10)
    ax_cal.set_xlabel("Mean Predicted Churn Probability", fontsize=10.5)
    ax_cal.set_ylabel("Fraction of True Churners", fontsize=10.5)
    ax_cal.grid(True, linestyle=':', alpha=0.6)
    ax_cal.legend(fontsize=8.5, loc='upper left')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved performance diagnostic curves to: {output_path}")

def plot_financial_profit_curve(df_curve: pd.DataFrame, optimal_row: pd.Series, output_path: str):
    """
    Renders 2-panel business decision impact plot:
    1. Net Campaign Profit ($) and Campaign Cost ($) vs Decision Threshold
    2. Precision and Recall vs Decision Threshold with optimal operating point
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
    
    # 1. Financial Net Profit Curve
    ax_fin = axes[0]
    ax_fin.plot(df_curve['threshold'], df_curve['net_profit_usd'], color='#0F766E', lw=2.5, label='Net Campaign Profit ($)')
    ax_fin.plot(df_curve['threshold'], df_curve['campaign_cost_usd'], color='#E11D48', lw=1.8, linestyle='--', label='Campaign Cost ($)')
    
    # Highlight optimum
    opt_tau = optimal_row['threshold']
    opt_profit = optimal_row['net_profit_usd']
    ax_fin.axvline(opt_tau, color='#0284C7', linestyle=':', lw=1.5, label=f'Optimal Threshold (tau={opt_tau:.2f})')
    ax_fin.scatter([opt_tau], [opt_profit], color='#0284C7', s=80, zorder=5)
    ax_fin.annotate(f"Max Net Profit: ${opt_profit:,.0f}\n(ROI: {optimal_row['campaign_roi_pct']:.1f}%)",
                    xy=(opt_tau, opt_profit), xytext=(opt_tau + 0.08, opt_profit - 15000),
                    arrowprops=dict(facecolor='#0284C7', arrowstyle='->', lw=1.5),
                    fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F9FF', edgecolor='#0284C7'))
    
    ax_fin.set_title("A. Campaign Profitability vs. Classification Threshold", fontsize=12, fontweight='bold', pad=10)
    ax_fin.set_xlabel("Decision Threshold (tau)", fontsize=10.5)
    ax_fin.set_ylabel("USD ($)", fontsize=10.5)
    ax_fin.yaxis.set_major_formatter('${x:,.0f}')
    ax_fin.grid(True, linestyle=':', alpha=0.6)
    ax_fin.legend(fontsize=9, loc='upper right')

    # 2. Operating Precision & Recall Curve
    ax_pr = axes[1]
    ax_pr.plot(df_curve['threshold'], df_curve['precision'], color='#2563EB', lw=2.2, label='Precision (Targeting Accuracy)')
    ax_pr.plot(df_curve['threshold'], df_curve['recall'], color='#D97706', lw=2.2, label='Recall (Churner Coverage)')
    ax_pr.plot(df_curve['threshold'], df_curve['targeting_rate_pct'] / 100.0, color='#64748B', lw=1.5, linestyle=':', label='Cohort Targeting Rate')
    
    ax_pr.axvline(opt_tau, color='#0284C7', linestyle=':', lw=1.5, label=f'Optimal tau={opt_tau:.2f}')
    ax_pr.set_title("B. Precision, Recall & Targeting Volume Trade-Off", fontsize=12, fontweight='bold', pad=10)
    ax_pr.set_xlabel("Decision Threshold (tau)", fontsize=10.5)
    ax_pr.set_ylabel("Rate / Ratio", fontsize=10.5)
    ax_pr.grid(True, linestyle=':', alpha=0.6)
    ax_pr.legend(fontsize=9, loc='lower left')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved financial decision plot to: {output_path}")

def plot_feature_importance_drivers(model, feature_names: list, output_path: str, top_n: int = 12):
    """
    Renders top predictive feature importance drivers for churn risk attribution.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("Model does not expose feature_importances_ or coef_")
        return
        
    df_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    df_imp = df_imp.sort_values(by='importance', ascending=False).head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.barh(df_imp['feature'], df_imp['importance'], color='#0284C7', edgecolor='#0369A1', height=0.65)
    
    ax.set_title(f"Top {top_n} Churn Risk Predictive Drivers (Feature Importance Attribution)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Relative Feature Importance Weight", fontsize=10.5)
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    
    # Annotate values
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height()/2.0, f"{w:.3f}", va='center', fontsize=8.5, color='#1E293B')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance plot to: {output_path}")

def plot_data_quality_gates(df_clean: pd.DataFrame, output_path: str):
    """
    Renders data quality and pre/post validation distributions:
    Tenure distribution, MonthlyCharges, TotalCharges, and Class Balance.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    
    # 1. Tenure
    axes[0, 0].hist(df_clean['tenure'], bins=30, color='#38BDF8', edgecolor='#0284C7')
    axes[0, 0].set_title("A. Customer Account Tenure (Months)", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel("Tenure (Months)")
    axes[0, 0].set_ylabel("Customer Count")
    axes[0, 0].grid(True, linestyle=':', alpha=0.5)
    
    # 2. Monthly Charges
    axes[0, 1].hist(df_clean['MonthlyCharges'], bins=30, color='#818CF8', edgecolor='#4F46E5')
    axes[0, 1].set_title("B. Monthly Recurring Charges (USD)", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel("Monthly Charges ($)")
    axes[0, 1].grid(True, linestyle=':', alpha=0.5)
    
    # 3. Total Charges (post sentinel cleaning)
    axes[1, 0].hist(df_clean['TotalCharges'], bins=30, color='#34D399', edgecolor='#059669')
    axes[1, 0].set_title("C. Total Accumulated Spend (USD)", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel("Total Charges ($)")
    axes[1, 0].set_ylabel("Customer Count")
    axes[1, 0].grid(True, linestyle=':', alpha=0.5)
    
    # 4. Target Class Balance
    churn_counts = df_clean['Churn'].value_counts()
    axes[1, 1].bar(['Retained (0)', 'Churned (1)'], churn_counts.values, color=['#10B981', '#EF4444'], width=0.5)
    axes[1, 1].set_title("D. Target Class Balance (26.5% Churn Rate)", fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel("Customer Count")
    for i, v in enumerate(churn_counts.values):
        axes[1, 1].text(i, v + 80, f"{v:,} ({v/len(df_clean)*100:.1f}%)", ha='center', fontweight='bold', fontsize=9.5)
    axes[1, 1].grid(axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved data quality distribution plot to: {output_path}")
