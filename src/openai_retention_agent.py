"""
OpenAI Generative AI Retention Agent & Decision Layer
End-to-End Enterprise ML Architecture - Customer Churn & Retention Engine
Author: Abdul Wasiu Adebayo (BAN6440, Nexford University)
"""

import os
import json
from typing import Dict, Any, List

class OpenAIRetentionAgent:
    """
    Production-grade agentic retention generator that interoperates with
    downstream ML model outputs (predictive churn risk scores + SHAP/feature attributions)
    to generate contextualized, financially bounded customer retention offers.
    """
    
    def __init__(self, api_key: str = None, model_name: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.is_live = bool(self.api_key)
        
    def build_system_prompt(self) -> str:
        return (
            "You are the Chief Customer Retention Specialist and Pricing Optimization Agent at an enterprise SaaS/Telecommunications provider. "
            "You receive high-risk customer profiles scored by an enterprise Machine Learning Churn Model along with key risk drivers. "
            "Your objective is to synthesize these signals into a mathematically sound, highly personalized retention proposal. "
            "Constraints: The total retention offer value must not exceed the $120 campaign intervention budget, and the proposal must directly address the identified churn drivers."
        )
        
    def format_customer_context(self, customer_profile: Dict[str, Any], top_risk_factors: List[str]) -> str:
        prompt = (
            f"Customer Profile Telemetry:\n"
            f"- Account Tenure: {customer_profile.get('tenure', 0)} months\n"
            f"- Contract Type: {customer_profile.get('Contract', 'Month-to-month')}\n"
            f"- Monthly Recurring Charges: ${customer_profile.get('MonthlyCharges', 0.0):.2f}\n"
            f"- Total Historical Spend: ${customer_profile.get('TotalCharges', 0.0):.2f}\n"
            f"- Internet Service: {customer_profile.get('InternetService', 'None')}\n"
            f"- Tech Support Enrolled: {customer_profile.get('TechSupport', 'No')}\n"
            f"- Online Security Enrolled: {customer_profile.get('OnlineSecurity', 'No')}\n"
            f"- Payment Rail: {customer_profile.get('PaymentMethod', 'Unknown')}\n"
            f"- Model Predicted Churn Probability: {customer_profile.get('churn_probability', 0.85):.1%}\n\n"
            f"Top Predictive Churn Drivers Identified by Model:\n"
        )
        for i, factor in enumerate(top_risk_factors, 1):
            prompt += f"{i}. {factor}\n"
            
        prompt += (
            "\nRequired Output Structure (Return strictly valid JSON):\n"
            "{\n"
            '  "churn_risk_level": "High" | "Critical",\n'
            '  "root_cause_diagnosis": "Concise analytical summary of why customer is leaving",\n'
            '  "recommended_retention_bundle": {\n'
            '    "incentive_name": "Name of offer",\n'
            '    "monthly_discount_usd": 15.00,\n'
            '    "duration_months": 6,\n'
            '    "service_add_on": "Complimentary service addition",\n'
            '    "contract_commitment_required": "1-Year Plan"\n'
            '  },\n'
            '  "financial_budget_check": {\n'
            '    "total_discount_value_usd": 90.00,\n'
            '    "budget_cap_usd": 120.00,\n'
            '    "within_budget": true\n'
            '  },\n'
            '  "outbound_personalized_email_body": "Polished executive customer email",\n'
            '  "csm_call_battlecard_talking_points": ["Point 1", "Point 2", "Point 3"]\n'
            "}"
        )
        return prompt

    def generate_retention_action(self, customer_profile: Dict[str, Any], top_risk_factors: List[str]) -> Dict[str, Any]:
        """
        Executes retention intervention generation. If OpenAI API key is configured,
        queries gpt-4o-mini; otherwise provides deterministic, domain-engineered response.
        """
        if self.is_live:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model=self.model_name,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": self.build_system_prompt()},
                        {"role": "user", "content": self.format_customer_context(customer_profile, top_risk_factors)}
                    ],
                    temperature=0.2
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                print(f"OpenAI API live query notice: {e}. Falling back to deterministic pipeline generator.")
                
        # Deterministic domain synthesis engine (guarantees offline reproducibility)
        tenure = customer_profile.get('tenure', 2)
        monthly_charge = customer_profile.get('MonthlyCharges', 85.0)
        contract = customer_profile.get('Contract', 'Month-to-month')
        
        # Calculate dynamic offer bounded by $120 budget
        monthly_disc = 15.0 if monthly_charge > 75.0 else 10.0
        duration_mo = 6
        total_val = monthly_disc * duration_mo
        
        add_on = "Complimentary Premium Tech Support & Cybersecurity Suite (6 months)" if customer_profile.get('TechSupport') == 'No' else "Free Cloud Backup Storage Tier"
        
        return {
            "customer_id": customer_profile.get("customerID", "CUST-SIMULATED"),
            "predicted_churn_prob": customer_profile.get('churn_probability', 0.88),
            "churn_risk_level": "Critical" if customer_profile.get('churn_probability', 0.88) >= 0.80 else "High",
            "root_cause_diagnosis": (
                f"Customer exhibits extreme churn susceptibility driven by month-to-month contract fragility "
                f"at {tenure} months tenure combined with elevated monthly fee (${monthly_charge:.2f}/mo) and unaddressed technical service gaps."
            ),
            "recommended_retention_bundle": {
                "incentive_name": "Loyalty Continuity & Tech Shield Package",
                "monthly_discount_usd": monthly_disc,
                "duration_months": duration_mo,
                "service_add_on": add_on,
                "contract_commitment_required": "1-Year Fixed Price Agreement"
            },
            "financial_budget_check": {
                "total_discount_value_usd": total_val,
                "budget_cap_usd": 120.00,
                "within_budget": total_val <= 120.00
            },
            "outbound_personalized_email_body": (
                f"Dear Customer, As a valued member of our network for the past {tenure} months, we noticed your current plan may not be "
                f"delivering the maximum value for your connectivity needs. To ensure your complete peace of mind, we have activated an exclusive "
                f"${monthly_disc:.2f}/month loyalty credit for the next {duration_mo} months, alongside complimentary access to our 24/7 dedicated "
                f"VIP Tech Support and Cyber Shield protection when upgrading to our 1-Year Price-Lock Guarantee."
            ),
            "csm_call_battlecard_talking_points": [
                f"Acknowledge {tenure} months of continuous account history and thank customer.",
                f"Directly neutralize price sensitivity by introducing the ${total_val:.2f} total annual loyalty subsidy.",
                f"Position the complimentary {add_on} as proactive account stabilization.",
                "Secure verbal commitment to lock in the 1-year rate before pending month-to-month billing cycle close."
            ]
        }

if __name__ == "__main__":
    agent = OpenAIRetentionAgent()
    sample_profile = {
        "customerID": "7590-VHVEG",
        "tenure": 2,
        "Contract": "Month-to-month",
        "MonthlyCharges": 89.85,
        "TotalCharges": 179.70,
        "InternetService": "Fiber optic",
        "TechSupport": "No",
        "OnlineSecurity": "No",
        "PaymentMethod": "Electronic check",
        "churn_probability": 0.884
    }
    sample_drivers = [
        "Contract: Month-to-month (+0.42 SHAP risk)",
        "InternetService: Fiber optic high price point (+0.28 SHAP risk)",
        "TechSupport: No coverage (+0.19 SHAP risk)",
        "Short tenure: < 3 months (+0.15 SHAP risk)"
    ]
    plan = agent.generate_retention_action(sample_profile, sample_drivers)
    print("OpenAI Retention Agent Generated Plan:")
    print(json.dumps(plan, indent=2))
