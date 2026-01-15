import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import numpy as np

print("--- Person 4: Explainability & Alerts Starting ---")

def load_resources(data_path, model_path):
    print("Loading model and test results...")
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    return df, model

def generate_global_explanation(model, X, output_path):
    print("Generating Global Feature Importance Plot...")
    
    # Get feature importance from Random Forest
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1] # Sort highest to lowest
    feature_names = X.columns
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.title("Top Fraud Predictors (Global Explainability)")
    plt.bar(range(X.shape[1]), importances[indices], align="center", color='salmon')
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Feature Importance graph saved to: {output_path}")

def generate_alert_report(df, output_path):
    print("Generating Suspicious Activity Report (Reason Codes)...")
    
    # Filter for transactions the model predicted as FRAUD (1)
    high_risk = df[df['Predicted_Fraud'] == 1].head(10) # Top 10 examples
    
    with open(output_path, "w") as f:
        f.write("=== FINANCIAL CRIME SUSPICIOUS ACTIVITY REPORT ===\n")
        f.write("System: Graph-Based Fraud Detection Engine\n")
        f.write("Status: URGENT REVIEW REQUIRED\n\n")
        
        if high_risk.empty:
            f.write("No high-risk transactions detected in this batch.\n")
        else:
            for index, row in high_risk.iterrows():
                f.write(f"Transaction Index: {index}\n")
                f.write(f"Risk Score: HIGH\n")
                f.write(f"Amount: ${row['amount']:.2f}\n")
                
                # Dynamic Reason Codes (Explain WHY it's fraud)
                reasons = []
                if row['pagerank_score'] > 0.001:
                    reasons.append(f"[GRAPH] High PageRank ({row['pagerank_score']:.5f}) - Hub Node")
                if row['degree_centrality'] > 0.01:
                    reasons.append(f"[GRAPH] High Degree Centrality - Connected to many accounts")
                if row['community_id'] != -1:
                    reasons.append(f"[RING] Part of Suspicious Community #{int(row['community_id'])}")
                if row['oldbalanceOrg'] == 0:
                    reasons.append(f"[BEHAVIOR] Instant Account Emptying")
                
                if not reasons:
                    reasons.append("Complex Pattern Anomaly")
                    
                f.write("Alert Reasons:\n")
                for r in reasons:
                    f.write(f" - {r}\n")
                f.write("-" * 50 + "\n")
            
    print(f"Alert report saved to: {output_path}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_PATH = os.path.join(BASE_DIR, 'data', 'test_results.csv')
    MODEL_PATH = os.path.join(BASE_DIR, 'data', 'fraud_model.pkl')
    PLOT_PATH = os.path.join(BASE_DIR, 'data', 'feature_importance.png')
    REPORT_PATH = os.path.join(BASE_DIR, 'data', 'final_alert_log.txt')
    
    if os.path.exists(RESULTS_PATH) and os.path.exists(MODEL_PATH):
        # 1. Load Data
        df, model = load_resources(RESULTS_PATH, MODEL_PATH)
        
        # Prepare X (features) by dropping non-feature columns
        # We drop the targets so we can just see the input features
        drop_cols = ['isFraud', 'Actual_Fraud', 'Predicted_Fraud']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        
        # 2. Generate Visual
        generate_global_explanation(model, X, PLOT_PATH)
        
        # 3. Generate Report
        generate_alert_report(df, REPORT_PATH)
        
        print("SUCCESS! Person 4 Complete.")
    else:
        print("Error: Missing test_results.csv or fraud_model.pkl. Did Person 3 finish?")