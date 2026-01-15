import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os
import joblib

print("--- Person 3: ML Training Starting ---")

def load_data(filepath):
    print(f"Loading graph-enriched data from: {filepath}")
    df = pd.read_csv(filepath)
    return df

def prepare_features(df):
    print("Preparing features for AI...")
    
    # 1. Define the columns we want for the MODEL (Math only)
    feature_cols = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest', 
        'pagerank_score', 'degree_centrality', 'community_id'
    ]
    
    # 2. Create the Feature Matrix (X)
    # We use get_dummies just for X, but we KEEP 'df' intact for later
    X = df[feature_cols + ['type']].copy()
    X = pd.get_dummies(X, columns=['type'], drop_first=True)
    
    # Fill any NaNs in X
    X = X.fillna(0)
    
    # 3. Define Target (y)
    y = df['isFraud']
    
    print(f"Training with {X.shape[1]} features.")
    
    # RETURN X, y, AND the original DF (so we don't lose names/steps)
    return X, y, df

def train_model(X, y, df):
    print("Splitting data and training Random Forest...")
    
    # KEY FIX: We split X, y, AND df together so indices match
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=42
    )
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    print("Model trained successfully.")
    
    # Evaluate
    print("--- Model Performance ---")
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Add predictions back to the TEST DATAFRAME (which still has names!)
    df_test = df_test.copy()
    df_test['Actual_Fraud'] = y_test
    df_test['Predicted_Fraud'] = y_pred
    
    return rf, df_test

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_PATH = os.path.join(BASE_DIR, 'data', 'graph_features.csv')
    MODEL_PATH = os.path.join(BASE_DIR, 'data', 'fraud_model.pkl')
    RESULTS_PATH = os.path.join(BASE_DIR, 'data', 'test_results.csv')
    
    if os.path.exists(INPUT_PATH):
        # 1. Load
        df = load_data(INPUT_PATH)
        
        # 2. Prepare
        X, y, df_full = prepare_features(df)
        
        # 3. Train & Get Results (Note the change in arguments)
        model, df_results = train_model(X, y, df_full)
        
        # 4. Save Model
        joblib.dump(model, MODEL_PATH)
        
        # 5. Save Results (Now containing names, steps, and types!)
        df_results.to_csv(RESULTS_PATH, index=False)
        
        print(f"SUCCESS! Model saved to {MODEL_PATH}")
        print(f"Test results (with transaction details) saved to {RESULTS_PATH}")
    else:
        print("Error: Could not find 'graph_features.csv'.")