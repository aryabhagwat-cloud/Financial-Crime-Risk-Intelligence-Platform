import pandas as pd
import numpy as np
import networkx as nx
from faker import Faker
import random
import os

print("--- Script Starting ---") # If you don't see this, the file isn't saved!

# Initialize Faker
fake = Faker()
Faker.seed(42)
random.seed(42)

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    # Load first 100k rows
    df = pd.read_csv(filepath, nrows=100000) 
    df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
    print(f"Data Loaded: {len(df)} rows retained.")
    return df

def generate_synthetic_features(df):
    print("Generating synthetic features...")
    unique_users = list(df['nameOrig'].unique())
    num_devices = int(len(unique_users) * 0.8) 
    num_ips = int(len(unique_users) * 0.7)
    
    device_pool = [f"DEV_{i}" for i in range(num_devices)]
    ip_pool = [fake.ipv4() for i in range(num_ips)]
    
    user_device_map = {user: random.choice(device_pool) for user in unique_users}
    user_ip_map = {user: random.choice(ip_pool) for user in unique_users}
    
    df['device_id'] = df['nameOrig'].map(user_device_map)
    df['ip_addr'] = df['nameOrig'].map(user_ip_map)
    print("Features added.")
    return df

def resolve_entities(df):
    print("Resolving entities...")
    G = nx.Graph()
    edges = list(zip(df['nameOrig'], df['device_id']))
    G.add_edges_from(edges)
    
    entity_map = {}
    for i, component in enumerate(nx.connected_components(G)):
        entity_id = f"ENT_{i}"
        for node in component:
            if str(node).startswith("C"): 
                entity_map[node] = entity_id
                
    df['entity_id'] = df['nameOrig'].map(entity_map)
    df['entity_id'] = df['entity_id'].fillna(df['nameOrig'])
    print("Resolution complete.")
    return df

if __name__ == "__main__":
    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_PATH = os.path.join(BASE_DIR, 'data', 'raw_paysim.csv')
    OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'enriched_transactions.csv')
    
    print(f"Looking for file at: {INPUT_PATH}")
    
    if os.path.exists(INPUT_PATH):
        df = load_data(INPUT_PATH)
        df = generate_synthetic_features(df)
        df = resolve_entities(df)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"SUCCESS! Output saved to: {OUTPUT_PATH}")
    else:
        print(f"ERROR: Could not find file at {INPUT_PATH}")