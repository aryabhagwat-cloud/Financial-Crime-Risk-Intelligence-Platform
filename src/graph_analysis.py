import pandas as pd
import networkx as nx
import os

print("--- Person 2: Graph Analysis Starting ---")

def load_data(filepath):
    print(f"Loading enriched data from: {filepath}")
    df = pd.read_csv(filepath)
    return df

def build_graph(df):
    print("Building Transaction Graph...")
    G = nx.DiGraph() # Directed Graph (Money moves A -> B)
    
    # We aggregate transactions: If A sends to B 5 times, that's one strong edge.
    # Group by Entity ID (Source -> Target)
    edges = df.groupby(['entity_id', 'nameDest'])['amount'].sum().reset_index()
    
    # Add edges to graph
    # Node A -> Node B with weight = Total Amount Sent
    for _, row in edges.iterrows():
        G.add_edge(row['entity_id'], row['nameDest'], weight=row['amount'])
        
    print(f"Graph Built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

def extract_graph_features(G, df):
    print("Calculating Graph Features (PageRank, Degree, Communities)...")
    
    # 1. PageRank (Measure of influence/risk propagation)
    pagerank = nx.pagerank(G, weight='weight')
    
    # 2. Degree Centrality (How many people do they transact with?)
    degree = nx.degree_centrality(G)
    
    # 3. Community Detection (Finding Fraud Rings)
    # We use a simple greedy modularity method for finding clusters
    # (In real life, small tight clusters = fraud rings)
    communities = list(nx.community.greedy_modularity_communities(G))
    
    # Map community ID to each node
    community_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_map[node] = i
            
    print(f"Detected {len(communities)} distinct communities (potential rings).")

    # --- MERGE FEATURES BACK TO DATAFRAME ---
    # We map the graph metrics back to the original transactions based on the Sender (entity_id)
    
    df['pagerank_score'] = df['entity_id'].map(pagerank)
    df['degree_centrality'] = df['entity_id'].map(degree)
    df['community_id'] = df['entity_id'].map(community_map)
    
    # Fill missing values (for receivers who never sent money)
    df['pagerank_score'] = df['pagerank_score'].fillna(0)
    df['degree_centrality'] = df['degree_centrality'].fillna(0)
    df['community_id'] = df['community_id'].fillna(-1)
    
    print("Graph features merged.")
    return df

if __name__ == "__main__":
    # Setup Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_PATH = os.path.join(BASE_DIR, 'data', 'enriched_transactions.csv')
    OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'graph_features.csv')
    
    if os.path.exists(INPUT_PATH):
        df = load_data(INPUT_PATH)
        G = build_graph(df)
        df_final = extract_graph_features(G, df)
        
        df_final.to_csv(OUTPUT_PATH, index=False)
        print(f"SUCCESS! Graph features saved to: {OUTPUT_PATH}")
    else:
        print("Error: Could not find 'enriched_transactions.csv'. Did Person 1 finish?")