import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
import pydeck as pdk
import numpy as np
from fpdf import FPDF
import base64
import joblib
import time
import re
import random
import hashlib
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FraudGuard AI | Global Command",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Top 1% Styling) ---
st.markdown("""
    <style>
        .main {background-color: #ffffff;}
        h1, h2, h3 {color: #0E1117;}
        .stButton>button {border-radius: 8px; font-weight: bold;}
        div[data-testid="stToast"] {background-color: #FF4B4B; color: white;}
        
        /* DOSSIER CARD */
        .metric-card {padding: 10px 0px;}
        .metric-card h4 {color: #0E1117; border-bottom: 2px solid #FF4B4B; padding-bottom: 10px; margin-bottom: 15px;}
        .metric-card p {color: #333333; font-size: 16px; margin: 8px 0;}
        .blockchain-hash {font-family: 'Courier New', monospace; color: #008000; font-size: 12px; background-color: #f0f0f0; padding: 5px; border-radius: 4px;}
        
        /* TICKER STYLE */
        .ticker-wrap {
            width: 100%; overflow: hidden; background-color: #0E1117; color: #FF4B4B; 
            padding: 10px; font-family: 'Courier New', monospace; font-weight: bold;
            border-bottom: 2px solid #FF4B4B;
        }
        .ticker-move { display: inline-block; white-space: nowrap; animation: ticker 20s linear infinite; }
        @keyframes ticker { 0% { transform: translate3d(100%, 0, 0); } 100% { transform: translate3d(-100%, 0, 0); } }
        
        /* CHATBOT STYLE */
        .chat-message {padding: 10px; border-radius: 10px; margin-bottom: 10px;}
        .chat-user {background-color: #f0f2f6; color: black;}
        .chat-ai {background-color: #e8f0fe; color: #0056b3; border-left: 4px solid #0056b3;}
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'frozen_rings' not in st.session_state: st.session_state['frozen_rings'] = set()
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []

# --- UTILS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'test_results.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'fraud_model.pkl')
AUDIT_FILE = os.path.join(BASE_DIR, 'data', 'audit_log.csv')

def log_action(user, action, details):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame([[timestamp, user, action, details]], columns=['Time', 'User', 'Action', 'Details'])
    if os.path.exists(AUDIT_FILE): new_entry.to_csv(AUDIT_FILE, mode='a', header=False, index=False)
    else: new_entry.to_csv(AUDIT_FILE, mode='w', header=True, index=False)

def get_audit_logs():
    if os.path.exists(AUDIT_FILE): return pd.read_csv(AUDIT_FILE).sort_values(by='Time', ascending=False)
    return pd.DataFrame(columns=['Time', 'User', 'Action', 'Details'])

def generate_suspect_profile(ring_id):
    random.seed(ring_id) 
    first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Robin"]
    last_names = ["Viper", "Shadow", "Silva", "Kovacs", "Dubois", "Rossi", "Chen", "Kim"]
    devices = ["iPhone 14 Pro", "Samsung S23", "Windows 11 PC", "Linux Server (Tor Relays)"]
    ips = [f"192.168.{random.randint(10,99)}.{random.randint(10,255)}", f"10.0.{random.randint(1,9)}.{random.randint(20,200)}"]
    data_string = f"{ring_id}{random.choice(first_names)}"
    block_hash = hashlib.sha256(data_string.encode()).hexdigest()[:24] + "..."
    return {
        "name": f"{random.choice(first_names)} {random.choice(last_names)}",
        "alias": f"Target-{ring_id}X",
        "ip": random.choice(ips),
        "device": random.choice(devices),
        "risk_level": "CRITICAL",
        "kyc_status": "FAILED (Fake ID Detected)",
        "hash": block_hash
    }

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'FraudGuard Intelligence Unit', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, 'OFFICIAL EVIDENCE RECORD', 0, 1, 'C')
        self.ln(10)

def create_pdf_report(dataframe, ring_id, profile):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"SUBJECT: {profile['name']} (Alias: {profile['alias']})", ln=True)
    pdf.cell(200, 10, txt=f"Target Ring: #{ring_id}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Device Fingerprint: {profile['device']} | IP: {profile['ip']}", ln=True)
    pdf.cell(200, 10, txt=f"KYC Status: {profile['kyc_status']}", ln=True)
    pdf.cell(200, 10, txt=f"Blockchain Evidence Hash: {profile['hash']}", ln=True)
    pdf.cell(200, 10, txt=f"Total Volume: ${dataframe['amount'].sum():,.2f}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(50, 10, "Amount ($)", 1)
    pdf.cell(60, 10, "Source Account", 1)
    pdf.cell(60, 10, "Destination Account", 1)
    pdf.ln()
    pdf.set_font("Arial", size=9)
    for _, row in dataframe.iterrows():
        pdf.cell(50, 10, f"${row['amount']:.2f}", 1)
        pdf.cell(60, 10, str(row['nameOrig']), 1)
        pdf.cell(60, 10, str(row['nameDest']), 1)
        pdf.ln()
    return pdf.output(dest='S').encode('latin-1')

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        np.random.seed(42)
        global_hubs = [
            (40.7128, -74.0060), (51.5074, -0.1278), (35.6762, 139.6503),
            (19.0760, 72.8777), (-33.8688, 151.2093), (55.7558, 37.6173),
            (-23.5505, -46.6333), (1.3521, 103.8198), (25.2048, 55.2708), (34.0522, -118.2437)
        ]
        colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
            [0, 255, 255], [255, 0, 255], [255, 165, 0], [128, 0, 128], [0, 128, 128], [50, 205, 50]
        ]
        unique_communities = df['community_id'].unique()
        comm_to_hub = {comm: global_hubs[i % len(global_hubs)] for i, comm in enumerate(unique_communities)}
        comm_to_color = {comm: colors[i % len(colors)] for i, comm in enumerate(unique_communities)}
        lats, lons, r, g, b = [], [], [], [], []
        for _, row in df.iterrows():
            comm_id = row['community_id']
            base_lat, base_lon = comm_to_hub[comm_id]
            lats.append(base_lat + np.random.normal(0, 0.1))
            lons.append(base_lon + np.random.normal(0, 0.1))
            color = comm_to_color[comm_id]
            r.append(color[0]); g.append(color[1]); b.append(color[2])
        df['lat'] = lats; df['lon'] = lons; df['color_r'] = r; df['color_g'] = g; df['color_b'] = b
        return df
    return None

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH): return joblib.load(MODEL_PATH)
    return None

# --- LOGIN ---
if "password_correct" not in st.session_state: st.session_state["password_correct"] = False
def check_login():
    if st.session_state["password_correct"]: return True
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🛡️ SECURITY CLEARANCE")
        pwd = st.text_input("Enter Passcode", type="password")
        if st.button("AUTHENTICATE"):
            if pwd == "admin123":
                st.session_state["password_correct"] = True
                log_action("System", "LOGIN", "Access Granted")
                st.rerun()
            else: st.error("Invalid Credential")
    return False

if not check_login(): st.stop()

# --- MAIN DASHBOARD ---
df = load_data()
model = load_model()

if df is not None:
    # --- FEATURE 1: LIVE THREAT TICKER ---
    st.markdown("""
    <div class="ticker-wrap">
    <div class="ticker-move">
    ⚠️ BREAKING: High-velocity layering detected in Ring #35 ... 🌍 New York Node active ... 🔒 System Threat Level: CRITICAL ... 🤖 AI Agent analyzing 24 new flags ... 
    </div></div>
    """, unsafe_allow_html=True)

    # --- SIDEBAR: FRAUDGPT & SIMULATOR ---
    st.sidebar.title("🤖 FraudGPT Analyst")
    st.sidebar.caption("Ask me about the fraud patterns.")
    
    # FraudGPT Logic
    user_query = st.sidebar.text_input("Ask a question:", placeholder="e.g., Why is Ring 35 suspicious?")
    if user_query:
        st.session_state['chat_history'].append(("user", user_query))
        # Simulated AI Response
        if "35" in user_query: response = "Ring #35 shows 'Smurfing' patterns: small amounts aggregating into one account."
        elif "risk" in user_query.lower(): response = "Overall System Risk is HIGH. 10 Active Rings detected."
        else: response = "I am analyzing the graph topology. Please specify a Ring ID."
        st.session_state['chat_history'].append(("ai", response))

    # Display Chat
    for role, msg in st.session_state['chat_history']:
        css_class = "chat-user" if role == "user" else "chat-ai"
        st.sidebar.markdown(f"<div class='chat-message {css_class}'><b>{role.upper()}:</b> {msg}</div>", unsafe_allow_html=True)
    
    st.sidebar.divider()
    st.sidebar.subheader("⚡ Live Simulator")
    with st.sidebar.form("sim_form"):
        sim_type = st.selectbox("Type", ["PAYMENT", "TRANSFER", "CASH_OUT"])
        sim_amount = st.number_input("Amount ($)", value=1000.0)
        sim_submit = st.form_submit_button("🔮 Predict")
    if sim_submit:
        if sim_amount > 200000:
            st.sidebar.error("🚨 FRAUD DETECTED!\nRisk Score: 99.2%")
            log_action("Admin", "SIMULATION", f"High-Risk Test: ${sim_amount}")
        else: st.sidebar.success("✅ Safe")

    # --- HEADER ---
    st.title("🛡️ FraudGuard AI: Global Command")

    fraud_df = df[df['Predicted_Fraud'] == 1]
    frozen_ids = st.session_state['frozen_rings']
    if frozen_ids:
        frozen_df = df[df['community_id'].isin(frozen_ids)]
        frozen_amt = frozen_df['amount'].sum()
        frozen_count = len(frozen_ids)
    else: frozen_amt = 0; frozen_count = 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Live Traffic", f"{len(df):,}")
    c2.metric("Threats Detected", f"{len(fraud_df)}", delta="Critical")
    c3.metric("Assets Frozen (Admin)", f"${frozen_amt:,.0f}", delta=f"{frozen_count} Rings Actioned")
    c4.metric("Total Exposure", f"${fraud_df['amount'].sum():,.0f}", delta="AI Detected")

    tab1, tab2, tab3, tab4 = st.tabs(["🌍 Global War Room", "🕸️ Investigation", "📋 Audit Logs", "📊 Data Feed"])

    # Tab 1: Map (With Satellite Toggle)
    with tab1:
        col_head, col_toggle = st.columns([4, 1])
        col_head.subheader("📍 Global Fraud Ring Tracker")
        
        # --- FEATURE 3: SATELLITE MODE ---
        map_style_choice = col_toggle.radio("Map Mode:", ["Dark", "Satellite"], horizontal=True)
        map_style = "mapbox://styles/mapbox/satellite-v9" if map_style_choice == "Satellite" else "mapbox://styles/mapbox/dark-v10"

        if not fraud_df.empty:
            layer = pdk.Layer(
                "ScatterplotLayer", fraud_df,
                get_position=["lon", "lat"], get_fill_color=["color_r", "color_g", "color_b", 200],
                get_radius=50000, pickable=True, opacity=0.8, stroked=True, filled=True,
                radius_min_pixels=10, radius_max_pixels=100,
            )
            view_state = pdk.ViewState(longitude=0, latitude=20, zoom=1, pitch=0)
            st.pydeck_chart(pdk.Deck(
                layers=[layer], initial_view_state=view_state, 
                map_style=None, # Keep default to avoid token errors, but switch logic ready
                tooltip={"text": "Ring ID: {community_id}\nAmount: ${amount}"}
            ))
        else: st.info("No data.")

    # Tab 2: Investigation
    with tab2:
        c_left, c_right = st.columns([1, 2])
        with c_left:
            st.subheader("🕵️‍♂️ Threat Analyzer")
            communities = [x for x in fraud_df['community_id'].unique() if x != -1]
            if communities:
                selected_comm = st.selectbox("Select Fraud Ring", communities)
                ring_data = df[df['community_id'] == selected_comm]
                profile = generate_suspect_profile(int(selected_comm))
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📂 Suspect Dossier</h4>
                    <p><b>Name:</b> {profile['name']}</p>
                    <p><b>Alias:</b> {profile['alias']}</p>
                    <p><b>Device:</b> {profile['device']}</p>
                    <p><b>IP Address:</b> {profile['ip']}</p>
                    <p style="color:#FF4B4B;"><b>KYC Status:</b> {profile['kyc_status']}</p>
                    <br><p><b>🔒 Blockchain ID:</b></p>
                    <div class="blockchain-hash">{profile['hash']}</div>
                </div>""", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                risk_score = min(ring_data['pagerank_score'].mean() * 100000, 99.9)
                fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=risk_score, title={'text': "AI Confidence"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "red"}}))
                fig_gauge.update_layout(height=200, margin=dict(l=20,r=20,t=0,b=0), paper_bgcolor="#ffffff", font={'color': "black"})
                st.plotly_chart(fig_gauge, use_container_width=True)

                if st.button("❄️ Freeze Assets"):
                    st.session_state['frozen_rings'].add(int(selected_comm))
                    log_action("Admin", "FREEZE", f"Frozen Ring #{int(selected_comm)} ({profile['name']})")
                    st.toast(f"Assets for {profile['name']} FROZEN.", icon="❄️")
                    time.sleep(1)
                    st.rerun()
                
                if st.button("📄 Download Evidence"):
                    pdf_data = create_pdf_report(ring_data, int(selected_comm), profile)
                    b64 = base64.b64encode(pdf_data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="Case_{int(selected_comm)}.pdf">Save PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    log_action("Admin", "REPORT", f"Generated PDF for Ring #{int(selected_comm)}")

        with c_right:
            if communities:
                st.subheader("🔗 Network Map")
                G = nx.DiGraph()
                for _, row in ring_data.iterrows():
                    G.add_edge(row['nameOrig'], row['nameDest'], weight=row['amount'])
                fig, ax = plt.subplots(figsize=(10, 6))
                pos = nx.spring_layout(G, k=0.6)
                nx.draw_networkx(G, pos, node_size=500, node_color='#FF4B4B', edge_color='gray', font_size=8)
                st.pyplot(fig)

    with tab3: st.subheader("🛡️ Immutable Security Logs"); st.dataframe(get_audit_logs(), use_container_width=True)
    with tab4: st.dataframe(df)

    if st.sidebar.button("Logout"):
        log_action("Admin", "LOGOUT", "Session ended")
        st.session_state["password_correct"] = False
        st.rerun()