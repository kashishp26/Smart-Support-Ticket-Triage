import streamlit as st
import requests
import os
import json
import pandas as pd

from auth_db import login_user, register_user

# Initialize session state for login
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None

# Page Configuration Setup
st.set_page_config(
    page_title="Smart Support Ticket Triage System",
    page_icon="🎟️",
    layout="wide"
)

# Main Header Title

if not st.session_state["logged_in"]:
    
    st.markdown("""
        <style>
        /* Sidebar and top header space alignment */
        [data-testid="stAppViewContainer"] > .main {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        </style>
    """, unsafe_allow_html=True)

    left_spacer, center_col, right_spacer = st.columns([1, 2, 1])

    with center_col:
        # Form box boundary block
        st.markdown("<h1 style='text-align: center;'></h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>Smart Support Triage - Portal</h2>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs([" Login", " Register"])
        
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            login_user_input = st.text_input("Username", key="login_user")
            login_pass_input = st.text_input("Password", type="password", key="login_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("Login", use_container_width=True, type="primary"):
                if login_user(login_user_input, login_pass_input):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = login_user_input
                    st.success(f"Welcome back, {login_user_input}!")
                    st.rerun()
                else:
                    st.error("Invalid Username or Password!")
                    
        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            reg_user_input = st.text_input("Choose Username", key="reg_user")
            reg_pass_input = st.text_input("Choose Password", type="password", key="reg_pass")
            reg_pass_confirm = st.text_input("Confirm Password", type="password", key="reg_pass_conf")
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("Sign Up", use_container_width=True):
                if not reg_user_input or not reg_pass_input:
                    st.error("Fields cannot be empty!")
                elif reg_pass_input != reg_pass_confirm:
                    st.error("Passwords do not match!")
                else:
                    success, msg = register_user(reg_user_input, reg_pass_input)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
                        
    st.stop() 

st.title("🎟️ Smart Support Ticket Triage System")
st.markdown("Automated ticket classification and routing powered by Classical ML & Gemini LLM Agents.")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header(" Pipeline Settings")

# Model selection dropdown
approach_option = st.sidebar.selectbox(
    "Select Inference Engine / Model:",
    [
        "Classical ML (Local Ensemble Suite)",
        "Gemini 2.5 Flash (Zero-Shot Prompting)",
        "Gemini 2.5 Flash (Few-Shot Prompting)"
    ]
)

# Metrics file  path is used to connect the backend models folder 
metrics_path = os.path.join("..", "backend", "models", "model_metrics.json")

if os.path.exists(metrics_path):
    st.sidebar.markdown("---")
    st.sidebar.subheader(" Classical ML Leaderboard")
    
    # Read the accuracy scores of all three models from the JSON file.
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
   # Convert the data into a tabular format (DataFrame).
    df_metrics = pd.DataFrame(metrics)
    
   # Append the '%' symbol to each accuracy score for better readability.
    df_styled = df_metrics.map(lambda x: f"{x:.2f}%")
    
   # Display the table in the Streamlit sidebar.
    st.sidebar.table(df_styled)
    st.sidebar.caption("The pipeline automatically utilizes the highest-scoring model for local inference.")

st.sidebar.markdown("---")
st.sidebar.info(" **Tip:** Use Classical ML for high-speed, zero-cost classification. Switch to Gemini LLM for complex context understanding and critical urgency routing.")

st.sidebar.markdown("---")
st.sidebar.write(f"Logged in as: **{st.session_state['username']}**")
if st.sidebar.button("Logout", type="primary", use_container_width=True):
    st.session_state["logged_in"] = False
    st.session_state["username"] = None
    st.rerun()


# --- MAIN SCREEN INTERFACE ---
st.subheader("📝 Enter Support Ticket Details")
ticket_text = st.text_area(
    "Paste the customer support ticket text below:",
    height=150,
    placeholder="Example: I cannot log in to my payroll account, it keeps throwing a 504 gateway timeout..."
)

# API endpoint backend URL
BACKEND_URL = "http://127.0.0.1:8000/predict"

if st.button(" Classify & Route Ticket", type="primary"):
    if not ticket_text.strip():
        st.warning(" Please enter some ticket text before submitting!")
    else:
        with st.spinner("Analyzing text and evaluating models... Please wait..."):
            try:
                # Backend API Payload preparing
                payload = {
                    "text": ticket_text,
                    "approach": approach_option
                }
                
                # Fast API backend standard post call
                response = requests.post(BACKEND_URL, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("Analysis Completed Successfully!")
                    
            # Arrange the columns to make the output more presentable.
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(label="Predicted Category", value=result.get("category", "N/A"))
                    with col2:
                       # Highlight urgency using color-coded tags.
                        urgency = result.get("urgency", "N/A")
                        st.metric(label="Urgency Level", value=urgency)
                    with col3:
                        st.metric(label="Engine / Model Deployed", value=result.get("model_used", "Classical ML Engine"))
                        
            # Print additional routing logs for clearer interpretation.
                    st.markdown("---")
                    st.markdown(f"** Routing Engine Log:** Ticket has been successfully categorized as **{result.get('category')}** with a priority matrix of **{result.get('urgency')}** via **{result.get('model_used', approach_option)}**.")
                    
                else:
                    st.error(f" Backend Server Error (Status Code: {response.status_code})")
                    st.json(response.json())
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the Backend API Server! Please ensure that your Uvicorn backend server is running on http://127.0.0.1:8000")