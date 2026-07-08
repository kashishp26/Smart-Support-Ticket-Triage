import streamlit as st
import requests
import os
import json
import pandas as pd

# Page Configuration Setup
st.set_page_config(
    page_title="Smart Support Ticket Triage System",
    page_icon="🎟️",
    layout="wide"
)

# Main Header Title
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
# -------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.info(" **Tip:** Use Classical ML for high-speed, zero-cost classification. Switch to Gemini LLM for complex context understanding and critical urgency routing.")


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