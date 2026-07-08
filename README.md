# 🎫 Smart Support Ticket Triage System

A production-grade, hybrid enterprise application designed to automatically classify, prioritize, and route customer support tickets. This project showcases a side-by-side comparison and functional integration between **Classical Machine Learning (Local)** and **Modern Generative AI LLM Orchestration (Cloud)**.

---

##  Key Features

- **Hybrid Dataset Pipeline:** Custom data engineering pipeline that merges realistic enterprise production patterns with synthetic template expansions to eliminate class imbalance.
- **Dual-Engine Routing Architecture:** 
  - **Local ML Engine:** Fast text vectorization using TF-IDF coupled with a Logistic Regression classifier (Requires zero internet or API costs).
  - **Cloud LLM Engine:** Powered by Google Gemini 2.5 Flash API with advanced rule-based system instructions and structural JSON enforcement.
- **Enterprise Architecture:** Clean decoupling of Frontend (Streamlit microservice) and Backend (FastAPI web server service).

---

##  Repository Structure

```text
Support_Ticket_Triage/              <-- Main Project Root
│
├── backend/                        <-- Core Backend Service
│   ├── data/                       # Generated Datasets (Train/Test CSVs)
│   ├── models/                     # Serialized Local ML Models (.pkl)
│   ├── src/
│   │   ├── baseline_ml.py          # Classical ML Training & Inference
│   │   ├── data_prep.py            # Hybrid Data Generation & Stratification
│   │   ├── evaluate_pipeline.py    # Pipeline Evaluation Metrics
│   │   ├── llm_classifier.py       # Google Gemini Live API Wrapper
│   │   └── main.py                 # FastAPI Gateway Web Server
│   └── .env                        # Secure Environment Variables (API Keys)
│
├── frontend/                       <-- Core Frontend UI Service
│   └── app.py                      # Streamlit Interactive Dashboard
│
└── README.md                       # Comprehensive Documentation

#Installation & Setup
1. Backend Microservice Setup
Navigate to the backend directory, activate your virtual environment, and boot up the FastAPI server:

Bash
cd backend
# Activate Environment (Windows)
venv\Scripts\activate

# Start the FastAPI Live Server
uvicorn src.main:app --reload
The backend gateway will boot up and remain alive at http://127.0.0.1:8000.

2. Frontend Dashboard Setup
Open a new terminal split (+ icon), navigate to the frontend folder, activate the environment, and launch Streamlit:

Bash
cd frontend
# Activate Environment (Windows)
..\backend\venv\Scripts\activate

# Start Streamlit UI
streamlit run app.py
The web app dashboard will automatically open in your browser session.

# Methodology & Technical Details
🔹 Day 1: Hybrid Data Strategy
Instead of utilizing generic static Kaggle sheets, a dedicated programmatic data engine (data_prep.py) was structured. It crafts a custom balanced distribution across 4 distinct corporate categories (Billing, Technical, Account, Other) using stratified splitting techniques.

🔹 Day 2: TF-IDF Text Classification
The local engine computes statistical word importance across the training dataset, outputting serialized mathematical weight assignments via Python's joblib module. This acts as a cost-effective, zero-latency local fallback.

🔹 Day 3: Generative AI Orchestration
Utilizes live zero-shot/few-shot system behavior mappings on top of the gemini-2.5-flash model. Strict rule-based guardrails ensure predictable corporate classification and programmatic JSON outputs.

## API Endpoints Docs
POST /predict
Routes the incoming ticket text to the designated engine.

Request Body Layout:

JSON
{
  "text": "Our main production database endpoint /v1/data is throwing a timeout error!",
  "approach": "Gemini LLM (Zero-Shot Prompting)"
}
Response Output Layout:

JSON
{
  "category": "Technical",
  "urgency": "Critical",
  "model_used": "Gemini 2.5 Flash (Zero-Shot)"
}