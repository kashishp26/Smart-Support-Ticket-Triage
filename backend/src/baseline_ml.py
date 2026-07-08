import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

# Load Environment Variables for Gemini API
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Paths setting
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
data_dir = os.path.join(backend_dir, "data")
models_dir = os.path.join(backend_dir, "models")

EMBEDDING_MODEL = "models/gemini-embedding-001"

# Helper Function: Generate embeddings with strict rate limiting and a safe fallback mechanism.
def get_gemini_embeddings(text_list, batch_size=10):
    print(f" Generating Gemini Embeddings for {len(text_list)} texts in small batches...")
    embeddings = []
    
    total_batches = (len(text_list) - 1) // batch_size + 1
    
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        batch = [str(t) if (str(t).strip() and pd.notna(t)) else "Empty ticket text" for t in batch]
        
        current_batch_num = i // batch_size + 1
        success = False
        
        try:
            # Small batch execution
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch,
                task_type="classification"
            )
            # Ensure elements inside are strict lists/arrays of floats
            for emb in result['embedding']:
                embeddings.append(list(emb))
            print(f" Processed batch {current_batch_num}/{total_batches}")
            success = True
            
           # Add a safe 5-second delay to avoid exceeding the quota limit.
            time.sleep(5)
            
        except Exception as e:
            print(f" Error in batch {current_batch_num}: {e}")
            print(" Quota cooldown activated. Waiting for 25 seconds before individual fallback...")
            time.sleep(25)
            
         # Apply individual text retry fallback logic if the entire batch request fails.  
            for text in batch:
                try:
                    res = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="classification")
                    embeddings.append(list(res['embedding'][0] if isinstance(res['embedding'][0], list) else res['embedding']))
                    time.sleep(3)
                except Exception as inner_e:
                    print(f"❌ Individual fallback failed: {inner_e}")
                    embeddings.append(list(np.zeros(768))) # Strict size match array
                    
    return np.array(embeddings, dtype=np.float32)

def load_ml_pipeline():
    cat_model_path = os.path.join(models_dir, "category_model.pkl")
    urg_model_path = os.path.join(models_dir, "urgency_model.pkl")
    
    if os.path.exists(cat_model_path) and os.path.exists(urg_model_path):
        cat_pipeline = joblib.load(cat_model_path)
        urg_pipeline = joblib.load(urg_model_path)
        return cat_pipeline, urg_pipeline
    else:
        return train_baseline_models()

def train_baseline_models():
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(" Datasets not found. Please run `data_prep.py` first.")
        
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    os.makedirs(models_dir, exist_ok=True)
    
    # Text datasets arrays generating via Gemini API
    X_train = get_gemini_embeddings(df_train['text'].tolist())
    X_test = get_gemini_embeddings(df_test['text'].tolist())
    
    # 3 Classical ML Models Suite
    models_to_compare = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, class_weight='balanced', C=1.5),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "Linear SVM": lambda: LinearSVC(class_weight='balanced', C=1.0, max_iter=2000)
    }
    
    metrics_data = {"Category": {}, "Urgency": {}}
    
    # --- 1. CATEGORY MODELS COMPARISON ---
    print("\n --- COMPARING EMBEDDING MODELS FOR CATEGORY ---")
    best_cat_score = -1
    best_cat_model = None
    best_cat_name = ""
    
    for name, model_init in models_to_compare.items():
        m = model_init()
        m.fit(X_train, df_train['category'])
        preds = m.predict(X_test)
        score = accuracy_score(df_test['category'], preds)
        metrics_data["Category"][name] = float(score * 100)
        print(f"🔹 {name} Accuracy: {score * 100:.2f}%")
        if score > best_cat_score:
            best_cat_score = score
            best_cat_model = m
            best_cat_name = name
            
    print(f" WINNER FOR CATEGORY: {best_cat_name} ({best_cat_score*100:.2f}%)")
    
    # --- 2. URGENCY MODELS COMPARISON ---
    print("\n --- COMPARING EMBEDDING MODELS FOR URGENCY ---")
    best_urg_score = -1
    best_urg_model = None
    best_urg_name = ""
    
    for name, model_init in models_to_compare.items():
        m = model_init()
        m.fit(X_train, df_train['urgency'])
        preds = m.predict(X_test)
        score = accuracy_score(df_test['urgency'], preds)
        metrics_data["Urgency"][name] = float(score * 100)
        print(f"🔹 {name} Accuracy: {score * 100:.2f}%")
        if score > best_urg_score:
            best_urg_score = score
            best_urg_model = m
            best_urg_name = name
            
    print(f" WINNER FOR URGENCY: {best_urg_name} ({best_urg_score*100:.2f}%)\n")
    
    # UI Dashboard Metrics mapping JSON save
    with open(os.path.join(models_dir, "model_metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=4)
    
    cat_pipeline = {"model": best_cat_model, "model_name": f"Gemini-Embed + {best_cat_name}"}
    urg_pipeline = {"model": best_urg_model, "model_name": f"Gemini-Embed + {best_urg_name}"}
    
    joblib.dump(cat_pipeline, os.path.join(models_dir, "category_model.pkl"))
    joblib.dump(urg_pipeline, os.path.join(models_dir, "urgency_model.pkl"))
    
    print(" Best Advanced Embedding Models saved successfully!")
    return cat_pipeline, urg_pipeline

def predict_classical_ml(ticket_text: str) -> dict:
    cat_pipeline, urg_pipeline = load_ml_pipeline()
    
    X_embedding = get_gemini_embeddings([ticket_text])
    
    pred_cat = cat_pipeline["model"].predict(X_embedding)[0]
    pred_urg = urg_pipeline["model"].predict(X_embedding)[0]
    
    model_info = f"{cat_pipeline.get('model_name', 'Embedding ML')}"
    
    return {
        "category": str(pred_cat),
        "urgency": str(pred_urg),
        "model_used": model_info
    }

if __name__ == "__main__":
    train_baseline_models()