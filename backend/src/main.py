import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

from baseline_ml import predict_classical_ml
from llm_classifier import call_llm_classifier

app = FastAPI(title="Smart Support Ticket Triage System API")

class TicketRequest(BaseModel):
    text: str
    approach: str

@app.post("/predict")
def predict_ticket(request: TicketRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Ticket text cannot be empty!")
        
    print(f" Received Ticket: {request.text[:30]}... | Approach: {request.approach}")
    
    if "Classical ML" in request.approach:
        result = predict_classical_ml(request.text)
    elif "Few-Shot" in request.approach:
        result = call_llm_classifier(request.text, approach="few-shot")
        result["model_used"] = "Gemini 2.5 Flash (Few-Shot)"
    else:
        result = call_llm_classifier(request.text, approach="zero-shot")
        result["model_used"] = "Gemini 2.5 Flash (Zero-Shot)"
        
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)