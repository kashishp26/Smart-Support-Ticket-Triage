import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load the Gemini API key from the `.env` file.
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini Client 
if api_key and "YourActual" not in api_key:
    client = genai.Client(api_key=api_key)
else:
    client = None

def call_llm_classifier(ticket_text, approach="few-shot"):
    """
    Real Google Gemini API call to classify the support ticket for FREE.
    """
   # Use a fallback simulation if the API key is not set.
    if not client:
        print(" Gemini API Key nahi mili. Fallback mode active.")
        text = ticket_text.lower()
        if "refund" in text or "charge" in text:
            return {"category": "Billing", "urgency": "High"}
        return {"category": "Other", "urgency": "Low"}

    # 2. PROMPT DESIGN (Strictly asking for JSON output)
    system_prompt = (
        "You are an expert customer support triage bot for a tech organization. "
        "Your task is to classify incoming tickets strictly into one of these categories: [Billing, Technical, Account, Other] "
        "and assign an urgency level: [Low, Medium, High, Critical].\n\n"
        "STRICT CLASSIFICATION RULES:\n"
        "1. If the text mentions servers, endpoints, databases, crashes, freezes, blank screens, APIs, integration, timeouts, or code errors -> ALWAYS classify as 'Technical' and urgency as 'Critical' or 'High'.\n"
        "2. If the text mentions invoices, double charge, money, payments, refunds, billing, or corporate tax -> ALWAYS classify as 'Billing'.\n"
        "3. If the text mentions 2FA, password resets, login lockouts, account deletion, or transferring workspace ownership -> ALWAYS classify as 'Account'.\n\n"
        "You must return the response strictly as a single JSON object with keys 'category' and 'urgency'."
    )
    
    few_shot_context = (
        "Examples:\n"
        "Ticket: 'I was charged twice on my card.' -> {\"category\": \"Billing\", \"urgency\": \"High\"}\n"
        "Ticket: 'The screen goes blank when clicking save.' -> {\"category\": \"Technical\", \"urgency\": \"Critical\"}\n\n"
    )
    
    user_content = f"Ticket Text: '{ticket_text}'"
    if approach == "few-shot":
        user_content = few_shot_context + user_content

    try:
        # 3. REAL GEMINI API CALL (Using fast & free gemini-2.5-flash model)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json", # Strict JSON structure 
                temperature=0.0
            ),
        )
        
        # Parse the Output 
        result_json = json.loads(response.text)
        return result_json

    except Exception as e:
        print(f" Gemini API Error: {e}. Switching to fallback.")
        return {"category": "Other", "urgency": "Medium"}

if __name__ == "__main__":
    print("Testing Real Gemini LLM Classifier locally...")
    sample_res = call_llm_classifier("My app screen freezes during user login page.")
    print("Real Gemini Output:", sample_res)