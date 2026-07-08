import os
import re
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. REAL-WORLD CORPORATE DATA SNIPPETS (~200 Rows Simulation)
real_world_samples = [
    # Billing
    {"text": "Hey finance team, I noticed an unrecognized line item on invoice #4421. Can you double check?", "category": "Billing", "urgency": "High"},
    {"text": "My subscription auto-renewed but I meant to cancel it last week. Is a refund possible?", "category": "Billing", "urgency": "High"},
    {"text": "Payment gateway threw a 502 error during checkout, but my bank account shows money deducted.", "category": "Billing", "urgency": "Critical"},
    {"text": "Where can I download the annual VAT invoices for our corporate tax filing?", "category": "Billing", "urgency": "Low"},
    # Technical
    {"text": "The API endpoint /v1/data is throwing a timeout error consistently since the latest deployment.", "category": "Technical", "urgency": "Critical"},
    {"text": "Getting a blank white screen on the mobile app after entering the OTP dashboard.", "category": "Technical", "urgency": "High"},
    {"text": "Database connection pooling issues are causing high latency on our user login routing.", "category": "Technical", "urgency": "Critical"},
    {"text": "The integration with Slack webhook is broken; notifications are not triggering.", "category": "Technical", "urgency": "Medium"},
    # Account
    {"text": "I need to transfer ownership of this workspace to my colleague as I am leaving the organization.", "category": "Account", "urgency": "Medium"},
    {"text": "Two-factor authentication (2FA) is locked out because I lost my authenticator device.", "category": "Account", "urgency": "High"},
    {"text": "Can we change our registered corporate email from admin@oldcompany.com to support@newcompany.com?", "category": "Account", "urgency": "Medium"},
    {"text": "Please delete all my personal data and close this account under GDPR compliance.", "category": "Account", "urgency": "Low"},
    # Other
    {"text": "Are you guys planning to launch a dark mode theme for the desktop application anytime soon?", "category": "Other", "urgency": "Low"},
    {"text": "Just wanted to say your support team is doing a fantastic job. Keep it up!", "category": "Other", "urgency": "Low"},
    {"text": "Where can I find the official product documentation for advanced webhook configurations?", "category": "Other", "urgency": "Low"}
]

# 2. SYNTHETIC TEMPLATES
templates = {
    'Billing': ["Please check my invoice for {amount}.", "Charged twice for subscription id {id}.", "Need a refund for duplicate billing."],
    'Technical': ["App keeps crashing on window startup.", "Error code {id} on the main server dashboard.", "System is running extremely slow today."],
    'Account': ["Reset link for password not received.", "Locked out of my profile account.", "How to update my profile security settings?"],
    'Other': ["What are the platform pricing plans?", "Do you support integration with third party tools?", "Send me marketing newsletter details."]
}

def generate_hybrid_dataset():
    print("⏳ Day 1: Creating Hybrid Dataset (with Urgency Columns)...")
    
    final_data = []
    
    # Step A: Add Real-World corporate samples
    for _ in range(14): 
        for sample in real_world_samples:
            final_data.append({
                "text": sample["text"],
                "category": sample["category"],
                "urgency": sample["urgency"] # Added Urgency here
            })
            
    # Step B: Add Synthetic templates
    categories = ['Billing', 'Technical', 'Account', 'Other']
    urgencies = ['Low', 'Medium', 'High', 'Critical']
    
    for _ in range(160):
        cat = random.choice(categories)
        temp = random.choice(templates[cat])
        filled_text = temp.format(amount=f"${random.randint(10, 500)}", id=f"ERR-{random.randint(1000, 9999)}")
        
        # Implement logic to keep the mapping slightly dynamic.
        if cat == 'Technical':
            urg = random.choice(['High', 'Critical'])
        elif cat == 'Billing':
            urg = random.choice(['Medium', 'High'])
        else:
            urg = random.choice(['Low', 'Medium'])
            
        final_data.append({
            "text": filled_text, 
            "category": cat,
            "urgency": urg # Added Urgency here
        })
        
    df = pd.DataFrame(final_data)
    
    # 3. TEXT CLEANING
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # 4. STRATIFIED TRAIN-TEST SPLIT
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['category']
    )
    
    os.makedirs('data', exist_ok=True)
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print(f" Hybrid Dataset Fixed & Ready with 'urgency' column!")

if __name__ == "__main__":
    generate_hybrid_dataset()