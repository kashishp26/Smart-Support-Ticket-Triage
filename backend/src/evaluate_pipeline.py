import os
import sys
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(backend_dir)

from baseline_ml import predict_classical_ml

def run_evaluation():
    test_path = os.path.join(backend_dir, "data", "test.csv")
    
    if not os.path.exists(test_path):
        print(f" Error: Test dataset nahi mila is path par: {test_path}")
        print(" Pehle 'python src/data_prep.py' chalakar data generate karein.")
        return

    df_test = pd.read_csv(test_path)
    print(f" Total {len(df_test)} test tickets par accuracy check ho rahi hai...\n")

    y_true_cat = df_test['category'].tolist()
    y_true_urg = df_test['urgency'].tolist()
    
    y_pred_cat = []
    y_pred_urg = []

    # Run the local ML model on each ticket.
    for text in df_test['text']:
        res = predict_classical_ml(text)
        y_pred_cat.append(res['category'])
        y_pred_urg.append(res['urgency'])

    # Print the Accuracy Metrics 
    print("==================================================")
    print(f"CATEGORY MODEL ACCURACY: {accuracy_score(y_true_cat, y_pred_cat) * 100:.2f}%")
    print("==================================================")
    print(classification_report(y_true_cat, y_pred_cat))

    print("\n==================================================")
    print(f"URGENCY MODEL ACCURACY: {accuracy_score(y_true_urg, y_pred_urg) * 100:.2f}%")
    print("==================================================")
    print(classification_report(y_true_urg, y_pred_urg))

if __name__ == "__main__":
    run_evaluation()