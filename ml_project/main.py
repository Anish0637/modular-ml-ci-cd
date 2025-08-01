
# main.py
from src.data_preprocessing import load_data, preprocess_data, split_data
from src.model_training import train_model, save_model
from src.model_evaluation import evaluate_model, print_metrics
import pandas as pd

def main():
    print("--- Starting ML Pipeline ----")
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = train_model(X_train, y_train)
    save_model(model, "models/iris_classifier.pkl")
    
    metrics = evaluate_model(model, X_test, y_test)
    
    print("\n--- Model Evaluation Metrics ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("--- Pipeline Finished ---")

if __name__ == "__main__":
    main
