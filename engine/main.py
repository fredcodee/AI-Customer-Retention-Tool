import sys
import os
from engine.preprocess import load_data
from engine.loadData import split_data 
from engine.train import train_and_evaluate_models, optimize_hyperparameters, evaluate_models
import json



# Main Workflow
if __name__ == "__main__":
    # Filepath to the dataset
    filepath = os.path.abspath ('./engine/data/synthetic_saas_users.csv')

    # Load and preprocess data
    data = load_data(filepath)

    # Split data
    X_train, X_test, y_train, y_test = split_data(data)

    # # Train and evaluate models
    # results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
     # Optimize hyperparameters
    optimized_models = optimize_hyperparameters(X_train, y_train)
    
     # Evaluate models
    results = evaluate_models(optimized_models, X_test, y_test)
    

    # Display results
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        print(f"Accuracy: {metrics['accuracy']}")
        print("Classification Report:\n", metrics['classification_report'])
        print("Confusion Matrix:\n", metrics['confusion_matrix'])
