
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Function to execute Random Forest analysis with different feature sets
def run_random_forest_analysis(data_file, drop_columns, analysis_number):
    print(f"--- Analysis {analysis_number} ---")
    
    # Load dataset
    df = pd.read_csv(data_file)

    # Split features and target
    X = df.drop(columns=["Model"] + drop_columns)  # Features
    y = df["Model"]  # Target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate model performance
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}\n")

    # Analyze feature importance
    feature_importances = rf_model.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"], importance_df["Importance"], color="blue")
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title(f"Feature Importance (Analysis {analysis_number})")
    plt.gca().invert_yaxis()
    plt.show()

    # Display features sorted by importance
    print("Feature Importance:")
    print(importance_df)
    print("\n")

# Define dataset and columns to drop for each analysis
data_file = "../data/data.csv"
analyses = [
    ([], 1),  # Use all features
    (["Engine Capacity"], 2),  # Drop 'Engine Capacity'
    (["Engine Capacity", "Horse Power"], 3)  # Drop 'Engine Capacity' and 'Horsepower'
]

# Run analyses
for drop_columns, analysis_number in analyses:
    run_random_forest_analysis(data_file, drop_columns, analysis_number)
