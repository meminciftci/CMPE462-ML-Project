import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    df = pd.read_csv("../data/data.csv")

    # Define feature columns and target column
    feature_columns = ["Fee", "Model Year", "Kilometer", "Fuel", "Transmission Type", "Accident ",
                       "Security hw", "# of interior equipment ", "# of exterior eq"]
    target_column = "Model"

    # Extract features (X) and target (y)
    X = df[feature_columns].values
    y = df[target_column].values

    # Normalize the data for consistent scaling
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std

    return X, y

def train_and_evaluate_svm(X_train, X_test, y_train, y_test):
    """Train and evaluate SVM models with different kernels."""
    kernels = ["linear", "poly", "rbf"]
    results = {}

    for kernel in kernels:
        print(f"Training SVM with {kernel} kernel...")

        # Train the SVM model
        model = SVC(kernel=kernel, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate on training and test data
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        print(f"{kernel.capitalize()} Kernel - Training Accuracy: {train_accuracy:.2f}")
        print(f"{kernel.capitalize()} Kernel - Test Accuracy: {test_accuracy:.2f}")
        print(classification_report(y_test, y_test_pred))

        results[kernel] = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy
        }

    return results

def main():
    """Main function to train and evaluate SVM models."""
    # Load and preprocess data
    X, y = load_and_preprocess_data()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate SVM models
    results = train_and_evaluate_svm(X_train, X_test, y_train, y_test)

    # Print results
    for kernel, metrics in results.items():
        print(f"{kernel.capitalize()} Kernel Results:")
        print(f"  Training Accuracy: {metrics['train_accuracy']:.2f}")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.2f}")

if __name__ == "__main__":
    main()