import numpy as np
import pandas as pd
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from svm_from_scratch import one_vs_all_svm, predict_one_vs_all

def load_and_preprocess_data(exclude_columns=None):
    """Load and preprocess data, allowing exclusion of specific features."""
    df = pd.read_csv("../data/data.csv")

    # All feature columns
    all_feature_columns = [
        "Fee", "Model Year", "Kilometer", "Fuel", "Transmission Type",
        "Accident ", "Security hw", "# of interior equipment ",
        "# of exterior eq", "Horse Power", "Engine Capacity"
    ]

    # Exclude specified columns if provided
    if exclude_columns is None:
        exclude_columns = []
    feature_columns = [col for col in all_feature_columns if col not in exclude_columns]

    print(f"Using features: {feature_columns}")

    # Extract features and target variable
    X = df[feature_columns].values
    y = df["Model"].values

    # Normalize features
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std

    return X, y

def measure_training_time(model_function, *args, **kwargs):
    """Measure the training time of a model."""
    start_time = time.time()
    model = model_function(*args, **kwargs)
    training_time = time.time() - start_time
    return model, training_time

def measure_prediction_time(model, X_test):
    """Measure the prediction time of a model."""
    start_time = time.time()
    if hasattr(model, 'predict'):
        predictions = model.predict(X_test)
    else:
        predictions = predict_one_vs_all(X_test, model['weights'], model['biases'])
    prediction_time = time.time() - start_time
    return predictions, prediction_time

def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics for a model."""
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average="weighted") * 100
    recall = recall_score(y_true, y_pred, average="weighted") * 100
    f1 = f1_score(y_true, y_pred, average="weighted") * 100
    return accuracy, precision, recall, f1

def compare_models(X_train, X_test, y_train, y_test):
    """Compare models in terms of runtime and performance metrics."""
    results = []

    # Linear Soft-Margin SVM (from scratch)
    print("Evaluating Linear Soft-Margin SVM (from scratch)...")
    start_time = time.time()
    unique_classes = np.unique(y_train)
    weights, biases = one_vs_all_svm(X_train, y_train, unique_classes, C=1.0)
    training_time = time.time() - start_time
    y_pred, prediction_time = measure_prediction_time({"weights": weights, "biases": biases}, X_test)
    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)

    results.append({
        "model": "Linear Soft-Margin SVM",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "training_time": training_time,
        "prediction_time": prediction_time
    })

    # Scikit-learn Linear Kernel SVM
    print("Evaluating Scikit-learn Linear Kernel SVM...")
    start_time = time.time()
    model = SVC(kernel="linear", random_state=42, C=1.0)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    y_pred, prediction_time = measure_prediction_time(model, X_test)
    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)

    results.append({
        "model": "Scikit-learn Linear SVM",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "training_time": training_time,
        "prediction_time": prediction_time
    })

    # Scikit-learn Polynomial Kernel SVM
    print("Evaluating Scikit-learn Polynomial Kernel SVM...")
    start_time = time.time()
    model = SVC(kernel="poly", degree=3, random_state=42, C=1.0)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    y_pred, prediction_time = measure_prediction_time(model, X_test)
    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)

    results.append({
        "model": "Scikit-learn Polynomial SVM",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "training_time": training_time,
        "prediction_time": prediction_time
    })

    # Scikit-learn RBF Kernel SVM
    print("Evaluating Scikit-learn RBF Kernel SVM...")
    start_time = time.time()
    model = SVC(kernel="rbf", random_state=42, C=1.0)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    y_pred, prediction_time = measure_prediction_time(model, X_test)
    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)

    results.append({
        "model": "Scikit-learn RBF SVM",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "training_time": training_time,
        "prediction_time": prediction_time
    })

    # Return results as DataFrame
    return pd.DataFrame(results)

def main():
    # Specify columns to exclude
    exclude_columns = ["Horse Power", "Engine Capacity"]

    # Load and preprocess data
    X, y = load_and_preprocess_data(exclude_columns=exclude_columns)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Compare models
    comparison_results = compare_models(X_train, X_test, y_train, y_test)

    # Print and save results
    print(comparison_results)
    comparison_results.to_csv("../results/svm_comparison_results.csv", index=False)
    print("\nComparison results have been saved to 'svm_comparison_results.csv'.")

if __name__ == "__main__":
    main()
