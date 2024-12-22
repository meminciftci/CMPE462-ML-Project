import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from svm_from_scratch import LinearSoftMarginSVM, one_vs_all_svm, predict_one_vs_all

# Evaluation function
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    print(f"\nEvaluating Model: {model_name}")

    if hasattr(model, 'predict'):
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    else:
        y_train_pred = predict_one_vs_all(X_train, model['weights'], model['biases'])
        y_test_pred = predict_one_vs_all(X_test, model['weights'], model['biases'])

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))

    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return {
        "model_name": model_name,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "classification_report": classification_report(y_test, y_test_pred, output_dict=True),
        "confusion_matrix": cm
    }

# Functions for training SVM models
def train_linear_svm(X_train, y_train):
    model = SVC(kernel="linear", random_state=42)
    model.fit(X_train, y_train)
    return model

def train_poly_svm(X_train, y_train, degree=3):
    model = SVC(kernel="poly", degree=degree, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_rbf_svm(X_train, y_train):
    model = SVC(kernel="rbf", random_state=42)
    model.fit(X_train, y_train)
    return model

# Main function
def main():
    from svm_with_sklearn import load_and_preprocess_data

    X, y = load_and_preprocess_data()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []

    # Linear Soft-Margin SVM (custom implementation)
    print("Training Linear Soft-Margin SVM from scratch...")
    unique_classes = np.unique(y_train)
    weights, biases = one_vs_all_svm(X_train, y_train, unique_classes, C=1.0)
    results.append(evaluate_model({"weights": weights, "biases": biases}, X_train, X_test, y_train, y_test, model_name="Linear Soft-Margin SVM"))

    # Scikit-learn Linear Kernel SVM
    sklearn_linear_model = train_linear_svm(X_train, y_train)
    results.append(evaluate_model(sklearn_linear_model, X_train, X_test, y_train, y_test, model_name="Scikit-learn Linear SVM"))

    # Scikit-learn Polynomial Kernel SVM
    sklearn_poly_model = train_poly_svm(X_train, y_train, degree=3)
    results.append(evaluate_model(sklearn_poly_model, X_train, X_test, y_train, y_test, model_name="Scikit-learn Polynomial SVM"))

    # Scikit-learn RBF Kernel SVM
    sklearn_rbf_model = train_rbf_svm(X_train, y_train)
    results.append(evaluate_model(sklearn_rbf_model, X_train, X_test, y_train, y_test, model_name="Scikit-learn RBF SVM"))

    metrics_df = pd.DataFrame([{k: v for k, v in res.items() if k != "confusion_matrix"} for res in results])
    metrics_df.to_csv("../results/svm_evaluation_results.csv", index=False)
    print("\nEvaluation results have been saved to 'svm_evaluation_results.csv'.")

if __name__ == "__main__":
    main()
