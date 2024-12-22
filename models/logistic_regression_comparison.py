import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from logistic_regression import one_vs_all, predict_one_vs_all
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

def load_and_preprocess_data():
    df = pd.read_csv("../data/data.csv")

    feature_columns = ["Fee", "Model Year", "Kilometer", "Fuel", "Transmission Type", "Accident ",
                       "Security hw", "# of interior equipment ", "# of exterior eq"]
    X = df[feature_columns].values
    y = df["Model"].values

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std

    return X, y

def compare_performance():
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

    print("Evaluating From-Scratch Implementation...")
    start_time = time.time()
    num_classes = len(np.unique(y))
    all_weights, all_biases = one_vs_all(X_train, y_train, num_classes, lr=0.01, epochs=1000)
    training_time_scratch = time.time() - start_time

    start_time = time.time()
    predictions_scratch = predict_one_vs_all(X_test, all_weights, all_biases)
    prediction_time_scratch = time.time() - start_time

    accuracy_scratch = accuracy_score(y_test, predictions_scratch) * 100
    precision_scratch = precision_score(y_test, predictions_scratch, average="weighted") * 100
    recall_scratch = recall_score(y_test, predictions_scratch, average="weighted") * 100
    f1_scratch = f1_score(y_test, predictions_scratch, average="weighted") * 100

    y_test_binarized = label_binarize(y_test, classes=np.unique(y))
    y_prob_scratch = None
    auroc_scratch = None
    if y_prob_scratch is not None:
        auroc_scratch = roc_auc_score(y_test_binarized, y_prob_scratch, average="weighted", multi_class="ovr") * 100

    print(f"From-Scratch Implementation:")
    print(f"  Accuracy: {accuracy_scratch:.2f}%")
    print(f"  Precision: {precision_scratch:.2f}%")
    print(f"  Recall: {recall_scratch:.2f}%")
    print(f"  F1-Score: {f1_scratch:.2f}%")
    print(f"  AUROC: {auroc_scratch}")
    print(f"  Training Time: {training_time_scratch:.8f} seconds")
    print(f"  Prediction Time: {prediction_time_scratch:.8f} seconds")

    print("\nEvaluating Scikit-Learn Implementation...")
    start_time = time.time()
    model = LogisticRegression(multi_class="ovr", max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)
    training_time_sklearn = time.time() - start_time

    start_time = time.time()
    predictions_sklearn = model.predict(X_test)
    y_prob_sklearn = model.predict_proba(X_test)
    prediction_time_sklearn = time.time() - start_time

    accuracy_sklearn = accuracy_score(y_test, predictions_sklearn) * 100
    precision_sklearn = precision_score(y_test, predictions_sklearn, average="weighted") * 100
    recall_sklearn = recall_score(y_test, predictions_sklearn, average="weighted") * 100
    f1_sklearn = f1_score(y_test, predictions_sklearn, average="weighted") * 100
    auroc_sklearn = roc_auc_score(y_test_binarized, y_prob_sklearn, average="weighted", multi_class="ovr") * 100

    print(f"Scikit-Learn Implementation:")
    print(f"  Accuracy: {accuracy_sklearn:.2f}%")
    print(f"  Precision: {precision_sklearn:.2f}%")
    print(f"  Recall: {recall_sklearn:.2f}%")
    print(f"  F1-Score: {f1_sklearn:.2f}%")
    print(f"  AUROC: {auroc_sklearn:.2f}")
    print(f"  Training Time: {training_time_sklearn:.8f} seconds")
    print(f"  Prediction Time: {prediction_time_sklearn:.8f} seconds")

    print("\nPerformance Comparison:")
    print(f"{'Metric':<20}{'From-Scratch':<15}{'Scikit-Learn':<15}")
    print(f"{'Accuracy':<20}{accuracy_scratch:<15.2f}{accuracy_sklearn:<15.2f}")
    print(f"{'Precision':<20}{precision_scratch:<15.2f}{precision_sklearn:<15.2f}")
    print(f"{'Recall':<20}{recall_scratch:<15.2f}{recall_sklearn:<15.2f}")
    print(f"{'F1-Score':<20}{f1_scratch:<15.2f}{f1_sklearn:<15.2f}")
    print(f"{'AUROC':<20}{auroc_scratch or 'N/A':<15}{auroc_sklearn:<15.2f}")
    print(f"{'Training Time (s)':<20}{training_time_scratch:<15.4f}{training_time_sklearn:<15.4f}")
    print(f"{'Prediction Time (s)':<20}{prediction_time_scratch:<15.4f}{prediction_time_sklearn:<15.4f}")

if __name__ == "__main__":
    compare_performance()
