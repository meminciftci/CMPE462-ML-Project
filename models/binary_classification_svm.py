import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import time

data = pd.read_csv("../data/data.csv")

features = ["Kilometer", "Fee"]
target = "Model"
X = data[features].values
y = data[target].values

binary_mask = (y == 0) | (y == 1)        
X = X[binary_mask]
y = y[binary_mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(kernel='linear', C=0.1, probability=True)  # C = 0.1

start_train = time.time()
svm.fit(X_train, y_train)
end_train = time.time()
training_time = end_train - start_train

start_test = time.time()
y_pred = svm.predict(X_test)
y_pred_proba = svm.predict_proba(X_test)[:, 1]  # Probability scores for ROC
end_test = time.time()
testing_time = end_test - start_test

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Results
metrics_results = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1,
    "AUROC": roc_auc,
    "Training Time (s)": training_time,
    "Testing Time (s)": testing_time
}

# Display results
print("Features:", features)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("AUROC:", roc_auc)
print("Training Time (s):", training_time)
print("Testing Time (s):", testing_time)
