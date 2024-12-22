import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, r2_score
from sklearn.preprocessing import label_binarize
import time

# 1. Load Data
data_file = "../data/data.csv"  # Our dataset file
df = pd.read_csv(data_file)

# 2. Detecting Dependent and Independent Variables
X = df.drop(columns=["Model", "Engine Capacity", "Horse Power"], axis=1)  # Feature Variables
y = df["Model"]  # Target Variable

# 3. Split Data into Test and Train Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Determining the Optimal Number of Trees with Cross Validation
n_estimators_list = [10, 50, 100, 200, 300]
cv_scores = []

for n in n_estimators_list:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

best_n = n_estimators_list[np.argmax(cv_scores)]
print(f"Best n_estimators: {best_n}, CV Accuracy: {max(cv_scores)}")

# 5. Final Model Training and Evaluation
start_time = time.time()
final_model = RandomForestClassifier(n_estimators=best_n, random_state=42)
final_model.fit(X_train, y_train)

# 6. Predictions
y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

# 7. Performance Report
print("\nTrain Set Performance Report:")
print(classification_report(y_train, y_train_pred))

print("\nTest Set Performance Report:")
print(classification_report(y_test, y_test_pred))

# 8. Model Execution Time
end_time = time.time()
print(f"Model Execution Time: {end_time - start_time} seconds")

# 9. Model Accuracy
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Set Accuracy: {accuracy}")

# 10. Precision, Recall, and F1 Scores
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"Test Set Precision: {precision}")
print(f"Test Set Recall: {recall}")
print(f"Test Set F1 Score: {f1}")

# 11. R^2 Score
r2 = r2_score(y_test, y_test_pred)
print(f"R^2 Score: {r2}")

def calculate_auroc(y_true, y_pred_proba):
    # Çok sınıflı AUROC için one-vs-rest yaklaşımı
    y_bin = label_binarize(y_true, classes=np.unique(y_true))
    return roc_auc_score(y_bin, y_pred_proba, multi_class='ovr')

y_train_pred_proba = final_model.predict_proba(X_train)
y_test_pred_proba = final_model.predict_proba(X_test)

# AUROC için probabiliteyi kullanıyoruz. Bu yüzden burada bir değişiklik yapmamıza gerek yok.
train_auroc = calculate_auroc(y_train, y_train_pred_proba)
test_auroc = calculate_auroc(y_test, y_test_pred_proba)

print(f"\nTraining Set AUROC: {train_auroc}")
print(f"Test Set AUROC: {test_auroc}")



