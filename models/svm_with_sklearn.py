import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Veri yükleme ve işleme
def load_and_preprocess_data():
    df = pd.read_csv("../data/data.csv")  # Kendi veri dosyanızı sağlayın

    feature_columns = ["Fee", "Model Year", "Kilometer", "Fuel", "Transmission Type", "Accident ",
                       "Security hw", "# of interior equipment ", "# of exterior eq", "Horse Power", "Engine Capacity"]
    target_column = "Model"

    X = df[feature_columns].values
    y = df[target_column].values

    # Veriyi normalize etme
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std

    return X, y

# Linear ve non-linear SVM'leri eğitme ve değerlendirme
def train_and_evaluate_svm(X_train, X_test, y_train, y_test):
    kernels = ["linear", "poly", "rbf"]
    results = {}

    for kernel in kernels:
        print(f"Training SVM with {kernel} kernel...")

        # SVM modelini oluştur ve eğit
        model = SVC(kernel=kernel, random_state=42)
        model.fit(X_train, y_train)

        # Eğitim ve test seti üzerinde değerlendirme
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

# Ana çalışma fonksiyonu
def main():
    # Veriyi yükle ve işle
    X, y = load_and_preprocess_data()

    # Eğitim ve test setine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SVM modellerini eğit ve değerlendir
    results = train_and_evaluate_svm(X_train, X_test, y_train, y_test)

    # Sonuçları yazdır
    for kernel, metrics in results.items():
        print(f"{kernel.capitalize()} Kernel Results:")
        print(f"  Training Accuracy: {metrics['train_accuracy']:.2f}")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.2f}")

if __name__ == "__main__":
    main()
