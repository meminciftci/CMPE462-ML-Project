import numpy as np
import pandas as pd
import time
from sklearn.svm import SVC
from svm_from_scratch import LinearSoftMarginSVM, one_vs_all_svm, predict_one_vs_all
from svm_with_sklearn import load_and_preprocess_data

# Çalışma sürelerini karşılaştırma için fonksiyonlar
def measure_training_time(model_function, *args, **kwargs):
    """Bir modelin eğitim süresini ölçer."""
    start_time = time.time()
    model = model_function(*args, **kwargs)
    training_time = time.time() - start_time
    return model, training_time

def measure_prediction_time(model, X_test):
    """Bir modelin tahmin süresini ölçer."""
    start_time = time.time()
    if hasattr(model, 'predict'):
        model.predict(X_test)
    else:
        predict_one_vs_all(X_test, model['weights'], model['biases'])
    prediction_time = time.time() - start_time
    return prediction_time

def compare_runtime(X_train, X_test, y_train, y_test):
    """Modellerin eğitim ve tahmin sürelerini karşılaştırır."""
    results = []

    # Linear Soft-Margin SVM (from scratch)
    print("Measuring runtime for Linear Soft-Margin SVM...")
    unique_classes = np.unique(y_train)
    start_time = time.time()
    weights, biases = one_vs_all_svm(X_train, y_train, unique_classes, C=1.0)
    training_time = time.time() - start_time
    prediction_time = measure_prediction_time({"weights": weights, "biases": biases}, X_test)
    results.append({"model": "Linear Soft-Margin SVM", "training_time": training_time, "prediction_time": prediction_time})

    # Scikit-learn Linear Kernel SVM
    print("Measuring runtime for Scikit-learn Linear Kernel SVM...")
    start_time = time.time()
    model = SVC(kernel="linear", random_state=42, C=1.0)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    prediction_time = measure_prediction_time(model, X_test)
    results.append({"model": "Scikit-learn Linear SVM", "training_time": training_time, "prediction_time": prediction_time})

    # Scikit-learn Polynomial Kernel SVM
    print("Measuring runtime for Scikit-learn Polynomial Kernel SVM...")
    start_time = time.time()
    model = SVC(kernel="poly", degree=3, random_state=42, C=1.0)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    prediction_time = measure_prediction_time(model, X_test)
    results.append({"model": "Scikit-learn Polynomial SVM", "training_time": training_time, "prediction_time": prediction_time})

    # Scikit-learn RBF Kernel SVM
    print("Measuring runtime for Scikit-learn RBF Kernel SVM...")
    start_time = time.time()
    model = SVC(kernel="rbf", random_state=42, C=1.0)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    prediction_time = measure_prediction_time(model, X_test)
    results.append({"model": "Scikit-learn RBF SVM", "training_time": training_time, "prediction_time": prediction_time})

    # Sonuçları DataFrame olarak döndür
    return pd.DataFrame(results)

def main():
    # Veriyi yükle ve işle
    X, y = load_and_preprocess_data()

    # Eğitim ve test setine ayırma
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Çalışma sürelerini karşılaştır
    runtime_results = compare_runtime(X_train, X_test, y_train, y_test)

    # Sonuçları yazdır ve kaydet
    print(runtime_results)
    runtime_results.to_csv("../results/svm_runtime_comparison_results.csv", index=False)
    print("\nRuntime comparison results have been saved to 'svm_runtime_comparison_results.csv'.")

if __name__ == "__main__":
    main()
