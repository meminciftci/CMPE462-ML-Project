import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Ana çalışma fonksiyonu
def main():
    # Veri yükleme
    df = pd.read_csv("../data/data.csv")  # Kendi dosyanızı ekleyin

    # Hedef (y) ve özellikler (X)
    feature_columns = ["Fee", "Model Year", "Kilometer", "Fuel", "Transmission Type", "Accident ",
                       "Security hw", "# of interior equipment ", "# of exterior eq"]
    X = df[feature_columns].values
    y = df["Model"].values

    # Veriyi standardize etme
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std  # Normalize

    # Eğitim ve test setine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression modeli (multi-class)
    model = LogisticRegression(multi_class="ovr", max_iter=1000)  # One-vs-Rest yaklaşımı
    model.fit(X_train, y_train)

    # Eğitim ve test doğruluklarını hesapla
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions) * 100
    test_accuracy = accuracy_score(y_test, test_predictions) * 100

    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Detaylı sınıflandırma raporu
    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, test_predictions))


if __name__ == "__main__":
    main()
