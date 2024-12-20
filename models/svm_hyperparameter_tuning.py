import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Veri yükleme ve işleme
def load_and_preprocess_data():
    df = pd.read_csv("../data/data.csv")  # Veri dosyasını sağlayın

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

# Hyperparameter tuning için GridSearchCV kullanımı
def tune_hyperparameters(X_train, y_train):
    # Pipeline: Standartlaştırma ve SVM modeli
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC())
    ])

    # Hyperparameter aralıkları
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__kernel': ['linear', 'rbf', 'poly'],
        'svm__gamma': [0.001, 0.01, 0.1, 1],
        'svm__degree': [2, 3, 4]  # Sadece poly kernel için
    }

    # GridSearchCV ile 5-fold cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # En iyi parametreler ve skor
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Accuracy:", grid_search.best_score_)

    return grid_search.best_estimator_

# Ana çalışma fonksiyonu
def main():
    # Veriyi yükle ve işle
    X, y = load_and_preprocess_data()

    # Eğitim ve test setine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning
    best_model = tune_hyperparameters(X_train, y_train)

    # En iyi parametreleri yazdır
    print("Final model hyperparameters have been tuned and saved.")

if __name__ == "__main__":
    main()
