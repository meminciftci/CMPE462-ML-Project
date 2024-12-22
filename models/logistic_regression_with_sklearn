import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def main():
    df = pd.read_csv("../data/data.csv")

    feature_columns = ["Fee", "Model Year", "Kilometer", "Fuel", "Transmission Type", "Accident ",
                       "Security hw", "# of interior equipment ", "# of exterior eq"]
    X = df[feature_columns].values
    y = df["Model"].values

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    model = LogisticRegression(multi_class="ovr", max_iter=1000)
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions) * 100
    test_accuracy = accuracy_score(y_test, test_predictions) * 100

    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, test_predictions))

if __name__ == "__main__":
    main()
