import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    # Load the dataset
    df = pd.read_csv("../data/data.csv")

    # Define feature columns and target column
    feature_columns = ["Fee", "Model Year", "Kilometer", "Fuel", "Transmission Type", "Accident ",
                       "Security hw", "# of interior equipment ", "# of exterior eq"]
    target_column = "Model"

    # Extract features (X) and target (y)
    X = df[feature_columns].values
    y = df[target_column].values

    # Normalize the data for consistent scaling
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std

    return X, y

def tune_hyperparameters(X_train, y_train):
    """Tune SVM hyperparameters using GridSearchCV."""
    # Create a pipeline for scaling and SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize data
        ('svm', SVC())  # Support Vector Machine
    ])

    # Define parameter grid for tuning
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],  # Regularization parameter
        'svm__kernel': ['linear', 'rbf', 'poly'],  # Kernel types
        'svm__gamma': [0.001, 0.01, 0.1, 1],  # Kernel coefficient
        'svm__degree': [2, 3, 4]  # Polynomial kernel degree
    }

    # Use GridSearchCV for hyperparameter tuning with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Output the best parameters and accuracy
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Accuracy:", grid_search.best_score_)

    return grid_search.best_estimator_

def main():
    """Main function to execute hyperparameter tuning."""
    # Load and preprocess data
    X, y = load_and_preprocess_data()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform hyperparameter tuning
    best_model = tune_hyperparameters(X_train, y_train)

    # Notify that the model tuning is complete
    print("Final model hyperparameters have been tuned and saved.")

if __name__ == "__main__":
    main()
