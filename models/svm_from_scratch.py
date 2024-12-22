import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

class LinearSoftMarginSVM:
    def __init__(self, C=1.0):
        self.C = C  # Regularization parameter

    def fit(self, X, y):
        """Train the SVM using Quadratic Programming."""
        solvers.options['show_progress'] = False
        m, n = X.shape
        y = y.astype(np.double).reshape(-1, 1)

        # Matrices for QP
        K = np.dot(X, X.T) * (y @ y.T)
        P = matrix(K)
        q = matrix(-np.ones((m, 1)))

        G_std = np.diag(-np.ones(m))  # For α_i >= 0
        G_slack = np.eye(m)           # For α_i <= C
        G = matrix(np.vstack((G_std, G_slack)))

        h_std = np.zeros(m)
        h_slack = np.ones(m) * self.C
        h = matrix(np.hstack((h_std, h_slack)))

        A = matrix(y.T)
        b = matrix(0.0)

        # Solve QP
        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution['x'])

        # Support vectors
        sv = alphas > 1e-5
        self.alphas = alphas[sv]
        self.support_vectors = X[sv]
        self.support_vector_labels = y[sv]

        self.w = np.sum((self.alphas * self.support_vector_labels.ravel()).reshape(-1, 1) * self.support_vectors, axis=0)
        self.b = np.mean(self.support_vector_labels - np.dot(self.support_vectors, self.w))

    def predict(self, X):
        """Predict class labels for given data."""
        return np.sign(np.dot(X, self.w) + self.b)

def one_vs_all_svm(X, y, unique_classes, C=1.0):
    """Train multi-class SVM using One-vs-All approach."""
    weights = []
    biases = []

    for cls in unique_classes:
        y_binary = np.where(y == cls, 1, -1)
        svm = LinearSoftMarginSVM(C=C)
        svm.fit(X, y_binary)
        weights.append(svm.w)
        biases.append(svm.b)

    return np.array(weights), np.array(biases)

def predict_one_vs_all(X, weights, biases):
    """Predict multi-class labels."""
    scores = np.dot(X, weights.T) + biases
    return np.argmax(scores, axis=1)

def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    """Custom train-test split function."""
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    split_index = int((1 - test_size) * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test

def main():
    # Load data
    df = pd.read_csv("../data/data.csv")

    # Define features and target
    feature_columns = ["Fee", "Model Year", "Kilometer", "Fuel", "Transmission Type", "Accident ",
                       "Security hw", "# of interior equipment ", "# of exterior eq"]
    target_column = "Model" 

    X = df[feature_columns].values
    y = df[target_column].values

    # Normalize data
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=27)

    # Train multi-class SVM
    weights, biases = one_vs_all_svm(X_train, y_train, unique_classes=np.unique(y))

    # Evaluate on test set
    predictions = predict_one_vs_all(X_test, weights, biases)
    test_accuracy = np.mean(predictions == [np.where(np.unique(y) == c)[0][0] for c in y_test]) * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Evaluate on train set
    predictions = predict_one_vs_all(X_train, weights, biases)
    train_accuracy = np.mean(predictions == [np.where(np.unique(y) == c)[0][0] for c in y_train]) * 100
    print(f"Train Accuracy: {train_accuracy:.2f}%")

if __name__ == "__main__":
    main()
