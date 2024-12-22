import numpy as np
import pandas as pd

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cross-entropy loss function
def compute_loss(y_true, y_pred):
    m = len(y_true)
    loss = -np.sum(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9)) / m
    return loss

# Logistic Regression training function (binary)
def logistic_regression_train(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    
    for epoch in range(epochs):
        linear_output = np.dot(X, weights) + bias
        y_pred = sigmoid(linear_output)
        
        dw = np.dot(X.T, (y_pred - y)) / m
        db = np.sum(y_pred - y) / m
        
        weights -= lr * dw
        bias -= lr * db
        
        if epoch % 100 == 0:
            loss = compute_loss(y, y_pred)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            
    return weights, bias

# One-vs-All approach for multi-class Logistic Regression
def one_vs_all(X, y, num_classes, lr=0.01, epochs=1000):
    m, n = X.shape
    all_weights = np.zeros((num_classes, n))
    all_biases = np.zeros(num_classes)
    
    for c in range(num_classes):
        print(f"Training for class {c}...")
        y_binary = (y == c).astype(int)
        weights, bias = logistic_regression_train(X, y_binary, lr, epochs)
        all_weights[c] = weights
        all_biases[c] = bias
    
    return all_weights, all_biases

# Prediction function
def predict_one_vs_all(X, all_weights, all_biases):
    probabilities = sigmoid(np.dot(X, all_weights.T) + all_biases)
    return np.argmax(probabilities, axis=1)

# Complete missing values for a new input
def complete_new_input(new_input, feature_columns):
    input_dict = dict(zip(feature_columns[:len(new_input)], new_input))
    complete_input = [input_dict.get(col, 0) for col in feature_columns]
    return np.array(complete_input)

# Predict new input
def predict_new_input(new_input, all_weights, all_biases, mean, std, feature_columns):
    new_input = complete_new_input(new_input, feature_columns)
    new_input = (new_input - mean) / std
    probabilities = sigmoid(np.dot(new_input, all_weights.T) + all_biases)
    predicted_class = np.argmax(probabilities)
    return predicted_class, probabilities

# Custom train-test split function
def train_test_split_custom(X, y, test_size=0.2, random_state=None):
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

# Calculate train and test accuracies
def compute_accuracies(X_train, y_train, X_test, y_test, all_weights, all_biases):
    train_predictions = predict_one_vs_all(X_train, all_weights, all_biases)
    test_predictions = predict_one_vs_all(X_test, all_weights, all_biases)

    train_accuracy = np.mean(train_predictions == y_train) * 100
    test_accuracy = np.mean(test_predictions == y_test) * 100

    return train_accuracy, test_accuracy

# Main function
def main():
    df = pd.read_csv("../data/data.csv")

    feature_columns = ["Fee", "Model Year", "Kilometer", "Fuel", "Transmission Type", "Accident ",
                       "Security hw", "# of interior equipment ", "# of exterior eq"]
    X = df[feature_columns].values
    y = df["Model"].values

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std

    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=None)

    num_classes = len(np.unique(y))
    all_weights, all_biases = one_vs_all(X_train, y_train, num_classes, lr=0.01, epochs=1000)

    train_accuracy, test_accuracy = compute_accuracies(X_train, y_train, X_test, y_test, all_weights, all_biases)
    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
