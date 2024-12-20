import numpy as np
import pandas as pd

# Sigmoid fonksiyonu
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cross-entropy loss fonksiyonu
def compute_loss(y_true, y_pred):
    m = len(y_true)
    loss = -np.sum(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9)) / m
    return loss

# Logistic Regression eğitim fonksiyonu (binary için)
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

# One-vs-All yaklaşımıyla multi-class Logistic Regression
def one_vs_all(X, y, num_classes, lr=0.01, epochs=1000):
    m, n = X.shape
    all_weights = np.zeros((num_classes, n))
    all_biases = np.zeros(num_classes)
    
    for c in range(num_classes):
        print(f"Training for class {c}...")
        y_binary = (y == c).astype(int)  # Current class vs others
        weights, bias = logistic_regression_train(X, y_binary, lr, epochs)
        all_weights[c] = weights
        all_biases[c] = bias
    
    return all_weights, all_biases

# Tahmin fonksiyonu
def predict_one_vs_all(X, all_weights, all_biases):
    probabilities = sigmoid(np.dot(X, all_weights.T) + all_biases)
    return np.argmax(probabilities, axis=1)

# Yeni girdiyi tamamlayıcı fonksiyon
def complete_new_input(new_input, feature_columns):
    input_dict = dict(zip(feature_columns[:len(new_input)], new_input))
    complete_input = [input_dict.get(col, 0) for col in feature_columns]
    return np.array(complete_input)

# Tahmin fonksiyonu
def predict_new_input(new_input, all_weights, all_biases, mean, std, feature_columns):
    # 1. Yeni girdiyi sütun sırasına göre tamamla
    new_input = complete_new_input(new_input, feature_columns)
    
    # 2. Yeni girdiyi normalize et
    new_input = (new_input - mean) / std
    
    # 3. Tahmin et
    probabilities = sigmoid(np.dot(new_input, all_weights.T) + all_biases)
    predicted_class = np.argmax(probabilities)
    
    return predicted_class, probabilities

# Eğitim ve test setini ayırma fonksiyonu
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

# Ana çalışma fonksiyonu
def main():
    # Veri yükleme
    df = pd.read_csv("../data/data.csv")  # Kendi dosyanızı ekleyin
    
    # Hedef (y) ve özellikler (X)
    feature_columns = ["Fee", "Model Year", "Kilometer", "Fuel", "Transmission Type", "Accident ",
                       "Security hw", "# of interior equipment ", "# of exterior eq", "Horse Power", "Engine Capacity"]
    X = df[feature_columns].values
    y = df["Model"].values
    
    # Veriyi standardize etme
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std  # Normalize

    # Eğitim ve test setine ayırma
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)

    # Multi-class Logistic Regression eğitimi
    num_classes = len(np.unique(y))
    all_weights, all_biases = one_vs_all(X_train, y_train, num_classes, lr=0.01, epochs=1000)

    # Test setinde tahmin yapma
    predictions = predict_one_vs_all(X_test, all_weights, all_biases)
    accuracy = np.mean(predictions == y_test) * 100
    print(f"Accuracy on test set: {accuracy:.2f}%")

    # Yeni bir girdi
    new_input = np.array([649000, 2022, 78000, 0, 1, 1, 8, 6, 3, 72, 1.2])
    
    # Tahmini al
    predicted_class, probabilities = predict_new_input(new_input, all_weights, all_biases, mean, std, feature_columns)
    print(f"Tahmin edilen sınıf: {predicted_class}")
    print(f"Olasılıklar: {probabilities}")

if __name__ == "__main__":
    main()
