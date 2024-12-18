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

# Ana çalışma fonksiyonu
def main():
    # Veri yükleme (örnek CSV dosyası, sen kendi dosyanı kullanabilirsin)
    df = pd.read_csv("../data/car_data.csv")  # Dosyanın adını buraya ekle
    
    # Hedef (y) ve özellikler (X)
    X = df.drop(columns=["Model"]).values  # "Model" hedef sütunun adı
    y = df["Model"].values
    
    # Veriyi standardize etme
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Multi-class Logistic Regression eğitimi
    num_classes = len(np.unique(y))
    all_weights, all_biases = one_vs_all(X, y, num_classes, lr=0.01, epochs=1000)
    
    # Tahmin ve değerlendirme
    predictions = predict_one_vs_all(X, all_weights, all_biases)
    accuracy = np.mean(predictions == y) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    
if __name__ == "__main__":
    main()
