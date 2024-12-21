import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

class LinearSoftMarginSVM:
    def __init__(self, C=1.0):
        self.C = C  # Regularization parameter

    def fit(self, X, y):
        """
        SVM'i Quadratic Programming (QP) ile eğitir.
        """
        solvers.options['show_progress'] = False
        m, n = X.shape
        y = y.astype(np.double).reshape(-1, 1)  # Sınıf etiketlerini sütun vektörü yap

        # QP için matrisler
        K = np.dot(X, X.T) * (y @ y.T)
        P = matrix(K)
        q = matrix(-np.ones((m, 1)))

        G_std = np.diag(-np.ones(m))  # α_i >= 0 için
        G_slack = np.eye(m)           # α_i <= C için
        G = matrix(np.vstack((G_std, G_slack)))

        h_std = np.zeros(m)           # α_i >= 0 için
        h_slack = np.ones(m) * self.C # α_i <= C için
        h = matrix(np.hstack((h_std, h_slack)))

        A = matrix(y.T)
        b = matrix(0.0)

        # QP çözümü
        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution['x'])

        # Destek vektörleri
        sv = alphas > 1e-5
        self.alphas = alphas[sv]
        self.support_vectors = X[sv]
        self.support_vector_labels = y[sv]

        self.w = np.sum((self.alphas * self.support_vector_labels.ravel()).reshape(-1, 1) * self.support_vectors, axis=0)
        self.b = np.mean(self.support_vector_labels - np.dot(self.support_vectors, self.w))

    def predict(self, X):
        """
        Yeni veri noktalarının sınıflarını tahmin eder.
        """
        return np.sign(np.dot(X, self.w) + self.b)

def one_vs_all_svm(X, y, unique_classes, C=1.0):
    """
    Multi-Class SVM'i One-vs-All yöntemi ile uygular.
    """
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
    """
    Multi-class tahmin yapar.
    """
    scores = np.dot(X, weights.T) + biases
    return np.argmax(scores, axis=1)

def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    """
    Eğitim ve test setini rastgele şekilde ayırır.
    """
    if random_state is not None:
        np.random.seed(random_state)  # Rastgeleliği sabitlemek için

    # Verilerin sırasını karıştır
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    # Karışık veriyi kullanarak X ve y'yi yeniden sıralama
    X = X[indices]
    y = y[indices]

    # Eğitim ve test seti için sınır belirleme
    split_index = int((1 - test_size) * len(X))

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test


def main():
    # Veri yükleme
    df = pd.read_csv("../data/data.csv")  # Kendi dosyanızı ekleyin

    # Özellik sütunları ve hedef sütun
    feature_columns = ["Fee", "Model Year", "Kilometer", "Fuel", "Transmission Type", "Accident ",
                       "Security hw", "# of interior equipment ", "# of exterior eq", "Horse Power", "Engine Capacity"]
    target_column = "Model" 

    X = df[feature_columns].values
    y = df[target_column].values

    # Benzersiz sınıfları al
    unique_classes = np.unique(y)

    # Veriyi normalize etme
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std

    # Eğitim ve test setine ayırma
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)

    # Multi-Class SVM eğitimi
    weights, biases = one_vs_all_svm(X_train, y_train, unique_classes)

    # Test setinde tahmin yapma
    predictions = predict_one_vs_all(X_test, weights, biases)
    accuracy = np.mean(predictions == [np.where(unique_classes == c)[0][0] for c in y_test]) * 100
        
    print(f"Test Accuracy: {accuracy:.2f}%")
    

if __name__ == "__main__":
    main()
