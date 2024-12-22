import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, adjusted_rand_score, silhouette_score
from scipy.optimize import linear_sum_assignment

def initialize_centroids(X, k):
    np.random.seed(42)
    random_indices = np.random.choice(X.shape[0], k, replace=False)
    return X[random_indices]

def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def kmeans(X, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(X, k)
    for i in range(max_iters):
        old_centroids = centroids
        labels = assign_clusters(X, centroids)
        centroids = update_centroids(X, labels, k)
        
        if np.linalg.norm(centroids - old_centroids) < tol:
            break
    return labels, centroids

data = pd.read_csv('../data/data.csv')
X = data.drop(columns=['Model', 'Engine Capacity', 'Horse Power'], axis=1).values
y_true = data['Model'].values

k = 3  
labels, centroids = kmeans(X, k)

ars = adjusted_rand_score(y_true, labels)
print(f"Adjusted Rand Score: {ars}")

sil_score = silhouette_score(X, labels)
print(f"Silhouette Score: {sil_score}")

def match_labels(y_true, y_pred):
    contingency_matrix = np.zeros((len(np.unique(y_true)), len(np.unique(y_pred))))
    for i in range(len(y_true)):
        contingency_matrix[y_true[i], y_pred[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    mapping = dict(zip(col_ind, row_ind))
    y_pred_mapped = np.array([mapping[label] for label in y_pred])
    return y_pred_mapped

labels_matched = match_labels(y_true, labels)
accuracy = accuracy_score(y_true, labels_matched)
print(f"Accuracy: {accuracy}")
