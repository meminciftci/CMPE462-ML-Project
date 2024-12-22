import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, silhouette_score
from scipy.optimize import linear_sum_assignment



data = pd.read_csv('../data/data.csv')
X = data.drop(columns=['Model', 'Engine Capacity', 'Horse Power'], axis=1).values
y_true = data['Model'].values

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
y_pred = kmeans.labels_

ars = adjusted_rand_score(y_true, y_pred)
print(f"Adjusted Rand Score: {ars}")

sil_score = silhouette_score(X, y_pred)
print(f"Silhouette Score: {sil_score}")

def match_labels(y_true, y_pred):
    contingency_matrix = np.zeros((len(np.unique(y_true)), len(np.unique(y_pred))))
    for i in range(len(y_true)):
        contingency_matrix[y_true[i], y_pred[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    mapping = dict(zip(col_ind, row_ind))
    y_pred_mapped = np.array([mapping[label] for label in y_pred])
    return y_pred_mapped

y_pred_matched = match_labels(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred_matched)
print(f"Accuracy: {accuracy}")
