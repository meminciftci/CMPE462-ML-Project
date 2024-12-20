# Gerekli kütüphaneleri yükle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Veri Yükleme
data_file = "final_data.csv"  # CSV dosyanızın adı
df = pd.read_csv(data_file)

# 2. Bağımlı ve Bağımsız Değişkenleri Ayırma
X = df.drop(columns=["Model"])  # Feature'lar (Bağımsız değişkenler)
y = df["Model"]  # Hedef değişken (Brand)

# 3. Eğitim ve Test Verilerini Ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Random Forest Modeli Oluşturma
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# 5. Model Performansını Değerlendirme
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# 6. Feature Importance Analizi
feature_importances = rf_model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# 7. Sonuçları Görselleştirme
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance Using Random Forest")
plt.gca().invert_yaxis()
plt.show()

# 8. Önem Derecesine Göre Sıralanmış Feature'lar
print(importance_df)
