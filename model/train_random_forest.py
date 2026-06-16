import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("dataset_fitur.csv")

print("Jumlah data:", len(df))
print(df["label"].value_counts())

# Hapus kolom filename dan label dari fitur
X = df.drop(["filename", "label"], axis=1)
y = df["label"]

# Kalau data masih sedikit, test_size bisa 0.25
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=None
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAkurasi:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, "model_random_forest_ikan.pkl")

print("\nModel berhasil disimpan sebagai model_random_forest_ikan.pkl")
