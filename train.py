import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_PATH = "data/credit_card_transactions.csv"
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("\n--- Dataset Preview ---")
print(df.head())
print("\n--- Dataset Shape ---")
print(df.shape)
print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n--- Class Distribution ---")
print(df["fraud"].value_counts())
print("\n--- Fraud Percentage ---")
print(round(df["fraud"].mean() * 100, 2), "%")

X = df.drop(columns=["fraud"])
y = df["fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Simple baseline
baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
])
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

# Main model
model = RandomForestClassifier(
    n_estimators=250,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
    min_samples_leaf=2
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1_score": float(f1_score(y_test, pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, prob)),
    "baseline_logistic_regression_f1": float(
        f1_score(y_test, baseline_pred, zero_division=0)
    )
}

print("\n--- Random Forest Results ---")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

print("\n--- Classification Report ---")
print(classification_report(
    y_test, pred, target_names=["Genuine", "Fraud"], zero_division=0
))

cm = confusion_matrix(y_test, pred)

with open("reports/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

with open("reports/classification_report.txt", "w", encoding="utf-8") as f:
    f.write(classification_report(
        y_test, pred, target_names=["Genuine", "Fraud"], zero_division=0
    ))

# Confusion matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks([0, 1], ["Genuine", "Fraud"])
plt.yticks([0, 1], ["Genuine", "Fraud"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=160)
plt.close()

# Feature importance
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(8, 5))
importance.plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("reports/feature_importance.png", dpi=160)
plt.close()

joblib.dump(model, "models/random_forest.joblib")

print("\nSaved model and reports.")
