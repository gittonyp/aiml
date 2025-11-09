import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

try:
    df = pd.read_csv("emails.csv")
except FileNotFoundError:
    print("Warning: 'emails.csv' not found. Using dummy data.")
    N = 1000
    data = {f"word_{i}": np.random.randint(0, 5, N) for i in range(50)}
    data["Prediction"] = np.random.choice([0, 1], N, p=[0.85, 0.15]) # Imbalanced
    df = pd.DataFrame(data)

df = df.select_dtypes(include=["number"]).dropna()

if "Prediction" not in df.columns:
    raise ValueError("The dataset must have a 'Prediction' column.")
    
X = df.drop(columns=["Prediction"])
y = df["Prediction"]

print("Class Distribution Before Balancing:
", y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Apply SMOTE (oversampling minority class)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("
Class Distribution After SMOTE:
", pd.Series(y_train_res).value_counts())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Train SVM Model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_scaled, y_train_res)

y_pred = svm_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("
=== SVM Email Spam Detection Results (with SMOTE) ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("
Confusion Matrix:
", cm)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Normal", "Spam"], 
            yticklabels=["Normal", "Spam"])
plt.title("SVM Confusion Matrix (SMOTE)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()