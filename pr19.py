import pandas as pd
import numpy as np

try:
    df = pd.read_csv("emails.csv")
except FileNotFoundError:
    print("Warning: 'emails.csv' not found. Using dummy data.")
    N = 1000
    data = {f"word_{i}": np.random.randint(0, 5, N) for i in range(50)}
    data["Prediction"] = np.random.choice([0, 1], N, p=[0.7, 0.3])
    df = pd.DataFrame(data)

df = df.select_dtypes(include=["number"]).dropna()

if "Prediction" not in df.columns:
    raise ValueError("The dataset must have a 'Prediction' column.")

X = df.drop(columns=["Prediction"]).values
y = df["Prediction"].values

# SVM requires labels -1 and 1
y = np.where(y == 1, 1, -1)

# Standardize
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)


split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

class FastSVM:
    def __init__(self, lr=0.0001, lambda_param=0.01, epochs=300):
        self.lr = lr
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):
            margin = y * (np.dot(X, self.w) - self.b)
            misclassified = margin < 1
            
            # Hinge Loss Gradient
            dw = (self.lambda_param * self.w) - np.dot(X[misclassified].T, y[misclassified])
            db = -np.sum(y[misclassified])

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)

model = FastSVM(lr=0.00001, lambda_param=0.01, epochs=500)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

tp = np.sum((y_test == 1) & (y_pred == 1))
tn = np.sum((y_test == -1) & (y_pred == -1))
fp = np.sum((y_test == -1) & (y_pred == 1))
fn = np.sum((y_test == 1) & (y_pred == -1))

accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
precision = tp / (tp + fp + 1e-6)
recall = tp / (tp + fn + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)

print("=== SVM (From Scratch) Email Spam Detection ===")
print(f"Confusion Matrix:
[[TP={tp}, FP={fp}], [FN={fn}, TN={tn}]]")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")