import pandas as pd
import numpy as np

try:
    df = pd.read_csv("performance.csv").dropna()
except FileNotFoundError:
    print("Warning: 'performance.csv' not found. Using dummy data.")
    N = 100
    data = {
        "Study_Hours_per_Week": np.random.rand(N) * 15 + 1,
        "Attendance_Rate": np.random.rand(N) * 30 + 70,
        "Internal_Scores": np.random.rand(N) * 40 + 60,
        "Pass_Fail": np.random.choice(["Pass", "Fail"], N)
    }
    df = pd.DataFrame(data)

# Ensure required columns exist
required_cols = ["Study_Hours_per_Week", "Attendance_Rate", "Internal_Scores", "Pass_Fail"]
if not all(col in df.columns for col in required_cols):
    raise ValueError("CSV missing required columns.")

X = df[["Study_Hours_per_Week", "Attendance_Rate", "Internal_Scores"]].values.astype(float)
y = np.where(df["Pass_Fail"] == "Pass", 1, -1)

# Standardize
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

def polynomial_kernel(X1, X2, degree=2, coef0=1):
    return np.power(np.dot(X1, X2.T) + coef0, degree)

class PolynomialSVM:
    def __init__(self, C=1.0, lr=0.001, n_iters=300, degree=2, coef0=1):
        self.C = C
        self.lr = lr
        self.n_iters = n_iters
        self.degree = degree
        self.coef0 = coef0
        self.alpha = None
        self.b = None
        self.X = None
        self.y = None

    def fit(self, X, y):
        n = X.shape[0]
        self.alpha = np.zeros(n)
        self.b = 0
        K = polynomial_kernel(X, X, self.degree, self.coef0)

        for _ in range(self.n_iters):
            margin = np.dot((self.alpha * y), K) + self.b
            
            for i in range(n):
                condition = y[i] * margin[i] < 1
                
                # Subgradient method for the dual
                if condition:
                    self.alpha[i] += self.lr * (1 - self.C * self.alpha[i])
                else:
                    self.alpha[i] += self.lr * (-self.C * self.alpha[i])

                self.alpha[i] = np.clip(self.alpha[i], 0, self.C) # Box constraint

            # Update bias (simplified)
            self.b -= self.lr * np.mean(y * (margin < 1))

        self.X, self.y = X, y

    def project(self, X):
        K = polynomial_kernel(X, self.X, self.degree, self.coef0)
        return np.dot(self.alpha * self.y, K.T) + self.b

    def predict(self, X):
        return np.sign(self.project(X))

model = PolynomialSVM(C=1.0, lr=0.0001, n_iters=500, degree=2, coef0=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

tp = np.sum((y_test == 1) & (y_pred == 1))
tn = np.sum((y_test == -1) & (y_pred == -1))
fp = np.sum((y_test == -1) & (y_pred == 1))
fn = np.sum((y_test == 1) & (y_pred == -1))

precision = tp / (tp + fp + 1e-6)
recall = tp / (tp + fn + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)

print("=== Polynomial SVM (From Scratch) - Student Performance ===")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"Confusion Matrix:
[[TP={tp}, FP={fp}], [FN={fn}, TN={tn}]]")