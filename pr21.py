import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    df = pd.read_csv("cancer.csv")
except FileNotFoundError:
    print("Warning: 'cancer.csv' not found. Using dummy data.")
    N = 500
    data = {f"feature_{i}": np.random.rand(N) for i in range(30)}
    data['id'] = range(N)
    data['diagnosis'] = np.random.choice(['M', 'B'], N)
    df = pd.DataFrame(data)

df = df.drop(columns=["id"], errors='ignore').dropna()

if "diagnosis" not in df.columns:
    raise ValueError("CSV must contain a 'diagnosis' column (M/B).")

y = np.where(df["diagnosis"] == "M", 1, -1)
X = df.drop(columns=["diagnosis"]).values.astype(float)

X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

split_idx = int(0.7 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

def polynomial_kernel(X1, X2, degree=3, coef0=1):
    return (np.dot(X1, X2.T) + coef0) ** degree

class PolynomialSVM:
    def __init__(self, C=1.0, lr=0.001, n_iters=300, degree=3, coef0=1):
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

                self.alpha[i] = np.clip(self.alpha[i], 0, self.C)

            self.b -= self.lr * np.mean(y * (margin < 1))

        self.X, self.y = X, y

    def project(self, X):
        K = polynomial_kernel(X, self.X, self.degree, self.coef0)
        return np.dot(self.alpha * self.y, K.T) + self.b

    def predict(self, X):
        return np.sign(self.project(X))

model = PolynomialSVM(C=1.0, lr=0.0001, n_iters=500, degree=3, coef0=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_scores = model.project(X_test)  # For ROC curve

tp = np.sum((y_test == 1) & (y_pred == 1))
tn = np.sum((y_test == -1) & (y_pred == -1))
fp = np.sum((y_test == -1) & (y_pred == 1))
fn = np.sum((y_test == 1) & (y_pred == -1))

accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
precision = tp / (tp + fp + 1e-6)
recall = tp / (tp + fn + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)

print("=== Polynomial SVM (From Scratch) – Breast Cancer Classification ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"
Confusion Matrix:
[[TP={tp}, FP={fp}], [FN={fn}, TN={tn}]]")

def compute_roc(y_true, y_score):
    # Sort scores and corresponding true labels
    indices = np.argsort(y_score)
    y_true_sorted = y_true[indices]
    y_score_sorted = y_score[indices]
    
    tpr_list, fpr_list = [0.0], [0.0]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == -1)
    
    tp, fp = 0, 0
    
    # Iterate through sorted scores from high to low
    for i in range(len(y_score_sorted) - 1, -1, -1):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        
        tpr_list.append(tp / (n_pos + 1e-6))
        fpr_list.append(fp / (n_neg + 1e-6))
        
    tpr_list.append(1.0)
    fpr_list.append(1.0)
    return np.array(fpr_list), np.array(tpr_list)

# Compute ROC points
fpr, tpr = compute_roc(y_test, y_scores)

# Plot ROC Curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', label='Polynomial SVM (Scratch)')
plt.plot([0, 1], [0, 1], 'r--', label='Random Chance')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve – SVM with Polynomial Kernel")
plt.legend()
plt.grid(True)
plt.show()