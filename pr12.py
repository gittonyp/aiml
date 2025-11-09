import pandas as pd
import numpy as np

try:
    df = pd.read_csv("student.csv")
except FileNotFoundError:
    print("Warning: 'student.csv' not found. Using dummy data.")
    data = {
        "student_id": range(1, 21),
        "hours_studied": np.random.rand(20) * 10 + 1,
        "exam_score": np.zeros(20) # will be filled
    }
    # Create a linear relationship with noise
    data["exam_score"] = data["hours_studied"] * 8.5 + 15 + np.random.randn(20) * 5
    df = pd.DataFrame(data)

if "hours_studied" not in df.columns or "exam_score" not in df.columns:
     raise ValueError("CSV must contain 'hours_studied' and 'exam_score' columns.")

X = df["hours_studied"].values
y = df["exam_score"].values

mean_x = np.mean(X)
mean_y = np.mean(y)

num = np.sum((X - mean_x) * (y - mean_y))
den = np.sum((X - mean_x) ** 2)

m = num / den
c = mean_y - m * mean_x

def predict(x):
    return m * x + c

y_pred = predict(X)

mse = np.mean((y - y_pred) ** 2)
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print("Linear Regression from Scratch")
print(f"Formula: score = {m:.4f} * hours + {c:.4f}")
print("-" * 30)
print("Slope (m):", round(m, 4))
print("Intercept (c):", round(c, 4))
print("Mean Squared Error (MSE):", round(mse, 4))
print("R² Score:", round(r2, 4))