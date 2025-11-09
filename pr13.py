import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

try:
    df = pd.read_csv("student.csv")
except FileNotFoundError:
    print("Warning: 'student.csv' not found. Using dummy data.")
    data = {
        "hours_studied": np.random.rand(50) * 10 + 1,
        "sleep_hours": np.random.rand(50) * 3 + 5,
        "attendance_percent": np.random.rand(50) * 30 + 70,
        "Internal_marks": np.random.rand(50) * 20 + 10,
        "exam_score": np.zeros(50)
    }
    data["exam_score"] = (data["hours_studied"]*3 + 
                          data["attendance_percent"]*0.5 + 
                          data["Internal_marks"]*1.5 + 
                          np.random.randn(50)*3)
    df = pd.DataFrame(data)

features = ["hours_studied", "sleep_hours", "attendance_percent", "Internal_marks"]
target = "exam_score"

if not all(col in df.columns for col in features + [target]):
    raise ValueError("CSV missing required columns.")

X = df[features]
y = df[target]

model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=1)

r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

print("K-Fold Cross-Validation for Student Scores")
print("R² Scores for each fold:", np.round(r2_scores, 4))
print("Average R² Score:", round(np.mean(r2_scores), 4))