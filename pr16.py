import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

try:
    df = pd.read_csv("emails.csv")
except FileNotFoundError:
    print("Warning: 'emails.csv' not found. Using dummy data.")
    # Create dummy word frequency data
    N = 1000
    data = {f"word_{i}": np.random.randint(0, 5, N) for i in range(50)}
    data["Prediction"] = np.random.choice([0, 1], N, p=[0.7, 0.3])
    df = pd.DataFrame(data)

# Drop any non-numeric columns (like "Email No." or text columns)
df = df.select_dtypes(include=["number"])
df = df.dropna()

if "Prediction" not in df.columns:
    raise ValueError("The dataset must have a 'Prediction' column as the target variable.")

X = df.drop(columns=["Prediction"])
y = df["Prediction"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=== Naïve Bayes Email Spam Detection ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("Confusion Matrix:
", cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Spam', 'Spam'], 
            yticklabels=['Not Spam', 'Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()