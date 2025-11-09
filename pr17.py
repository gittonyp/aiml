import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

X = df.drop(columns=["Prediction"])
y = df["Prediction"].values
X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_train = X_train_df.values
X_test = X_test_df.values

def fit_naive_bayes(X, y):
    classes = np.unique(y)
    class_prob = {}
    feature_prob = {}
    n_features = X.shape[1]
    
    for c in classes:
        X_c = X[y == c]
        class_prob[c] = X_c.shape[0] / X.shape[0]
        # Laplace (add-1) smoothing
        feature_prob[c] = (X_c.sum(axis=0) + 1) / (X_c.sum() + n_features)
    return classes, class_prob, feature_prob

def predict_naive_bayes(X, classes, class_prob, feature_prob):
    preds = []
    for x in X:
        posteriors = []
        for c in classes:
            # Use log probabilities to avoid underflow
            log_prior = np.log(class_prob[c])
            # Handle cases where feature_prob[c] is 0 (though smoothing should prevent this)
            log_likelihood = np.sum(np.log(feature_prob[c] + 1e-9) * x)
            posteriors.append(log_prior + log_likelihood)
        preds.append(classes[np.argmax(posteriors)])
    return np.array(preds)


classes, class_prob, feature_prob = fit_naive_bayes(X_train, y_train)
y_pred = predict_naive_bayes(X_test, classes, class_prob, feature_prob)

acc = np.mean(y_pred == y_test)
print("=== Naïve Bayes (From Scratch) ===")
print("Email Spam Detection Accuracy:", round(acc, 4))