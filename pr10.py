import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    df = pd.read_csv("uber.csv")
except FileNotFoundError:
    print("Error: 'uber.csv' not found.")
    print("Please download the dataset and place it in the same directory.")
    exit()
    
print("Dataset Loaded Successfully!")
print("Shape:", df.shape)
print(df.head())

df = df.drop_duplicates().dropna()
numeric_df = df.select_dtypes(include=[np.number])

target_col = None
for col in ['fare_amount', 'price', 'amount', 'fare']:
    if col in numeric_df.columns:
        target_col = col
        break
if target_col is None:
    if numeric_df.shape[1] > 0:
        target_col = numeric_df.columns[0]
    else:
        raise ValueError("No numeric columns found in dataset.")

X = numeric_df.drop(columns=[target_col])
y = numeric_df[target_col]

print(f"Target Variable: {target_col}")

print("
Summary Statistics:
", df.describe())

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

plt.figure(figsize=(7, 4))
sns.histplot(y, bins=30, kde=True, color='skyblue')
plt.title(f"Distribution of {target_col}")
plt.xlabel(target_col)
plt.ylabel("Frequency")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_no_pca = LinearRegression()
model_no_pca.fit(X_train_scaled, y_train)
y_pred_no_pca = model_no_pca.predict(X_test_scaled)

r2_no_pca = r2_score(y_test, y_pred_no_pca)
rmse_no_pca = np.sqrt(mean_squared_error(y_test, y_pred_no_pca))
mae_no_pca = mean_absolute_error(y_test, y_pred_no_pca)

print("
=== Model Without PCA ===")
print(f"R² Score: {r2_no_pca:.4f}")
print(f"RMSE: {rmse_no_pca:.4f}")
print(f"MAE: {mae_no_pca:.4f}")

pca_full = PCA()
pca_full.fit(X_train_scaled)
explained_cumsum = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(7, 4))
plt.plot(range(1, len(explained_cumsum) + 1), explained_cumsum, marker='o')
plt.axhline(0.95, color='red', linestyle='--', label='95% Variance')
plt.title("Cumulative Explained Variance by PCA Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained")
plt.legend()
plt.grid(True)
plt.show()

n_components_95 = int(np.searchsorted(explained_cumsum, 0.95) + 1)
print(f"
Number of components to retain 95% variance: {n_components_95}")

pca = PCA(n_components=n_components_95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

model_pca = LinearRegression()
model_pca.fit(X_train_pca, y_train)
y_pred_pca = model_pca.predict(X_test_pca)

r2_pca = r2_score(y_test, y_pred_pca)
rmse_pca = np.sqrt(mean_squared_error(y_test, y_pred_pca))
mae_pca = mean_absolute_error(y_test, y_pred_pca)

print("
=== Model With PCA ===")
print(f"R² Score: {r2_pca:.4f}")
print(f"RMSE: {rmse_pca:.4f}")
print(f"MAE: {mae_pca:.4f}")

comparison = pd.DataFrame({
    "Model": ["Without PCA", "With PCA"],
    "R² Score": [r2_no_pca, r2_pca],
    "RMSE": [rmse_no_pca, rmse_pca],
    "MAE": [mae_no_pca, mae_pca]
})

print("
Model Performance Comparison:
", comparison)

plt.figure(figsize=(8, 5))
sns.barplot(data=comparison.melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Metric", y="Score", hue="Model", palette="viridis")
plt.title("Model Comparison (With vs Without PCA)")
plt.ylabel("Score Value")
plt.grid(True)
plt.show()