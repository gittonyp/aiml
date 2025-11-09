import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

try:
    df = pd.read_csv("uber.csv")
except FileNotFoundError:
    print("Error: 'uber.csv' not found.")
    print("Please download the dataset and place it in the same directory.")
    exit()
    
df = df.drop_duplicates().dropna()
numeric_df = df.select_dtypes(include=[np.number])

target_col = None
for col in ['fare_amount', 'price', 'amount', 'fare']:
    if col in numeric_df.columns:
        target_col = col
        break
if target_col is None:
    # Fallback if no obvious target column
    if numeric_df.shape[1] > 0:
        target_col = numeric_df.columns[0]
    else:
        raise ValueError("No numeric columns found in dataset.")

X = numeric_df.drop(columns=[target_col])
y = numeric_df[target_col]

print("Dataset Shape:", df.shape)
print("Target Variable:", target_col)
print("Feature Columns:", list(X.columns))

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(7, 4))
sns.histplot(y, kde=True, bins=30, color='skyblue')
plt.title(f"Distribution of {target_col}")
plt.xlabel(target_col)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model1 = LinearRegression()
model1.fit(X_train_scaled, y_train)
y_pred1 = model1.predict(X_test_scaled)

r2_no_pca = r2_score(y_test, y_pred1)
rmse_no_pca = np.sqrt(mean_squared_error(y_test, y_pred1))

pca = PCA()
pca.fit(X_train_scaled)
explained = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(explained) + 1), explained, marker='o')
plt.axhline(0.95, color='r', linestyle='--')
plt.title("Cumulative Explained Variance by PCA")
plt.xlabel("No. of Components")
plt.ylabel("Cumulative Variance")
plt.grid(True)
plt.show()

n_components_95 = int(np.searchsorted(explained, 0.95) + 1)
print(f"Components needed for 95% variance: {n_components_95}")

pca = PCA(n_components=n_components_95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

model2 = LinearRegression()
model2.fit(X_train_pca, y_train)
y_pred2 = model2.predict(X_test_pca)

r2_pca = r2_score(y_test, y_pred2)
rmse_pca = np.sqrt(mean_squared_error(y_test, y_pred2))

comparison = pd.DataFrame({
    "Model": ["Without PCA", "With PCA"],
    "R² Score": [r2_no_pca, r2_pca],
    "RMSE": [rmse_no_pca, rmse_pca]
})
print("
Model Performance Comparison:
", comparison)

plt.figure(figsize=(6, 4))
sns.barplot(data=comparison.melt(id_vars="Model", var_name="Metric", value_name="Value"),
            x="Metric", y="Value", hue="Model", palette="viridis")
plt.title("Model Comparison (With vs Without PCA)")
plt.show()