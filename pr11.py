import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

try:
    df = pd.read_csv("house.csv")
except FileNotFoundError:
    print("Error: 'house.csv' not found.")
    print("Please download a suitable house price dataset and ensure columns match.")
    # Making up data if file is missing/wrong
    data = {
        "Id": range(100),
        "Price": np.random.randint(100000, 500000, 100),
        "Location": np.random.choice(["Urban", "Suburban", "Rural"], 100),
        "Condition": np.random.choice(["Good", "Fair", "Poor"], 100),
        "Garage": np.random.choice(["Yes", "No"], 100),
        "Area": np.random.randint(1000, 3000, 100),
        "Bedrooms": np.random.randint(2, 6, 100)
    }
    df = pd.DataFrame(data)
    print("Using dummy data as 'house.csv' was not found or valid.")


X = df.drop(columns=["Id", "Price"])
y = df["Price"].values

# Identify categorical columns
categorical_features = ["Location", "Condition", "Garage"]
# Identify numerical columns
numerical_features = [col for col in X.columns if col not in categorical_features]

# Create a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

# Create the pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

kf = KFold(n_splits=5, shuffle=True, random_state=1)

mses = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
rmses = np.sqrt(mses)
r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

print("5-Fold MSEs:", np.round(mses, 2))
print("Mean MSE:", np.round(mses.mean(), 2))
print("5-Fold RMSEs:", np.round(rmses, 2))
print("Mean RMSE:", np.round(rmses.mean(), 2))
print("5-Fold R² Scores:", np.round(r2_scores, 4))
print("Mean R² Score:", np.round(r2_scores.mean(), 4))