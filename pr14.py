import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

try:
    df = pd.read_csv("salary.csv")
except FileNotFoundError:
    print("Warning: 'salary.csv' not found. Using dummy data.")
    data = {
        'Age': np.random.randint(22, 60, 100),
        'Gender': np.random.choice(['Male', 'Female'], 100),
        'Education Level': np.random.choice(["Bachelor's", "Master's", "PhD"], 100),
        'Job Title': np.random.choice(['Developer', 'Manager', 'Analyst'], 100),
        'Years of Experience': np.random.randint(0, 30, 100),
        'Salary': np.zeros(100)
    }
    data['Salary'] = 50000 + data['Years of Experience']*2000 + np.random.randn(100)*5000
    df = pd.DataFrame(data)

df = df.dropna()

X = df.drop(columns=["Salary"])
y = df["Salary"]

# Identify categorical columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns
# Identify numerical columns
numerical_features = X.select_dtypes(include=np.number).columns

# Create a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

# Create the pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

print("K-Fold Cross-Validation for IT Salaries")
print("R² Scores for each fold:", np.round(r2_scores, 4))
print("Average R² Score:", round(np.mean(r2_scores), 4))