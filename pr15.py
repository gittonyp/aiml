import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

# Data provided in the prompt
data = {
    "ad_spend": [1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,
                 1200,1700,2200,2700,3200,3700,4200,4700,5200,5700],
    "discount": [5,10,0,15,20,5,10,25,30,15,
                 8,12,3,18,22,7,14,28,33,17],
    "customer_footfall": [200,250,300,350,400,450,500,550,600,650,
                          210,260,310,360,410,460,510,560,610,660],
    "sales": [10000,15000,13000,20000,25000,22000,27000,30000,35000,40000,
              11000,16000,14000,21000,26000,23000,28000,31000,36000,41000]
}
# Increased N to 20 for KFold=5 to work
df = pd.DataFrame(data)

X = df[["ad_spend", "discount", "customer_footfall"]]
y = df["sales"]

model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=1)
scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

print("K-Fold Cross-Validation for Sales Forecast")
print("R² Scores for each fold:", np.round(scores, 4))
print("Average R²:", np.round(np.mean(scores), 4))