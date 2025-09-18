import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


df = pd.read_csv("Automobile.csv")


df_clean = df.dropna(subset=['horsepower'])


X = df_clean[['displacement', 'weight', 'acceleration', 'cylinders']]
y = df_clean['horsepower']   # cieľová premenná


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


r2 = r2_score(y_test, y_pred)
print("R² score:", r2)


plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, edgecolors="k")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Skutočný výkon (horsepower)")
plt.ylabel("Predikovaný výkon")
plt.title(f"Porovnanie skutočných a predikovaných hodnôt\nR² = {r2:.3f}")
plt.grid(True)
plt.show()
