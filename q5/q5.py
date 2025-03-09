import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# (a)
df = pd.read_excel("q5/Concrete_Data.xls")
print("First few rows of the dataset:")
print(df.head())
print("\nMissing values in each column:")
print(df.isnull().sum())

target_col = "Concrete compressive strength(MPa, megapascals) "
if target_col not in df.columns:
    raise ValueError("Target column not found. Check column names in the dataset.")

X = df.drop(target_col, axis=1)
y = df[target_col]

print("\nFeatures (Independent Variables):", list(X.columns))
print("Dependent Variable:", target_col)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# (b)
# Train the Linear Regression model using all features
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred_lin = lin_reg.predict(X_test)


def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# Calculate performance metrics for Linear Regression
mse_lin = compute_mse(y_test, y_pred_lin)
r2_lin = compute_r2(y_test, y_pred_lin)

# Save Linear Regression predicted vs. actual plot
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred_lin, alpha=0.7, color="blue")
plt.xlabel("Actual Compressive Strength")
plt.ylabel("Predicted Compressive Strength")
plt.title("Linear Regression: Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.savefig("q5/linear_regression_predicted_vs_actual.png")
plt.close()


# (c)
# Implementing Polynomial Regression (Degrees 2, 3, 4)
degrees = [2, 3, 4]
poly_results = {}

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    y_pred_poly = poly_model.predict(X_test_poly)

    mse_poly = compute_mse(y_test, y_pred_poly)
    r2_poly = compute_r2(y_test, y_pred_poly)

    poly_results[degree] = {"MSE": mse_poly, "R2": r2_poly, "predictions": y_pred_poly}

    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred_poly, alpha=0.7, color="purple")
    plt.xlabel("Actual Compressive Strength")
    plt.ylabel("Predicted Compressive Strength")
    plt.title(f"Polynomial Regression (Degree {degree}): Predicted vs Actual")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.savefig(f"q5/polynomial_regression_degree{degree}_predicted_vs_actual.png")
    plt.close()


# (d)
sorted_idx = np.argsort(y_test)
y_test_sorted = np.array(y_test)[sorted_idx]
lin_pred_sorted = y_pred_lin[sorted_idx]

plt.figure(figsize=(10, 6))
plt.plot(
    y_test_sorted,
    lin_pred_sorted,
    label=f"Linear Regression (MSE: {mse_lin:.2f}, R²: {r2_lin:.2f})",
    linewidth=2,
    color="blue",
)

colors = {2: "red", 3: "green", 4: "orange"}
for degree in degrees:
    poly_pred = poly_results[degree]["predictions"]
    poly_pred_sorted = poly_pred[sorted_idx]
    mse_poly = poly_results[degree]["MSE"]
    r2_poly = poly_results[degree]["R2"]
    plt.plot(
        y_test_sorted,
        poly_pred_sorted,
        label=f"Poly Degree {degree} (MSE: {mse_poly:.2f}, R²: {r2_poly:.2f})",
        linewidth=2,
        color=colors[degree],
    )

plt.xlabel("Actual Compressive Strength")
plt.ylabel("Predicted Compressive Strength")
plt.title("Overlay of Predictions: Linear vs. Polynomial Regression")
plt.legend()
plt.savefig("q5/overlay_predictions.png")
plt.close()


metrics_output = ""
metrics_output += "Linear Regression Performance:\n"
metrics_output += f"Mean Squared Error (MSE): {mse_lin:.4f}\n"
metrics_output += f"Coefficient of Determination (R²): {r2_lin:.4f}\n\n"

for degree in degrees:
    metrics_output += f"Polynomial Regression (Degree {degree}) Performance:\n"
    metrics_output += f"Mean Squared Error (MSE): {poly_results[degree]['MSE']:.4f}\n"
    metrics_output += (
        f"Coefficient of Determination (R²): {poly_results[degree]['R2']:.4f}\n\n"
    )


models = {"Linear Regression": {"MSE": mse_lin, "R2": r2_lin}}
for degree in degrees:
    models[f"Polynomial Regression (Degree {degree})"] = {
        "MSE": poly_results[degree]["MSE"],
        "R2": poly_results[degree]["R2"],
    }

best_mse_model = min(models, key=lambda k: models[k]["MSE"])
best_r2_model = max(models, key=lambda k: models[k]["R2"])

comparison_text = "Comparison of Models:\n"
comparison_text += f"Model with lowest MSE: {best_mse_model} (MSE: {models[best_mse_model]['MSE']:.4f})\n"
comparison_text += (
    f"Model with highest R²: {best_r2_model} (R²: {models[best_r2_model]['R2']:.4f})\n"
)

metrics_output += comparison_text
with open("q5/metrics.txt", "w") as f:
    f.write(metrics_output)


# (e)
"""

High Bias, Low Variance:
Linear Regression has an MSE of 95.9755 and R² of 0.6275. 
Its simplicity leads to systematic underfitting—high bias—because it cannot capture the underlying data patterns well, 
but it is stable (low variance) across different samples.

Low Bias, High Variance:
Polynomial Regression (Degree 4) shows an MSE of 677.4237 and a negative R² (-1.6289). 
This indicates that while the model is flexible enough to fit the training data very closely (low bias), 
it overfits the noise, resulting in poor generalization and highly variable predictions (high variance).

Balanced Bias and Variance:
Polynomial Regression (Degree 3) achieves the best balance with the lowest MSE of 40.2710 and the highest R² of 0.8437. 
It is complex enough to capture the true data patterns without overfitting the noise.



Why Higher-Degree Polynomials Overfit:

Excessive Flexibility: Higher-degree polynomials have many parameters and degrees of freedom. 
This allows them to fit the training data very closely—even the random noise.

Oscillations and Instability: The flexibility can lead to wild oscillations between data points, 
which are not reflective of the underlying trend, causing poor performance on new, unseen data.

Poor Generalization: Overfitting means the model learns the specific details of the training data, 
which often do not generalize well to other datasets, leading to high variance.


"""
