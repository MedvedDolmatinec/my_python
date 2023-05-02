"""Machine learning and data analysis"""
import numpy as np
import matplotlib
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def action(X_quad, y_quad):
    """Use the default settings for the decision tree"""

    tree_reg = DecisionTreeRegressor(max_depth=4, random_state=42)
    tree_reg.fit(X_quad, y_quad)

    tree_reg2 = DecisionTreeRegressor(max_depth=5, random_state=42)
    tree_reg2.fit(X_quad, y_quad)

    """Show the received scattergram"""

    fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey='all')
    plt.sca(axes[0])
    plot_regression_predictions(tree_reg, X_quad, y_quad)
    plt.ylabel("$y$", rotation=0)
    plt.legend(loc="upper center", fontsize=16)
    plt.title("max_depth=4")

    plt.sca(axes[1])
    plot_regression_predictions(tree_reg2, X_quad, y_quad)
    plt.title("max_depth=5")

    plt.show()

    """calculation of R^2 and RMS metrics"""

    print("DT Train RMSE: %.2f(tree_reg)" % np.sqrt(mean_squared_error(y_quad, tree_reg.predict(X_quad))))
    print("DT Train R^2 Score: %.2f(tree_reg)" % r2_score(y_quad, tree_reg.predict(X_quad)))

    print("\nDT Train RMSE: %.2f(tree_reg2)" % np.sqrt(mean_squared_error(y_quad, tree_reg2.predict(X_quad))))
    print("DT Train R^2 Score: %.2f(tree_reg2)" % r2_score(y_quad, tree_reg2.predict(X_quad)))

    """Use different decision trees and show the resulting graphs"""

    tree_reg3 = DecisionTreeRegressor(max_depth=6, max_features=1, max_leaf_nodes=20, random_state=42)
    tree_reg3.fit(X_quad, y_quad)
    pred = tree_reg3.predict(X_quad)

    tree_reg4 = DecisionTreeRegressor(random_state=42)
    tree_reg4.fit(X_quad, y_quad)

    fig2, axes = plt.subplots(ncols=2, figsize=(14, 6), sharey='all')
    plt.sca(axes[0])
    plot_regression_predictions(tree_reg3, X_quad, y_quad)
    plt.ylabel("$y$", rotation=0)
    plt.legend(loc="upper center", fontsize=12)
    plt.title("max_depth=6, max_features=1, max_leaf_nodes=20")

    plt.sca(axes[1])
    plot_regression_predictions(tree_reg4, X_quad, y_quad)
    plt.title("no restricted")

    plt.show()

    print("\nDT Train RMSE: %.2f(tree_reg3)" % np.sqrt(mean_squared_error(y_quad, tree_reg3.predict(X_quad))))
    print("DT Train R^2 Score: %.2f(tree_reg3)" % r2_score(y_quad, tree_reg3.predict(X_quad)))

    print("\nDT Train RMSE: %.2f(tree_reg4)" % np.sqrt(mean_squared_error(y_quad, tree_reg4.predict(X_quad))))
    print("DT Train R^2 Score: %.2f(tree_reg4)" % r2_score(y_quad, tree_reg4.predict(X_quad)))


def dataset(n):
    np.random.seed(n)
    X = np.random.rand(250, 1) - 0.5  # a single random input feature
    y = X ** 3 - 2 * X ** 2 + 0.15 * np.random.randn(250, 1)
    return X, y


def plot_regression_predictions(tree_reg, X, y, axes=[-0.55, 0.55, -0.55, 0.45]):
    """graph boundaries, dataset conversion into graph points and their settings, adding labels"""

    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$x_1$")
    plt.plot(X, y, "g.")
    plt.plot(x1, y_pred, "b.-", linewidth=2, label=r"$\hat{y}$")


if __name__ == "__main__":
    """Regression analysis based on decision trees"""

    X_quad, y_quad = dataset(100)
    X_quad_eval, y_quad_eval = dataset(200)

    action(X_quad, y_quad)
    action(X_quad_eval, y_quad_eval)
