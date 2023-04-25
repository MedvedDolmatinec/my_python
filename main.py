"""Machine learning and data analysis"""
import numpy as np
import matplotlib
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def plot_regression_predictions(tree_reg, X, y):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis([-0.1, 0.5, -0.05, 0.25])
    plt.xlabel("$x_1$")
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")


if __name__ == "__main__":
    """Generate dataset"""

    np.random.seed(100)
    X_quad = np.random.rand(250, 1) - 0.5  # a single random input feature
    y_quad = X_quad ** 3 - 2 * X_quad ** 2 + 0.15 * np.random.randn(250, 1)

    """Use the default settings for the decision tree"""

    tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg.fit(X_quad, y_quad)
    pred = tree_reg.predict(X_quad)
    print("DT Train RMSE: %.2f"
          % np.sqrt(mean_squared_error(y_quad, pred)))
    print("DT Train R^2 Score: %.2f"
          % r2_score(y_quad, pred))

    """Show the received received scattergram"""

    fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
    plt.sca(axes[0])
    plot_regression_predictions(tree_reg, X_quad, y_quad)

    th0, th1a, th1b = tree_reg.tree_.threshold[[0, 1, 4]]
    for split, style in ((th0, "k-"), (th1a, "k--"), (th1b, "k--")):
        plt.plot([split, split], [-0.05, 0.25], style, linewidth=2)
    plt.text(th0, 0.16, "Depth=0", fontsize=15)
    plt.text(th1a + 0.01, -0.01, "Depth=1", horizontalalignment="center", fontsize=13)
    plt.text(th1b + 0.01, -0.01, "Depth=1", fontsize=13)
    plt.ylabel("$y$", rotation=0)
    plt.legend(loc="upper center", fontsize=16)
    plt.title("max_depth=2")

    tree_reg2 = DecisionTreeRegressor(max_depth=3, max_features=1, max_leaf_nodes=5, random_state=42)
    tree_reg2.fit(X_quad, y_quad)
    pred = tree_reg2.predict(X_quad)

    plt.sca(axes[1])
    th2s = tree_reg2.tree_.threshold[[0, 1, 4]]

    plot_regression_predictions(tree_reg2, X_quad, y_quad)
    for split, style in ((th0, "k-"), (th1a, "k--"), (th1b, "k--")):
        plt.plot([split, split], [-0.05, 0.25], style, linewidth=2)
    for split in th2s:
        plt.plot([split, split], [-0.05, 0.25], "k:", linewidth=1)
    plt.text(th2s[2] + 0.01, 0.15, "Depth=2", fontsize=13)
    plt.title("max_depth=3")

    plt.show()

    np.random.seed(200)
    X_quad_eval = np.random.rand(250, 1) - 0.5  # a single random input feature
    y_quad_eval = X_quad_eval ** 3 - 2 * X_quad_eval ** 2 + 0.15 * np.random.randn(250, 1)

    pred = tree_reg2.predict(X_quad)
    print("DT Train RMSE: %.2f"
          % np.sqrt(mean_squared_error(y_quad, pred)))
    print("DT Train R^2 Score: %.2f"
          % r2_score(y_quad, pred))
