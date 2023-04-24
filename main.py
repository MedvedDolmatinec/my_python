"""Regression analysis based on SVM"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR
import matplotlib
matplotlib.use('TkAgg')


def find_support_vectors(svm_reg, X, y):
    y_pred = svm_reg.predict(X)
    epsilon = svm_reg[-1].epsilon
    off_margin = np.abs(y - y_pred) >= epsilon
    return np.argwhere(off_margin)


def plot_svm_regression(svm_reg_fun, x_fun, y_fun, axes_fun):
    support_vectors = find_support_vectors(svm_reg_fun, x_fun, y_fun)
    x1s = np.linspace(axes_fun[0], axes_fun[1], 100).reshape(100, 1)
    y_pred = svm_reg_fun.predict(x1s)
    epsilon = svm_reg_fun[-1].epsilon
    plt.plot(x1s, y_pred, "k-", linewidth=2, label=r"$\hat{y}$", zorder=-2)
    plt.plot(x1s, y_pred + epsilon, "k--", zorder=-2)
    plt.plot(x1s, y_pred - epsilon, "k--", zorder=-2)
    plt.scatter(x_fun[support_vectors], y_fun[support_vectors], s=180, facecolors='red', zorder=-1)
    plt.plot(x_fun, y_fun, "bo")
    plt.plot(x1s, y_pred)
    plt.xlabel("$x_1$")
    plt.legend(loc="upper left")
    plt.axis(axes_fun)


if __name__ == "__main__":
    """Generate the dataset using the following code"""

    np.random.seed(100)
    X = 2 * np.random.rand(100, 1) - 1
    y = 0.2 + 0.1 * X[:, 0] + 0.5 * X[:, 0] ** 2 + np.random.randn(100) / 10

    """Use a linear kernel for SVM"""

    svm_linear_reg = make_pipeline(StandardScaler(),
                                   LinearSVR(C=3, epsilon=0.3, random_state=100))

    svm_linear_reg.fit(X, y)

    """Use polynomial kernel for SVM"""

    svm_poly_reg = make_pipeline(StandardScaler(),
                                 SVR(kernel="poly", degree=2,
                                     C=1, epsilon=0.2))

    svm_poly_reg.fit(X, y)

    """Show the resulting scattergram"""

    fig, axes = plt.subplots(ncols=2, figsize=(9, 4), sharey=True)

    plt.sca(axes[0])
    plot_svm_regression(svm_linear_reg, X, y, [-1.5, 1.5, -0.4, 1])
    plt.title(f"C={svm_linear_reg[-1].C}, "
              f"epsilon={svm_linear_reg[-1].epsilon}")
    plt.grid()

    plt.sca(axes[1])
    plot_svm_regression(svm_poly_reg, X, y, [-1.5, 1.5, -0.4, 1])
    plt.title(f"degree={svm_poly_reg[-1].degree}, "
              f"C={svm_poly_reg[-1].C}, "
              f"epsilon={svm_poly_reg[-1].epsilon}")
    plt.grid()

    plt.show()

    """Use rbf and sigmoid kernels for SVM, and show what you received"""

    svm_rbf_reg = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1, gamma=0.1, epsilon=0.2))
    svm_rbf_reg.fit(X, y)

    svm_sigmoid_reg = make_pipeline(StandardScaler(), SVR(kernel="sigmoid", coef0=0, C=3, gamma=0.2, epsilon=0.3))
    svm_sigmoid_reg.fit(X, y)

    fig2, axes = plt.subplots(ncols=2, figsize=(9, 4), sharey=True)

    plt.sca(axes[0])
    plot_svm_regression(svm_rbf_reg, X, y, [-1.5, 1.5, -0.4, 1])
    plt.title(f"C={svm_rbf_reg[-1].C}, "
              f"gamma={svm_rbf_reg[-1].gamma}, "
              f"epsilon={svm_rbf_reg[-1].epsilon}")
    plt.grid()

    plt.sca(axes[1])
    plot_svm_regression(svm_sigmoid_reg, X, y, [-1.5, 1.5, -0.4, 1])
    plt.title(f"coef0={svm_sigmoid_reg[-1].coef0}, "
              f"C={svm_sigmoid_reg[-1].C}, "
              f"gamma={svm_sigmoid_reg[-1].gamma}, "
              f"epsilon={svm_sigmoid_reg[-1].epsilon}")
    plt.grid()

    plt.show()
