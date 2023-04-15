"""Machine learning and data analysis"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use('TkAgg')


def find_support_vectors(svm_reg, X, y):
    y_pred = svm_reg.predict(X)
    epsilon = svm_reg[-1].epsilon
    off_margin = np.abs(y - y_pred) >= epsilon
    return np.argwhere(off_margin)


def plot_svm_regression(svm_reg, X, y, axes):
    x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
    y_pred = svm_reg.predict(x1s)
    epsilon = svm_reg[-1].epsilon
    plt.plot(x1s, y_pred, "k-", linewidth=2, label=r"$\hat{y}$", zorder=-2)
    plt.plot(x1s, y_pred + epsilon, "k--", zorder=-2)
    plt.plot(x1s, y_pred - epsilon, "k--", zorder=-2)
    plt.scatter(X[svm_reg._support], y[svm_reg._support], s=180, facecolors='#AAA', zorder=-1)
    plt.plot(X, y, "bo")
    plt.xlabel("$x_1$")
    plt.legend(loc="upper left")
    plt.axis(axes)


if __name__ == "__main__":
    np.random.seed(100)
    X = 2 * np.random.rand(100, 1) - 1
    y = 0.2 + 0.1 * X[:, 0] + 0.5 * X[:, 0] ** 2 + np.random.randn(100) / 10

    svm_reg = make_pipeline(StandardScaler(),
                            LinearSVR(epsilon=10,
                                      random_state=100))
    svm_reg.fit(X, y)
    pred = svm_reg.predict(X)
    print("SVR Train RMSE: %.2f"
          % np.sqrt(mean_squared_error(y, pred)))
    print("SVR Train R^2 Score: %.2f"
          % r2_score(y, pred))

    fig, axes = plt.subplots(ncols=2, figsize=(9, 4), sharey=True)
    plt.sca(axes[0])
    plot_svm_regression(svm_reg, X, y, [-1, 1, 0, 1])
    plt.title(f"degree={svm_reg[-1].degree}, "
              f"C={svm_reg[-1].C}, "
              f"epsilon={svm_reg[-1].epsilon}")
    plt.ylabel("$y$", rotation=0)
    plt.grid()

    # plt.sca(axes[1])
    # plot_svm_regression(svm_reg2, X, y, [-1, 1, 0, 1])
    # plt.title(f"degree={svm_reg2[-1].degree}, "
    #           f"C={svm_reg2[-1].C}, "
    #           f"epsilon={svm_reg2[-1].epsilon}")
    # plt.grid()
    # plt.show()
