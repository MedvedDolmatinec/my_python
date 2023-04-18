"""Classification using SVM"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


"""
Convert the data set to points on the map and mark them with shapes.
"""


def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "ms")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
    plt.axis(axes)
    plt.grid(True)


"""
adjustment, predictions for each instance of the test, contour
"""


def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.1)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)


if __name__ == "__main__":
    """creating dataset"""

    X, y = make_moons(n_samples=100, noise=0.2, random_state=100)
    axes = [-1.75, 2.5, -1.25, 1.75]

    """Use a linear kernel for SVM and show the resulting graphs"""

    model = SVC(kernel="linear", C=1000)
    model.fit(X, y)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    DecisionBoundaryDisplay.from_estimator(model, X,
                                           plot_method="contour",
                                           colors="k",
                                           levels=[-1, 0, 1],
                                           alpha=0.5,
                                           linestyles=['--', '-', '--'],
                                           ax=ax)

    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, linewidths=1, facecolors='none',
               edgecolors='k')

    plt.title("Linear")
    plt.grid()

    plt.show()

    """Use the polynomial kernel for SVM and show the resulting graphs"""

    polynomial_svm_clf = make_pipeline(
        PolynomialFeatures(degree=3),
        SVC(kernel="poly", gamma="auto", C=50)
    )
    polynomial_svm_clf.fit(X, y)

    plot_predictions(polynomial_svm_clf, axes)
    plot_dataset(X, y, axes)

    plt.title("Poly")

    plt.show()

    """Use the RBF kernel for SVM and show the resulting graphs"""

    rbf_kernel_svm_clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", gamma=1, C=1)
    )
    rbf_kernel_svm_clf.fit(X, y)

    plot_predictions(rbf_kernel_svm_clf, axes)
    plot_dataset(X, y, axes)

    plt.title("rbf")

    plt.show()
