"""Classification using decision trees"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier


"""
    Convert the given data set to points 
    on the map and mark them with shapes.
    Also set contours and labels
"""


def plot_decision_boundary(clf, X, y, axes, cmap):
    x1, x2 = np.meshgrid(np.linspace(axes[0], axes[1], 100),
                         np.linspace(axes[2], axes[3], 100))
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)

    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=cmap)
    plt.contour(x1, x2, y_pred, cmap="Greys", alpha=0.5)
    colors = {"Wistia": ["#5c6978", "#872f2f"]}
    markers = ("o", "^")
    for idx in (0, 1):
        plt.plot(X[:, 0][y == idx], X[:, 1][y == idx],
                 color=colors[cmap][idx], marker=markers[idx], linestyle="none")
    plt.axis(axes)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$", rotation=0)


"""
Train the models. Display models on screen, one without restrictions, the other with
"""


def Model(X, y, model):
    model.fit(X, y)
    fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
    plt.sca(axes[0])
    plot_decision_boundary(model, X, y, axes=[-1.5, 2.4, -1, 1.5], cmap="Wistia")
    plt.title("No restrictions")
    plt.sca(axes[1])
    plot_decision_boundary(model, X, y, axes=[-1.5, 2.4, -1, 1.5], cmap="Wistia")
    plt.title(f"min_samples_leaf = {model.min_samples_leaf}")
    plt.ylabel("")
    plt.show()


"""
Creating models based on DecisionTreeClassifier using a dataset
"""


if __name__ == "__main__":
    X, y = make_moons(n_samples=200, noise=0.3, random_state=200)
    model = DecisionTreeClassifier(random_state=50)
    Model(X, y, model)

    model_restricted = DecisionTreeClassifier(min_samples_leaf=4, min_samples_split=4, max_depth=5, random_state=50)
    model_restricted.fit(X, y)
    Model(X, y, model_restricted)
