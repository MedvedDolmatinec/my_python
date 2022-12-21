import matplotlib.pyplot as plt
import numpy as np


def action(n, x, x_noisy, M):
    poLyFunc = np.poly1d(np.polyfit(n, x_noisy, M))

    E = 0
    for i in n:
        E += (poLyFunc(n)[i] ** 2 - 2 * poLyFunc(n)[i] * x[i] + x[i] ** 2) / 2

    plt.title(f"M = {M}, E = {E}")
    plt.plot(n, poLyFunc(n), 'b')
    plt.plot(n, x_noisy, 'r')
    plt.show()


if __name__ == "__main__":
    N = 20
    n = np.arange(N)

    x = np.sin(2 * np.pi * n * 0.03)
    x_noisy = x + 0.1 * np.random.randn(N)

    plt.title("Навчальний набір даних")
    plt.plot(n, x, 'b')
    plt.plot(n, x_noisy, 'r')
    plt.show()

    action(n, x, x_noisy, 0)
    action(n, x, x_noisy, 1)
    action(n, x, x_noisy, 3)
    action(n, x, x_noisy, 9)

    Ntest = 20
    n = np.arange(Ntest)
    std_deviation = 0.3
    xtest = np.random.random_sample(Ntest)
    ytest = np.sin(2*np.pi*xtest) + np.random.randn(Ntest) * std_deviation

    plt.title("Тестовий набір даних")
    plt.plot(n, xtest, 'b')
    plt.plot(n, ytest, 'r')
    plt.show()

    action(n, xtest, ytest, 0)
    action(n, xtest, ytest, 1)
    action(n, xtest, ytest, 3)
    action(n, xtest, ytest, 9)

