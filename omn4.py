import matplotlib.pyplot as plt
import numpy as np


def main(listX, listY, N):
    plt.plot(listX, listY)
    plt.show()

    a1 = (1/N) * sum(list(x * y for x, y in zip(listX, listY)))
    a2 = (1/N) * sum(list(map(lambda i: i ** 2, listX)))
    mx = (1/N) * sum(listX)
    my = (1/N) * sum(listY)
    # print(f"a1: {a1} a2: {a2} mx: {mx} my: {my}")

    k = (a1 - mx * my) / (a2 - mx ** 2)
    b = my - k * mx
    # print(f"k: {k} b: {b}")

    fx = list(map(lambda i: k * i + b, listX))
    E = sum(list(y**2 - 2*y*f + f**2 for y, f in zip(listY, fx)))
    print(f"F(x): {fx} \nE: {E}")


if __name__ == "__main__":
    N = 20
    listX = list(range(1, N * 2 + 1, 2))       # [1, 3, 5, 7..]
    listY = list(range(1, N + 1))              # [1, 2, 3, .. N]

    # Лінійна функція з похибками

    xListE = list(x + y for x, y in zip(listX, list(np.random.normal(0, 0.5, N))))
    yListE = list(x + y for x, y in zip(listY, list(np.random.normal(1, 0.5, N))))

    # main(listX, listY, N)     # без похибок
    main(xListE, yListE, N)     # за похибками



