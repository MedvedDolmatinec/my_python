import matplotlib.pyplot as plt
import numpy as np


N = 50
n = np.arange(N)
x = np.sin(2 * np.pi * n * 0.03)
x_noisy = x + 0.1 * np.random.randn(N)

plt.subplot(2, 1, 1)
plt.plot(n, x)
plt.plot(n, x_noisy)

poLyFunc = np.poly1d(np.polyfit(n, x_noisy, 3))

plt.subplot(2, 1, 2)
plt.plot(n, poLyFunc(n), 'b')
plt.plot(n, x_noisy, 'r')
plt.show()
