import matplotlib.pyplot as plt
import numpy as np


N = 50
n = np.arange(N)
x = np.sin(2 * np.pi * n * 0.03)
x_noisy = x + 0.1 * np.random.randn(N)

polyFunction = np.polyfit(x_noisy, n, 3)
print(polyFunction)

plt.plot(n, x)
plt.plot(n, x_noisy)
plt.show()
