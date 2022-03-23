import numpy as np
import matplotlib.pyplot as plt


lamb, r_L, a, I0 = 5.4e-3, 30e6, 0.5e-3, 10

def V_ss(x):
  return (lamb * r_L) / (np.pi * a * a) * I0 * np.exp(- x / lamb)

res = 200
x = np.linspace(0., 0.1, res)
V = V_ss(x)

plt.plot(x, V)
plt.xlabel('$x$')
plt.ylabel('$V_{ss}$')

plt.show()
