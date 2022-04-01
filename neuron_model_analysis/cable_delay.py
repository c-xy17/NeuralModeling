import numpy as np
import matplotlib.pyplot as plt


def exp(x, lamb=1., a=1.):
  return a * np.exp(- x / lamb)


def plot_lines(x):
  y = exp(x)
  plt.plot([x, x], [0, y], '--', color='grey')
  plt.plot([0, x], [y, y], '--', color='grey')


res = 300
x = np.linspace(0., 5, res)
V = exp(x)

plt.plot(x, V)

plot_lines(1.)
plot_lines(2.)
plot_lines(3.)

plt.xlabel('$x$')
plt.ylabel('$V_{ss}$')
plt.xlim(np.min(x), np.max(x))
plt.ylim(np.min(V), np.max(V))
plt.xticks([])
plt.yticks([])

plt.show()
