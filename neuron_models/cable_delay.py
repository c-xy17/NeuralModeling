import numpy as np
import matplotlib.pyplot as plt
import brainpy as bp

plt.rcParams.update({"font.size": 15})
plt.rcParams['font.sans-serif'] = ['Times New Roman']


def exp(x, lamb=1., a=1.):
  return a * np.exp(- x / lamb)


def plot_lines(x):
  y = exp(x)
  plt.plot([x, x], [0, y], '--', color='grey')
  plt.plot([0, x], [y, y], '--', color='grey')



res = 300
x = np.linspace(0., 5, res)
V = exp(x)

fig, gs = bp.visualize.get_figure(1, 1, 4, 4 * 2.039)
ax = fig.add_subplot(gs[0, 0])
plt.plot(x, V)
plot_lines(1.)
plot_lines(2.)
plot_lines(3.)
plt.xlim(np.min(x), np.max(x))
plt.ylim(np.min(V), np.max(V))
plt.xticks([])
plt.yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.text(0.95, -0.05, r'$\lambda$')
plt.text(1.95, -0.05, r'$2\lambda$')
plt.text(2.95, -0.05, r'$3\lambda$')
plt.text(-0.25, exp(0.), r'$V_0$')
plt.text(-0.25, exp(1.), r'$V_1$')
plt.text(-0.25, exp(2.), r'$V_2$')
plt.text(-0.25, exp(3.), r'$V_3$')
# plt.savefig('cable_delay.png', transparent=True, dpi=500)

plt.show()
