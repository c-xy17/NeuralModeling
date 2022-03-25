import brainpy as bp
import numpy as np
import matplotlib.pyplot as plt


def tau_inf_alpha_beta(var, alpha_x, beta_x):
  V = np.arange(-110, 50, 0.1)
  alpha = alpha_x(V)
  beta = beta_x(V)
  tau = 1 / (alpha + beta)
  inf = alpha / (alpha + beta)
  fig, gs = bp.visualize.get_figure(2, 2, 3, 4.5)
  fig.add_subplot(gs[0, 0])
  plt.plot(V, tau)
  plt.title(r'$\tau_{}$'.format(var))
  fig.add_subplot(gs[0, 1])
  plt.plot(V, inf)
  plt.title(r'${}_\infty$'.format(var))
  fig.add_subplot(gs[1, 0])
  plt.plot(V, alpha)
  plt.title(r'$\alpha_{}$'.format(var))
  plt.xlabel('V (mV)')
  fig.add_subplot(gs[1, 1])
  plt.plot(V, beta)
  plt.title(r'$\beta_{}$'.format(var))
  plt.xlabel('V (mV)')
  plt.show()


alpha_n = lambda V: 0.01 * (V + 55.) / (1. - np.exp(-(V + 55.) / 10.))
beta_n = lambda V: 0.125 * np.exp(-(V + 65.) / 80.)

alpha_h = lambda V: 0.07 * np.exp(-(V + 65.) / 20.)
beta_h = lambda V: 1. / (np.exp(-(V + 35.) / 10) + 1.)

alpha_m = lambda V: 0.1 * (V + 40.) / (1 - np.exp(-(V + 40.) / 10.))
beta_m = lambda V: 4. * np.exp(-(V + 65.) / 18.)

tau_inf_alpha_beta('n', alpha_n, beta_n)
tau_inf_alpha_beta('h', alpha_h, beta_h)
tau_inf_alpha_beta('m', alpha_m, beta_m)

