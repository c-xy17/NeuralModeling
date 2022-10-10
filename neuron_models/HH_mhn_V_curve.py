import brainpy as bp
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15})
plt.rcParams['font.sans-serif'] = ['Times New Roman']


def tau_inf_alpha_beta(var, alpha_x, beta_x):
  V = np.arange(-110, 50, 0.1)
  alpha = alpha_x(V)
  beta = beta_x(V)
  tau = 1 / (alpha + beta)
  inf = alpha / (alpha + beta)

  fig, gs = bp.visualize.get_figure(1, 1, 3, 4.5)
  ax = fig.add_subplot(gs[0, 0])
  plt.plot(V, tau)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.set_xlabel(r'$V$ (mV)')
  ax.set_ylabel(r'$\tau_{}$'.format(var))
  plt.savefig(f'HH_{var}_tau.pdf', transparent=True, dpi=500)

  fig, gs = bp.visualize.get_figure(1, 1, 3, 4.5)
  ax = fig.add_subplot(gs[0, 0])
  plt.plot(V, inf)
  ax.set_xlabel(r'$V$ (mV)')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.set_ylabel(r'${}_\infty$'.format(var))
  plt.savefig(f'HH_{var}_infty.pdf', transparent=True, dpi=500)

  fig, gs = bp.visualize.get_figure(1, 1, 3, 4.5)
  ax = fig.add_subplot(gs[0, 0])
  plt.plot(V, alpha)
  ax.set_xlabel(r'$V$ (mV)')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.set_ylabel(r'$\alpha_{}$'.format(var))
  plt.savefig(f'HH_{var}_alpha.pdf', transparent=True, dpi=500)

  fig, gs = bp.visualize.get_figure(1, 1, 3, 4.5)
  ax = fig.add_subplot(gs[0, 0])
  plt.plot(V, beta)
  ax.set_xlabel(r'$V$ (mV)')
  ax.set_ylabel(r'$\beta_{}$'.format(var))
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.savefig(f'HH_{var}_beta.pdf', transparent=True, dpi=500)
  # plt.show()


alpha_n = lambda V: 0.01 * (V + 55.) / (1. - np.exp(-(V + 55.) / 10.))
beta_n = lambda V: 0.125 * np.exp(-(V + 65.) / 80.)

alpha_h = lambda V: 0.07 * np.exp(-(V + 65.) / 20.)
beta_h = lambda V: 1. / (np.exp(-(V + 35.) / 10) + 1.)

alpha_m = lambda V: 0.1 * (V + 40.) / (1 - np.exp(-(V + 40.) / 10.))
beta_m = lambda V: 4. * np.exp(-(V + 65.) / 18.)

tau_inf_alpha_beta('n', alpha_n, beta_n, )
tau_inf_alpha_beta('h', alpha_h, beta_h, )
tau_inf_alpha_beta('m', alpha_m, beta_m, )
