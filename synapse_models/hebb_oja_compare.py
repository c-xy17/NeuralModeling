import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

from neuron_models.FRNeuron import FR
from synapse_models.Hebb import Hebb
from synapse_models.Oja import Oja


def _visualize_cos(ax, x, w, step, label, linestyle='.-'):
  # 计算向量t_m和每个时间点w_m的夹角
  a2 = np.sum(x * x, axis=1)
  b2 = np.sum(w * w, axis=1)
  cos_m = np.sum(x * w, axis=1) / np.sqrt(a2 * b2)
  ax.plot(step, np.abs(cos_m), linestyle, label=label)


def compare_w_normal():
  dur = 100.  # 模拟总时长
  n_pre = 32  # 32个突触前神经元
  num_sample = 20  # 挑选20个时间点可视化
  n_steps = int(dur / bm.get_dt())  # 模拟总步长

  rng = bm.random.RandomState(299)
  I_pre = rng.normal(scale=0.1, size=(n_steps, n_pre)) + rng.uniform(size=n_pre)
  step_m = np.linspace(0, n_steps - 1, num_sample).astype(int)
  x = np.asarray(I_pre)[step_m]

  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.)
  ax = fig.add_subplot(gs[0, 0])
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)

  # Hebb learning rule
  pre = FR(I_pre.shape[1])
  post = FR(1)
  syn = Hebb(pre, post, conn=bp.conn.All2All(), eta=0.003)
  net = bp.dyn.Network(pre=pre, post=post, syn=syn)
  runner = bp.dyn.DSRunner(net, inputs=[(pre.input, I_pre, 'iter')],
                           monitors=['pre.r', 'post.r', 'syn.w'])
  runner(dur)
  plt.plot(runner.mon.ts,
           np.sqrt(np.sum(np.square(runner.mon['syn.w']), axis=1)),
           label='Hebb learning', linestyle='--')

  # Oja's rule
  pre = FR(I_pre.shape[1])
  post = FR(1)
  syn = Oja(pre, post, conn=bp.conn.All2All(), eta=0.003)
  net = bp.dyn.Network(pre=pre, post=post, syn=syn)
  runner = bp.dyn.DSRunner(net, inputs=[(pre.input, I_pre, 'iter')],
                           monitors=['pre.r', 'post.r', 'syn.w'])
  runner(dur)
  plt.plot(runner.mon.ts, np.sqrt(np.sum(np.square(runner.mon['syn.w']), axis=1)), label='Oja\'rule')

  plt.ylabel('$||w||$')
  plt.legend()
  plt.xlabel('$t$ (ms)')
  plt.xlim(-1, dur + 1)
  plt.savefig('hebb_oja_compare1.pdf', transparent=True, dpi=500)
  plt.show()


def compare_cos_x_w():
  dur = 100.  # 模拟总时长
  n_pre = 32  # 32个突触前神经元
  num_sample = 20  # 挑选20个时间点可视化
  n_steps = int(dur / bm.get_dt())  # 模拟总步长

  rng = bm.random.RandomState(299)
  I_pre = rng.normal(scale=0.1, size=(n_steps, n_pre)) + rng.uniform(size=n_pre)
  step_m = np.linspace(0, n_steps - 1, num_sample).astype(int)
  x = np.asarray(I_pre)[step_m]

  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.)
  ax = fig.add_subplot(gs[0, 0])
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)

  # Hebb learning rule
  pre = FR(I_pre.shape[1])
  post = FR(1)
  syn = Hebb(pre, post, conn=bp.conn.All2All(), eta=0.003)
  net = bp.dyn.Network(pre=pre, post=post, syn=syn)
  runner = bp.dyn.DSRunner(net, inputs=[(pre.input, I_pre, 'iter')],
                           monitors=['pre.r', 'post.r', 'syn.w'])
  runner(dur)
  _visualize_cos(ax, x,
                 runner.mon['syn.w'][step_m],
                 runner.mon['ts'][step_m],
                 'Hebb learning')

  # Oja's rule
  pre = FR(I_pre.shape[1])
  post = FR(1)
  syn = Oja(pre, post, conn=bp.conn.All2All(), eta=0.003)
  net = bp.dyn.Network(pre=pre, post=post, syn=syn)
  runner = bp.dyn.DSRunner(net, inputs=[(pre.input, I_pre, 'iter')],
                           monitors=['pre.r', 'post.r', 'syn.w'])
  runner(dur)
  _visualize_cos(ax, x,
                 runner.mon['syn.w'][step_m],
                 runner.mon['ts'][step_m],
                 'Oja\'rule', linestyle='-')

  # eigenvectors
  C = np.dot(x.T, x)
  eigvals, eigvecs = np.linalg.eig(C)
  eigvals, eigvecs = eigvals.real, eigvecs.T.real
  largest = eigvecs[np.argsort(eigvals)[-1]]
  _visualize_cos(ax, x,
                 np.ones((num_sample, n_pre)) * largest,
                 runner.mon['ts'][step_m],
                 'cos($x, v_1$)', linestyle='--')

  plt.ylabel('$\cos(x, w)$')
  plt.xlabel('$t$ (ms)')
  plt.legend()
  plt.xlim(-1, dur + 1)
  plt.savefig('hebb_oja_compare2.pdf', transparent=True, dpi=500)
  plt.show()


if __name__ == '__main__':
  compare_w_normal()
  compare_cos_x_w()
