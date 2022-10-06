import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt

from neuron_models.FRNeuron import FR

def run_FR(syn_model, I_pre, dur, ax, label, **kwargs):
  # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
  num_pre = I_pre.shape[1]
  pre = FR(num_pre)
  post = FR(1)
  syn = syn_model(pre, post, conn=bp.conn.All2All(), **kwargs)
  net = bp.dyn.Network(pre=pre, post=post, syn=syn)

  # 运行模拟
  runner = bp.dyn.DSRunner(net, inputs=[('pre.input', I_pre, 'iter')],
                           monitors=['pre.r', 'post.r', 'syn.w'])
  runner(dur)

  # plt.sca(ax)
  # if label == 'Hebb learning':
  #   plt.plot(runner.mon.ts, np.sqrt(np.sum(np.square(runner.mon['syn.w']), axis=1)), label=label, linestyle='--')
  # else:
  #   plt.plot(runner.mon.ts, np.sqrt(np.sum(np.square(runner.mon['syn.w']), axis=1)), label=label)

  return runner


def visualize_cos(ax, x, w, step, label, linestyle='.-'):
  # 计算向量t_m和每个时间点w_m的夹角
  a2 = np.sum(x * x, axis=1)
  b2 = np.sum(w * w, axis=1)
  cos_m = np.sum(x * w, axis=1) / np.sqrt(a2 * b2)

  # plt.sca(ax)
  plt.plot(step, np.abs(cos_m), linestyle, label=label)
  plt.xlabel('time steps')


  # # 可视化
  # fig, gs = plt.subplots(3, 1, figsize=(6, 6))
  #
  # plt.sca(gs[0])
  # plt.plot(runner.mon.ts, runner.mon['pre.r'][:, 0], label='pre0 $r$', color=u'#ff7f0e')
  # plt.plot(runner.mon.ts, runner.mon['pre.r'][:, 1], label='pre1 $r$', color=u'#1f77b4')
  #
  # plt.sca(gs[1])
  # plt.plot(runner.mon.ts, runner.mon['post.r'], label='post $r$', color=u'#d62728')
  #
  # plt.sca(gs[2])
  # plt.plot(runner.mon.ts, runner.mon['syn.w'][:, 0], label='$w0$', color=u'#ff7f0e')
  # plt.plot(runner.mon.ts, runner.mon['syn.w'][:, 1], label='$w1$', color=u'#1f77b4')
  #
  # for i in range(2):
  #   gs[i].set_xticks([])
  # for i in range(3):
  #   gs[i].legend(loc='center right')
  #
  # plt.xlabel('t (ms)')
  # plt.tight_layout()
  # plt.subplots_adjust(hspace=0.)
  # plt.show()
