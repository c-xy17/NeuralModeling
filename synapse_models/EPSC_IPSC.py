# -*- coding: utf-8 -*-

import brainpy as bp
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 15})
plt.rcParams['font.sans-serif'] = ['Times New Roman']


def demonstrate_PSP():
  def show_PSP(E, ax):
    neu1 = bp.neurons.SpikeTimeGroup(1, times=[100], indices=[0])
    neu2 = bp.neurons.LIF(1, noise=0.01)
    syn1 = bp.synapses.Exponential(neu1, neu2, conn=bp.connect.All2All(),
                                   output=bp.synouts.COBA(E=E), g_max=1, tau=1.)
    net1 = bp.dyn.Network(pre=neu1, syn=syn1, post=neu2)

    # 运行模拟
    runner = bp.dyn.DSRunner(net1, monitors=['post.V'])
    runner.run(200)

    # 可视化
    i_start = int(80 / bp.math.get_dt())

    times = runner.mon.ts[i_start:] - 80
    potentials = runner.mon['post.V'][i_start:].flatten()
    plt.plot(times, potentials)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(r'$t$ (ms)')
    ax.set_ylabel('Potential (mV)')

    if E > 0:
      i = np.argmax(potentials)
      v1 = np.max(potentials)
      v0 = 0
    else:
      i = np.argmin(potentials)
      v1 = np.min(potentials)
      v0 = 0
    t1 = times[i] - 3
    t0 = times[i] + 3
    plt.plot([t0, t1], [0, 0], color='r')
    plt.plot([t0, t1], [v1, v1], color='r')
    plt.annotate('', xy=((t0 + t1) / 2, v0), xytext=((t0 + t1) / 2, v1),
                 arrowprops=dict(arrowstyle='<->', color='red'))
    plt.text(t0 - 25, (v0 + v1) / 2, 'EPSP' if E > 0 else 'IPSP')

  fig, gs = bp.visualize.get_figure(2, 1, 3, 4.5)
  ax = fig.add_subplot(gs[0, 0])
  show_PSP(20., ax=ax)
  ax.set_xlabel('')

  ax = fig.add_subplot(gs[1, 0])
  show_PSP(-20., ax=ax)

  plt.savefig('PSP.pdf', transparent=True, dpi=500)
  plt.show()


def demonstrate_PSC():
  def show_PSC(E, ax):
    neu1 = bp.neurons.SpikeTimeGroup(1, times=[100], indices=[0])
    neu2 = bp.neurons.LIF(1, noise=0.01)
    syn1 = bp.synapses.Exponential(neu1, neu2, conn=bp.connect.All2All(),
                                   output=bp.synouts.COBA(E=E), g_max=1, tau=1.)
    net1 = bp.dyn.Network(pre=neu1, syn=syn1, post=neu2)

    # 运行模拟
    runner = bp.dyn.DSRunner(net1, monitors=['post.V', 'post.input'])
    runner.run(200)

    # 可视化
    i_start = int(90 / bp.math.get_dt())
    i_end = int(120 / bp.math.get_dt())

    times = runner.mon.ts[i_start:i_end] - 90
    potentials = runner.mon['post.input'][i_start:i_end].flatten()
    plt.plot(times, potentials)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(r'$t$ (ms)')
    ax.set_ylabel('Current (pA)')

    if E > 0:
      i = np.argmax(potentials)
      v1 = np.max(potentials)
      v0 = 0
    else:
      i = np.argmin(potentials)
      v1 = np.min(potentials)
      v0 = 0
    t0 = times[i]
    t1 = times[i] - 3
    plt.plot([t0, t1], [0, 0], color='r')
    plt.plot([t0, t1], [v1, v1], color='r')
    plt.annotate('', xy=((t0 + t1) / 2, v0), xytext=((t0 + t1) / 2, v1),
                 arrowprops=dict(arrowstyle='<->', color='red'))
    plt.text(t0 - 8, (v0 + v1) / 2, 'EPSC' if E > 0 else 'IPSC')

  fig, gs = bp.visualize.get_figure(2, 1, 3, 4.5)
  ax = fig.add_subplot(gs[0, 0])
  show_PSC(20., ax=ax)
  ax.set_xlabel('')

  ax1 = fig.add_subplot(gs[1, 0])
  show_PSC(-20., ax=ax1)

  fig.align_ylabels([ax, ax1])
  plt.savefig('PSC.pdf', transparent=True, dpi=500)
  plt.show()


if __name__ == '__main__':
  demonstrate_PSP()
  demonstrate_PSC()
