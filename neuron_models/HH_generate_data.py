# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 15})
plt.rcParams['font.sans-serif'] = ['Times New Roman']


class SeparateNaK(bp.DynamicalSystem):
  def __init__(self, num=1, ENa=50., gNa=120., EK=-77., gK=36., EL=-54.387,
               gL=0.03, V_th=20., C=1.0):
    # 初始化
    super(SeparateNaK, self).__init__()

    # 定义神经元参数
    self.ENa = ENa
    self.EK = EK
    self.EL = EL
    self.g_Na_max = gNa
    self.g_K_max = gK
    self.gL = gL
    self.C = C
    self.V_th = V_th

    # 定义神经元变量
    self.V = bm.Variable(bm.asarray([-70.67647] * num))  # 膜电位
    self.m = bm.Variable(bm.asarray([0.02657777] * num))  # 离子通道m
    self.h = bm.Variable(bm.asarray([0.77206403] * num))  # 离子通道h
    self.n = bm.Variable(bm.asarray([0.23536022] * num))  # 离子通道n
    self.gNa = bm.Variable(bm.zeros(num))
    self.gK = bm.Variable(bm.zeros(num))
    self.INa = bm.Variable(bm.zeros(num))
    self.IK = bm.Variable(bm.zeros(num))
    self.IL = bm.Variable(bm.zeros(num))
    self.Ifb = bm.Variable(bm.zeros(num))

    # 定义积分函数
    self.integral = bp.odeint(f=bp.JointEq(self.dm, self.dh, self.dn),
                              method='exponential_euler')

  def dm(self, m, t):
    alpha = 0.1 * (self.V + 40) / (1 - bm.exp(-(self.V + 40) / 10))
    beta = 4.0 * bm.exp(-(self.V + 65) / 18)
    dmdt = alpha * (1 - m) - beta * m
    return dmdt

  def dh(self, h, t):
    alpha = 0.07 * bm.exp(-(self.V + 65) / 20.)
    beta = 1 / (1 + bm.exp(-(self.V + 35) / 10))
    dhdt = alpha * (1 - h) - beta * h
    return dhdt

  def dn(self, n, t):
    alpha = 0.01 * (self.V + 55) / (1 - bm.exp(-(self.V + 55) / 10))
    beta = 0.125 * bm.exp(-(self.V + 65) / 80)
    dndt = alpha * (1 - n) - beta * n
    return dndt

  def update(self, tdi):
    t, dt = tdi.t, tdi.dt
    m, h, n = self.integral(self.m, self.h, self.n, t, dt)
    self.m.value = m
    self.h.value = h
    self.n.value = n
    self.gNa.value = self.g_Na_max * self.m ** 3 * self.h
    self.gK.value = self.g_K_max * self.n ** 4
    self.INa.value = self.gNa * (self.V - self.ENa)
    self.IK.value = self.gK * (self.V - self.EK)
    self.IL.value = self.gL * (self.V - self.EL)
    self.Ifb.value = self.INa + self.IK + self.IL


def try_steady_state():
  runner = bp.DSRunner(SeparateNaK(), monitors=['m', 'h', 'n', 'INa', 'IK', 'IL', 'Ifb'])
  runner.run(100.)

  bp.visualize.line_plot(runner.mon.ts, runner.mon.m)
  bp.visualize.line_plot(runner.mon.ts, runner.mon.h)
  bp.visualize.line_plot(runner.mon.ts, runner.mon.n, show=True)


def separation_of_Na_and_K_currents():
  dt = 0.1
  vs, duration = bp.inputs.section_input([-70.67647, -70.67647 + 56],
                                         durations=[1, 9],
                                         return_length=True,
                                         dt=dt)

  runner = bp.DSRunner(SeparateNaK(),
                       monitors=['m', 'h', 'n', 'INa', 'IK', 'IL', 'Ifb', 'gNa', 'gK'],
                       inputs=['V', vs, 'iter', '='],
                       dt=dt)
  runner.run(duration)

  # fig, gs = bp.visualize.get_figure(1, 1, 4, 8)
  # bp.visualize.line_plot(runner.mon.ts, runner.mon.m)
  # bp.visualize.line_plot(runner.mon.ts, runner.mon.h)
  # bp.visualize.line_plot(runner.mon.ts, runner.mon.n, show=True)

  fig, gs = bp.visualize.get_figure(1, 1, 4, 8)
  ax = fig.add_subplot(gs[0, 0])
  plt.plot(runner.mon.ts, runner.mon.INa, lw=3)
  plt.text(1.9, 203, 'INa')
  plt.plot(runner.mon.ts, runner.mon.IK, lw=3)
  plt.text(5.03, 470, 'Ifb')
  plt.plot(runner.mon.ts, runner.mon.Ifb, lw=3)
  plt.text(5.03, -320, 'IK')
  plt.xlabel(r'$t$ (ms)')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.savefig('separation_of_Na_and_K_currents.pdf', transparent=True, dpi=500)
  plt.show()


def try_step_voltage2():
  steps = bm.asarray([41, 55, 70, 84, 99, 113, 127])
  steps = bm.asarray([6, 10, 20, 35, 50, 75, 100])
  vs, duration = bp.inputs.section_input([-70.67647, -70.67647 + steps],
                                         durations=[1, 20],
                                         return_length=True)
  runner = bp.DSRunner(SeparateNaK(steps.size),
                       monitors=['INa', 'IK', 'IL', 'Ifb', 'gNa', 'gK'],
                       inputs=['V', vs, 'iter', '='])
  runner.run(duration)

  names1 = 'abcdefg'
  names2 = 'hijklmn'

  Na_lims = [(0., 0.015), (0., 0.06), (0., 1.), (-0.5, 10), (-1, 23.), (-1, 42), (-1, 55)]
  Na_ticks = [(0., 0.015), (0., 0.06), (0., 1.), (0., 10), (0., 23.), (0, 42), (0, 55)]
  K_lims = [(0.1, 0.4), (0., 0.8), (0., 3.1), (0., 10.), (-1., 18), (-1, 28), (-1, 32)]
  K_ticks = [(0.1, 0.4), (0., 0.8), (0., 3.1), (0., 10.), (0., 18), (0, 28), (0, 32)]

  plt.rcParams.update({"font.size": 15})
  plt.rcParams['font.sans-serif'] = ['Times New Roman']
  fig, gs = bp.visualize.get_figure(steps.size, 2, 1, 4)
  axes = []
  for i in range(steps.size):
    ax = fig.add_subplot(gs[i, 0])
    data = runner.mon.gNa[:, i]
    plt.plot(runner.mon.ts, data, )
    # plt.ylabel('mS/cm$^2$', fontsize=12)
    if i == 0:
      plt.title('gNa')
    if i == steps.size - 1:
      plt.xlabel(r'$t$ [ms]')
    else:
      plt.xticks([])
    axes.append(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.text(16, (data.max() - data.min()) * 0.9 + data.min(),
             f'+{steps[i]} mV'
             # f'({names1[i]})'
             )
    plt.ylim(*Na_lims[i])
    plt.yticks(Na_ticks[i])
    if i < 2:
      plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax = fig.add_subplot(gs[i, 1])
    data = runner.mon.gK[:, i]
    plt.plot(runner.mon.ts, data, 'r')
    # plt.ylabel('mS/cm$^2$')
    if i == 0:
      plt.title('gK')
    if i == steps.size - 1:
      plt.xlabel(r'$t$ [ms]')
    else:
      plt.xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.text(16, (data.max() - data.min()) * 0.1 + data.min(),
             f'+{steps[i]} mV'
             # f'({names2[i]})'
             )
    plt.ylim(*K_lims[i])
    plt.yticks(K_ticks[i])

  fig.align_ylabels(axes)
  plt.savefig('voltage_gated_Na_and_K.pdf', transparent=True, dpi=500)
  plt.show()


def try_step_voltage_for_gNa():
  dt = 0.01
  step = 60
  vs, duration = bp.inputs.section_input(
    [-70.67647, -70.67647 + step,
     -70.67647, -70.67647 + step,
     -70.67647],
    durations=[1, 1.8,
               25, 1.8,
               10],
    return_length=True,
    dt=dt)
  runner = bp.DSRunner(SeparateNaK(1),
                       monitors=['INa', 'IK', 'IL', 'Ifb', 'gNa', 'gK',
                                 'm', 'h', 'n'],
                       inputs=['V', vs, 'iter', '='],
                       dt=dt)
  runner.run(duration)

  fig, gs = bp.visualize.get_figure(3, 1, 2, 6)
  fig.add_subplot(gs[0, 0])
  bp.visualize.line_plot(runner.mon.ts, runner.mon.gNa, legend='gNa')
  fig.add_subplot(gs[1, 0])
  bp.visualize.line_plot(runner.mon.ts, runner.mon.m, legend='m')
  fig.add_subplot(gs[2, 0])
  bp.visualize.line_plot(runner.mon.ts, runner.mon.h, legend='h', show=True)


def INa_inactivation():
  dt = 0.01
  vs, duration = bp.inputs.section_input(
    [-70.67647, -70.67647 + 8, -70.67647 + 44],
    durations=[1, 10, 20],
    return_length=True,
    dt=dt)
  runner = bp.DSRunner(SeparateNaK(1),
                       monitors=['INa', 'IK', 'IL', 'Ifb', 'gNa', 'gK',
                                 'm', 'h', 'n'],
                       inputs=['V', vs, 'iter', '='],
                       dt=dt)
  runner.run(duration)

  fig, gs = bp.visualize.get_figure(3, 2, 2, 5)
  fig.add_subplot(gs[0, 0])
  bp.visualize.line_plot(runner.mon.ts, runner.mon.Ifb, legend='Ifb')
  fig.add_subplot(gs[1, 0])
  bp.visualize.line_plot(runner.mon.ts, runner.mon.INa, legend='INa')
  fig.add_subplot(gs[2, 0])
  bp.visualize.line_plot(runner.mon.ts, runner.mon.IK, legend='IK')

  fig.add_subplot(gs[0, 1])
  bp.visualize.line_plot(runner.mon.ts, runner.mon.gNa, legend='gNa')
  bp.visualize.line_plot(runner.mon.ts, runner.mon.gK, legend='gK')
  fig.add_subplot(gs[1, 1])
  bp.visualize.line_plot(runner.mon.ts, runner.mon.m, legend='m')
  bp.visualize.line_plot(runner.mon.ts, runner.mon.h, legend='h')
  fig.add_subplot(gs[2, 1])
  bp.visualize.line_plot(runner.mon.ts, runner.mon.n, legend='n', show=True)


def INa_inactivation_1():
  dt = 0.01
  duration = 50
  lengths = [0, 5, 10, 20]

  all_vs = []
  for l in lengths:
    vs = bp.inputs.section_input([-70.67647, -70.67647 + 8, -70.67647 + 44], durations=[1, l, duration - l], dt=dt)
    all_vs.append(vs)
  all_vs = bm.vstack(all_vs).T

  runner = bp.DSRunner(SeparateNaK(len(lengths)),
                       monitors=['INa', 'IK', 'IL', 'Ifb', 'gNa', 'gK', 'm', 'h', 'n'],
                       inputs=['V', all_vs, 'iter', '='], dt=dt)
  runner.run(duration + 1)
  all_vs = all_vs.to_numpy()

  # fig, gs = bp.visualize.get_figure(2, 2, 3, 4)
  # for i in range(4):
  #   ax = fig.add_subplot(gs[i // 2, i % 2])
  #   ax.plot(runner.mon.ts, -runner.mon.INa[:, i], 'b')
  #   plt.title(f't = {lengths[i]} ms')
  #   ax.set_ylim([0, 1400])
  #   ax.tick_params('y', colors='b')
  #   ax.set_ylabel(r'$I_{\mathrm{Na}}$', color='b')
  #   ax = ax.twinx()
  #   ax.plot(runner.mon.ts, all_vs[:, i], 'r')
  #   ax.tick_params('y', colors='r')
  #   ax.set_ylim([-100, 100])
  #   ax.set_ylabel(r'Membrane potential', color='r')

  plt.rcParams.update({"font.size": 15})
  plt.rcParams['font.sans-serif'] = ['Times New Roman']
  fig, gs = bp.visualize.get_figure(6, 2, 0.8, 3)
  for i in range(4):
    row_i = (i // 2) * 3
    ax = fig.add_subplot(gs[row_i: row_i + 2, i % 2])
    idx = np.argmax(-runner.mon.INa[:, i])
    ax.plot(runner.mon.ts, -runner.mon.INa[:, i], 'b')
    ax.set_ylim([-100, 1400])
    ax.set_xlim([-1, duration + 1])
    plt.plot([-1, runner.mon.ts[idx]], [-runner.mon.INa[idx, i]] * 2, 'k--')
    if i % 2 == 0:
      ax.set_ylabel(r'$I_{\mathrm{Na}}$', color='b')
    ax.tick_params('y', colors='b')
    plt.text(20, 1000, f't = {lengths[i]} ms')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])

    ax = fig.add_subplot(gs[row_i + 2, i % 2])
    ax.plot(runner.mon.ts, all_vs[:, i], 'r')
    if i % 2 == 0:
      ax.set_ylabel(r'V', color='r')
    ax.tick_params('y', colors='r')
    ax.set_xlim([-1, duration + 1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if i // 2 == 1:
      ax.set_xlabel(r'$t$ (ms)')

  plt.savefig('Na_inactivation_example.pdf', transparent=True, dpi=500)
  plt.show()


def INa_inactivation_2():
  dt = 0.01
  duration = 50
  lengths = list(range(0, 26, 1))
  v_list = [25, 20, 10, -10, -20, -30]

  plt.rcParams.update({"font.size": 15})
  plt.rcParams['font.sans-serif'] = ['Times New Roman']

  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
  ax = fig.add_subplot(gs[0, 0])
  for v in v_list:
    all_vs = []
    for l in lengths:
      vs = bp.inputs.section_input([-70.67647, -70.67647 + v, -70.67647 + 44], durations=[1, l, duration - l], dt=dt)
      all_vs.append(vs)
    all_vs = bm.vstack(all_vs).T

    runner = bp.DSRunner(SeparateNaK(len(lengths)),
                         monitors=['INa', 'IK', 'IL', 'Ifb', 'gNa', 'gK', 'm', 'h', 'n'],
                         inputs=['V', all_vs, 'iter', '='], dt=dt)
    runner.run(duration)

    base = runner.mon.INa[:, 0].min()
    all_min = runner.mon.INa.min(axis=0)
    plt.plot(lengths, all_min / base, label=f'$v_1$={v} mV')
    if v == 25:
      plt.text(4.8, .19, r'$V_1$=25mV')
    elif v == 20:
      plt.text(15.8, 0.3, r'$V_1$=20mV')
    elif v == 10:
      plt.text(15.8, 0.68, r'$V_1$=10mV')
    elif v == -10:
      plt.text(15.8, 1.11, r'$V_1$=-10mV')
    elif v == -20:
      plt.annotate(r'$V_1$=-20mV', xy=(6., 1.19), xytext=(9.2, 1.06), arrowprops=dict(arrowstyle="->"))
    elif v == -30:
      plt.annotate(r'$V_1$=-30mV', xy=(3.19, 1.19), xytext=(5.12, 0.93), arrowprops=dict(arrowstyle="->"))
    else:
      raise ValueError
  plt.xlabel(r'$t$ [ms]')
  plt.ylabel(r'$\frac{(I_{Na})_{v_1}}{(I_{Na})_{v_0}}$')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.savefig('Na_inactivation_example2.pdf', transparent=True, dpi=500)
  plt.show()


def INa_inactivation_steady_state1():
  dt = 0.01
  steps = bm.linspace(-30, 50, 9)
  vs, duration = bp.inputs.section_input([-70.67647, -70.67647 + steps, -70.67647 + 44],
                                         durations=[5, 30, 20], dt=dt,
                                         return_length=True)
  runner = bp.DSRunner(SeparateNaK(len(steps)),
                       monitors=['INa', 'IK', 'IL', 'Ifb', 'gNa', 'gK', 'm', 'h', 'n'],
                       inputs=['V', vs, 'iter', '='],
                       dt=dt)
  runner.run(duration)
  vs = bm.as_numpy(vs)

  plt.rcParams.update({"font.size": 15})
  plt.rcParams['font.sans-serif'] = ['Times New Roman']
  fig, gs = bp.visualize.get_figure(9, 3, 1, 4)
  for i in range(9):
    row_i = (i // 3) * 3
    ax = fig.add_subplot(gs[row_i: row_i + 2, i % 3])
    ax.plot(runner.mon.ts, -runner.mon.INa[:, i], 'b')
    plt.text(10, 1500, f'$\\Delta v$ = {steps[i]} mV')
    ax.set_ylim([-100, 1700])
    ax.tick_params('y', colors='b')
    if i % 3 == 0:
      ax.set_ylabel(r'$I_{\mathrm{Na}}$', color='b')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])

    ax = fig.add_subplot(gs[row_i + 2, i % 3])
    ax.plot(runner.mon.ts, vs[:, i], 'r')
    ax.tick_params('y', colors='r')
    # ax.set_ylim([-120, 100])
    if i % 3 == 0: ax.set_ylabel('V', color='r')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if i // 3 == 2:
      ax.set_xlabel(r'$t$ (ms)')

  plt.savefig('Ina_inactivation_steady_state.pdf',
              transparent=True, dpi=500)
  plt.show()


def INa_inactivation_steady_state2():
  dt = 0.01
  steps = bm.linspace(-60, 60, 201)
  steps = bm.hstack([0, steps])
  vs, duration = bp.inputs.section_input([-70.67647, -70.67647 + steps, -70.67647 + 44],
                                         durations=[5, 30, 20], dt=dt,
                                         return_length=True)
  runner = bp.DSRunner(SeparateNaK(len(steps)),
                       monitors=['INa', 'IK', 'IL', 'Ifb', 'gNa', 'gK', 'm', 'h', 'n'],
                       inputs=['V', vs, 'iter', '='],
                       dt=dt)
  runner.run(duration)

  start = int(30 / dt)
  base = runner.mon.INa[start:, 0].min() - runner.mon.INa[-1, 0]
  scales = (runner.mon.INa[start:, 1:].min(axis=0) - runner.mon.INa[-1, 1:]) / base

  plt.rcParams.update({"font.size": 15})
  plt.rcParams['font.sans-serif'] = ['Times New Roman']
  fig, gs = bp.visualize.get_figure(1, 1, 9, 6)
  ax = fig.add_subplot(gs[0, 0])
  ax.plot(steps.numpy()[1:], scales)
  ax.set_xlabel(r'$\Delta V$ (mV)')
  # ax.set_ylabel(r'${I_{Na}(\Delta V)}/{I_{Na}(0)}$', )
  plt.text(-63, 0.5, r'${I_{Na}(\Delta V)}/{I_{Na}(0)}$', rotation=90)
  ax.yaxis.set_label_coords(-0.13, 0.5)
  plt.ylim([-scales.max() * .05, scales.max() * 1.05])
  plt.axvline(0, linestyle='--', color='grey')
  plt.axhline(1, linestyle='--', color='grey')
  ax = ax.twinx()
  plt.yticks(bm.linspace(0, scales.max(), 11).numpy(), np.around(np.linspace(0, 1, 11), 1))
  plt.text(60, 0.5, r'$h_\infty(V+\Delta V)$', rotation=90)
  # ax.set_ylabel(r'$h_\infty(V+\Delta V)$', )
  ax.yaxis.set_label_coords(1.18, 0.55)
  plt.ylim([-scales.max() * .05, scales.max() * 1.05])
  plt.savefig('Ina_inactivation_steady_state2.pdf',
              transparent=True, dpi=500)
  plt.show()


if __name__ == '__main__':
  pass
  separation_of_Na_and_K_currents()
  try_step_voltage2()
  try_step_voltage_for_gNa()
  INa_inactivation()
  INa_inactivation_1()
  INa_inactivation_2()
  INa_inactivation_steady_state1()
  INa_inactivation_steady_state2()
