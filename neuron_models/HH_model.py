import brainpy as bp
import brainpy.math as bm

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 15})
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


class HH(bp.dyn.NeuGroup):
  def __init__(self, size, ENa=50., gNa=120., EK=-77., gK=36.,
               EL=-54.387, gL=0.03, V_th=20., C=1.0, T=6.3):
    # 初始化
    super(HH, self).__init__(size=size)

    # 定义神经元参数
    self.ENa = ENa
    self.EK = EK
    self.EL = EL
    self.gNa = gNa
    self.gK = gK
    self.gL = gL
    self.C = C
    self.V_th = V_th
    self.Q10 = 3.
    self.T_base = 6.3
    self.phi = self.Q10 ** ((T - self.T_base) / 10)

    # 定义神经元变量
    self.V = bm.Variable(-70.68 * bm.ones(self.num))  # 膜电位
    self.m = bm.Variable(0.0266 * bm.ones(self.num))  # 离子通道m
    self.h = bm.Variable(0.772 * bm.ones(self.num))  # 离子通道h
    self.n = bm.Variable(0.235 * bm.ones(self.num))  # 离子通道n
    # 神经元接收到的输入电流
    self.input = bm.Variable(bm.zeros(self.num))
    # 神经元发放状态
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    # 神经元上次发放的时刻
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

    # 定义积分函数
    self.integral = bp.odeint(f=self.derivative, method='exp_auto')

  @property
  def derivative(self):
    return bp.JointEq([self.dV, self.dm, self.dh, self.dn])

  def dm(self, m, t, V):
    alpha = 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
    beta = 4.0 * bm.exp(-(V + 65) / 18)
    dmdt = alpha * (1 - m) - beta * m
    return self.phi * dmdt

  def dh(self, h, t, V):
    alpha = 0.07 * bm.exp(-(V + 65) / 20.)
    beta = 1 / (1 + bm.exp(-(V + 35) / 10))
    dhdt = alpha * (1 - h) - beta * h
    return self.phi * dhdt

  def dn(self, n, t, V):
    alpha = 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
    beta = 0.125 * bm.exp(-(V + 65) / 80)
    dndt = alpha * (1 - n) - beta * n
    return self.phi * dndt


  f_gNa = lambda self, m, h: (self.gNa * m ** 3.0 * h)
  f_gK = lambda self, n: (self.gK * n ** 4.0)

  def dV(self, V, t, m, h, n):
    I_Na = (self.gNa * m ** 3.0 * h) * (V - self.ENa)
    I_K = (self.gK * n ** 4.0) * (V - self.EK)
    I_leak = self.gL * (V - self.EL)
    dVdt = (- I_Na - I_K - I_leak + self.input) / self.C
    return dVdt

  # 更新函数：每个时间步都会运行此函数完成变量更新
  def update(self, tdi):
    t, dt = tdi.t, tdi.dt
    # 更新下一时刻变量的值
    V, m, h, n = self.integral(self.V, self.m, self.h, self.n, t, dt=dt)
    # 判断神经元是否产生膜电位
    self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
    # 更新神经元发放的时间
    self.t_last_spike.value = bm.where(self.spike, t, self.t_last_spike)
    self.V.value = V
    self.m.value = m
    self.h.value = h
    self.n.value = n
    self.input[:] = 0.  # 重置神经元接收到的输入



def simple_run():
  current_sizes = [1., 2., 4., 8., 10., 15.]
  currents, length = bp.inputs.section_input(
    values=[0., bm.asarray(current_sizes), 0.],
    durations=[10, 2, 25],
    return_length=True
  )

  hh = HH(currents.shape[1])
  runner = bp.DSRunner(hh,
                       monitors=['V', 'm', 'h', 'n'],
                       inputs=['input', currents, 'iter'])
  runner.run(length)

  # 可视化
  # line_styles = [(0, (1, 1)), (0, (5, 5)), (0, (5, 1)), 'dotted', 'dashed', 'solid', ]
  line_styles = ['solid', 'dashed', 'dotted', ]
  colors = ['tab:gray', 'black']
  for i in range(hh.num):
    plt.plot(runner.mon.ts, runner.mon.V[:, i],
             label=f'I={current_sizes[i]} mA',
             linestyle=line_styles[i % 3],
             color=colors[i // 3]
             )
  plt.ylabel('V (mV)')
  # 将电流变化画在膜电位变化的下方
  plt.plot(runner.mon.ts,
           bm.where(currents[:, -1] > 0, 10., 0.).to_numpy() - 90,
           label='Current',
           linestyle=(0, (1, 1)))
  plt.tight_layout()
  plt.legend()
  plt.show()


def visualize_hh_responses_in_book():
  # currents
  current_sizes = [1., 2., 4., 8., 10., 15.]
  currents, length = bp.inputs.section_input(
    values=[0., bm.asarray(current_sizes), 0.],
    durations=[10, 2, 25],
    return_length=True
  )

  # simulation
  hh = HH(currents.shape[1])
  runner = bp.DSRunner(hh, monitors=['V', 'm', 'h', 'n'],
                       inputs=['input', currents, 'iter'])
  runner.run(length)

  # visualization
  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
  ax = fig.add_subplot(gs[0, 0])
  plt.plot(runner.mon.ts, runner.mon.V[:, 0])
  plt.annotate(f'I={current_sizes[0]} mA', xy=(12.2, -69), xytext=(14.2, -83),
               arrowprops=dict(arrowstyle="->"))
  plt.plot(runner.mon.ts, runner.mon.V[:, 1])
  plt.annotate(f'I={current_sizes[1]} mA', xy=(11.7, -68.5), xytext=(1.5, -63),
               arrowprops=dict(arrowstyle="->"))
  plt.plot(runner.mon.ts, runner.mon.V[:, 2])
  plt.annotate(f'I={current_sizes[2]} mA', xy=(16.3, -61), xytext=(20.5, -55),
               arrowprops=dict(arrowstyle="->"))
  plt.plot(runner.mon.ts, runner.mon.V[:, 3])
  plt.annotate(f'I={current_sizes[3]} mA', xy=(12.9, -4), xytext=(2.9, -1),
               arrowprops=dict(arrowstyle="->"))
  plt.plot(runner.mon.ts, runner.mon.V[:, 4])
  plt.annotate(f'I={current_sizes[4]} mA', xy=(12.5, 8), xytext=(2.5, 11),
               arrowprops=dict(arrowstyle="->"))
  plt.plot(runner.mon.ts, runner.mon.V[:, 5])
  plt.annotate(f'I={current_sizes[5]} mA', xy=(12, 19), xytext=(2, 22),
               arrowprops=dict(arrowstyle="->"))
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.ylabel(r'$V$ (mV)')
  plt.xlabel(r'$t$ (ms)')
  # 将电流变化画在膜电位变化的下方
  plt.plot(runner.mon.ts, bm.where(currents[:, -1] > 0, 10., 0.).to_numpy() - 90)
  plt.annotate('Current', xy=(30, -87), xytext=(30, -87), )
  plt.savefig('HH_responses_with_different_currents.pdf',
              transparent=True, dpi=500)
  plt.show()


def visualize_constant_current_response_in_book():
  # currents
  currents, length = bp.inputs.section_input(values=[0., 10., 0.],
                                             durations=[10, 50, 10],
                                             return_length=True)
  # simulation
  hh = HH(1)
  runner = bp.DSRunner(hh, monitors=['V', 'm', 'h', 'n'],
                       inputs=['input', currents, 'iter'])
  runner.run(length)

  # visualization
  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
  ax = fig.add_subplot(gs[0, 0])
  plt.plot(runner.mon.ts, runner.mon.V, label='V')
  plt.plot(runner.mon.ts, currents - 90, label='Current (I=5mA)')
  plt.xlabel(r'$t$ (ms)')
  plt.ylabel(r'$V$ (mV)')
  plt.text(0, -67, r'$V$')
  plt.text(0, -88, r'$I$=5mA')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.savefig('HH_responses_with_constant_current.pdf', transparent=True, dpi=500)
  plt.show()


def conductance_during_action_potential():
  # currents
  currents, length = bp.inputs.section_input(values=[0., 10., 0.], durations=[10, 5, 20], return_length=True)
  # simulation
  hh = HH(1)
  runner = bp.DSRunner(hh, monitors=['V', 'm', 'h', 'n'],
                       inputs=['input', currents, 'iter'])
  runner.run(length)

  # visualization
  fig, gs = bp.visualize.get_figure(3, 1, 2, 8)

  ax = fig.add_subplot(gs[0, 0])
  plt.plot(runner.mon.ts, runner.mon.V, lw=2)
  plt.ylabel(r'$V$ (mV)')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  ax = fig.add_subplot(gs[1, 0])
  plt.plot(runner.mon.ts, hh.f_gNa(runner.mon.m, runner.mon.h), label='gNa', lw=2)
  plt.plot(runner.mon.ts, hh.f_gK(runner.mon.n), label='gK',  lw=2)
  plt.text(13.8, 33, 'gNa')
  plt.text(16.5, 10, 'gK')
  plt.ylabel('Conductance')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  ax = fig.add_subplot(gs[2, 0])
  plt.plot(runner.mon.ts, runner.mon.m, label='m', lw=2)
  plt.plot(runner.mon.ts, runner.mon.h, label='h',  lw=2)
  plt.plot(runner.mon.ts, runner.mon.n, label='n',  lw=2)
  plt.text(14.5, 0.94, 'm')
  plt.text(26.1, 0.60, 'h')
  plt.text(26.1, 0.31, 'n')
  plt.ylabel('Channel')
  plt.xlabel(r'$t$ (ms)')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  plt.savefig('HH_conductance_during_AP.pdf', transparent=True, dpi=500)
  plt.show()


if __name__ == '__main__':
  pass
  # simple_run()
  visualize_hh_responses_in_book()
  visualize_constant_current_response_in_book()
  conductance_during_action_potential()

