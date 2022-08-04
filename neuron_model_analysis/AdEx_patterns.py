import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np


class AdEx(bp.dyn.NeuGroup):
  def __init__(self, size, V_rest=-70., V_reset=-65., V_th=0., V_T=-50., delta_T=2., a=1.,
               b=2.5, R=0.5, tau=10., tau_w=30.):
    # 初始化父类
    super(AdEx, self).__init__(size=size)

    # 初始化参数
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.V_T = V_T
    self.delta_T = delta_T
    self.a = a
    self.b = b
    self.tau = tau
    self.tau_w = tau_w
    self.R = R

    # 初始化变量
    self.V = bm.Variable(bm.zeros(self.num) + V_reset)
    self.w = bm.Variable(bm.zeros(self.num))
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))  # 脉冲发放状态

    # 定义积分器
    self.integral = bp.odeint(f=self.derivative, method='exp_auto')

  def dV(self, V, t, w, Iext):
    _tmp = self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
    dVdt = (- V + self.V_rest + _tmp - self.R * w + self.R * Iext) / self.tau
    return dVdt

  def dw(self, w, t, V):
    dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
    return dwdt

  # 将两个微分方程联合为一个，以便同时积分
  @property
  def derivative(self):
    return bp.JointEq([self.dV, self.dw])

  def update(self, _t, _dt):
    # 以数组的方式对神经元进行更新
    V, w = self.integral(self.V, self.w, _t, self.input, dt=_dt)  # 更新膜电位V和权重值w
    spike = V > self.V_th  # 将大于阈值的神经元标记为发放了脉冲
    self.spike.value = spike  # 更新神经元脉冲发放状态
    self.V.value = bm.where(spike, self.V_reset, V)  # 将发放了脉冲的神经元膜电位置为V_reset，其余不变
    self.w.value = bm.where(spike, w + self.b, w)  # 发放了脉冲的神经元 w = w + b
    self.input[:] = 0.  # 重置外界输入


def subplot(i, neu, title=None, input=('input', 65.), duration=400):
  runner = bp.DSRunner(neu, monitors=['V', 'w', 'spike'], inputs=input)
  runner(duration)

  ax1 = plt.subplot(2, 3, i)
  runner.mon.V = np.where(runner.mon.spike, 20., runner.mon.V)
  ax1.plot(runner.mon.ts, runner.mon.V, label='V')
  ax1.set_ylim(- 65, 15)
  ax1.set_xlabel('t (ms)')
  if i % 3 == 1:
    ax1.set_ylabel('V')

  ax2 = ax1.twinx()
  ax2.plot(runner.mon.ts, runner.mon.w, label='w', color=u'#ff7f0e')
  w_min, w_max = min(runner.mon.w), max(runner.mon.w)
  ax2.set_ylim(w_max - (w_max - w_min) * 3, w_max + 5)
  if i % 3 == 0:
    ax2.set_ylabel('w')

  handles1, labels1 = ax1.get_legend_handles_labels()
  handles2, labels2 = ax2.get_legend_handles_labels()
  plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
  plt.title(title)


def vt_plot(i, neu, title=None, input=('input', 65.), duration=400):
  runner = bp.DSRunner(neu, monitors=['V', 'w', 'spike'], inputs=input)
  runner(duration)

  ax1 = plt.subplot()
  runner.mon.V = np.where(runner.mon.spike, 0., runner.mon.V)
  ax1.plot(runner.mon.ts, runner.mon.V, label='V')
  ax1.set_ylim(- 65, 15)
  ax1.set_xlabel('t (ms)')

  ax2 = ax1.twinx()
  ax2.plot(runner.mon.ts, runner.mon.w, label='w', color=u'#ff7f0e')
  w_min, w_max = min(runner.mon.w), max(runner.mon.w)
  ax2.set_ylim(w_max - (w_max - w_min) * 3, w_max + 5)

  handles1, labels1 = ax1.get_legend_handles_labels()
  handles2, labels2 = ax2.get_legend_handles_labels()
  plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
  plt.title(title)
  plt.show()


# plt.figure(figsize=(12, 6))
#
# subplot(1, AdEx(1, tau=20., a=0., tau_w=30., b=60., V_reset=-55.), title='Tonic Spiking')
# subplot(2, AdEx(1, tau=20., a=0., tau_w=100., b=5., V_reset=-55.), title='Adaptation')
# subplot(3, AdEx(1, tau=5., a=0.5, tau_w=100., b=7., V_reset=-51.), title='Initial Bursting')
# subplot(4, AdEx(1, tau=5., a=-0.5, tau_w=100., b=7., V_reset=-47.), title='Bursting')
# subplot(5, AdEx(1, tau=10., a=1., tau_w=100., b=10., V_reset=-60.), title='Transient Spiking', input=('input', 55.))
# subplot(6, AdEx(1, tau=5., a=-1., tau_w=100., b=5., V_reset=-60.), title='Delayed Spiking', input=('input', 25.))
# # subplot(7, AdEx(1, tau=9.9, a=-0.5, tau_w=100., b=7., V_reset=-46.), title='irregular')
#
# plt.tight_layout()
# plt.show()


vt_plot(1, AdEx(1, tau=20., a=0., tau_w=30., b=60., V_reset=-55.), title='Tonic Spiking')
vt_plot(2, AdEx(1, tau=20., a=0., tau_w=100., b=5., V_reset=-55.), title='Adaptation')
vt_plot(3, AdEx(1, tau=5., a=0.5, tau_w=100., b=7., V_reset=-51.), title='Initial Bursting')
vt_plot(4, AdEx(1, tau=5., a=-0.5, tau_w=100., b=7., V_reset=-47.), title='Bursting')
vt_plot(5, AdEx(1, tau=10., a=1., tau_w=100., b=10., V_reset=-60.), title='Transient Spiking', input=('input', 55.))
vt_plot(6, AdEx(1, tau=5., a=-1., tau_w=100., b=5., V_reset=-60.), title='Delayed Spiking', input=('input', 25.))
# vt_plot(7, AdEx(1, tau=9.9, a=-0.5, tau_w=100., b=7., V_reset=-46.), title='irregular')
