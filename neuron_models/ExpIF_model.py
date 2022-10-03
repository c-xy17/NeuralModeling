import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 15})
plt.rcParams['font.sans-serif'] = ['Times New Roman']


class ExpIF(bp.dyn.NeuGroup):
  def __init__(self, size, V_rest=-65., V_reset=-68., V_th=20., V_T=-60., delta_T=1.,
               R=1., tau=10., tau_ref=2., method='exp_euler'):
    # 初始化父类
    super(ExpIF, self).__init__(size=size)

    # 初始化参数
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.V_T = V_T
    self.delta_T = delta_T
    self.R = R
    self.tau = tau
    self.tau_ref = tau_ref

    # 初始化变量
    self.V = bm.Variable(bm.random.randn(self.num) + V_reset)
    self.input = bm.Variable(bm.zeros(self.num))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)  # 上一次脉冲发放时间
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))  # 是否处于不应期
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))  # 脉冲发放状态

    # 使用指数欧拉方法进行积分
    self.integral = bp.odeint(f=self.derivative, method=method)

  # 定义膜电位关于时间变化的微分方程
  def derivative(self, V, t, Iext):
    exp_v = self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
    dvdt = (- (V - self.V_rest) + exp_v + self.R * Iext) / self.tau
    return dvdt

  def update(self, tdi):
    # 以数组的方式对神经元进行更新
    refractory = (tdi.t - self.t_last_spike) <= self.tau_ref  # 判断神经元是否处于不应期
    V = self.integral(self.V, tdi.t, self.input, tdi.dt)  # 根据时间步长更新膜电位
    V = bm.where(refractory, self.V, V)  # 若处于不应期，则返回原始膜电位self.V，否则返回更新后的膜电位V
    spike = V > self.V_th  # 将大于阈值的神经元标记为发放了脉冲
    self.spike.value = spike  # 更新神经元脉冲发放状态
    self.t_last_spike.value = bm.where(spike, tdi.t, self.t_last_spike)  # 更新最后一次脉冲发放时间
    self.V.value = bm.where(spike, self.V_reset, V)  # 将发放了脉冲的神经元膜电位置为V_reset，其余不变
    self.refractory.value = bm.logical_or(refractory, spike)  # 更新神经元是否处于不应期
    self.input[:] = 0.  # 重置外界输入


def run_ExpIF():
  # 运行ExpIF模型
  group = ExpIF(1)
  runner = bp.dyn.DSRunner(group, monitors=['V'], inputs=('input', 5.), dt=0.01)
  runner(500)

  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
  fig.add_subplot(gs[0, 0])
  # 结果可视化
  plt.plot(runner.mon.ts, runner.mon.V)
  plt.xlabel(r'$t$ (ms)')
  plt.ylabel(r'$V$ (mV)')
  # plt.savefig('ExpIF_output2.png', transparent=True, dpi=500)
  plt.show()

def effect_of_delta_T_v1():
  duration = 200
  I = 6.

  fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6), sharex=True, sharey=True)

  neu1 = ExpIF(1, delta_T=5.)
  neu1.V[:] = bm.array([-68.])
  runner = bp.DSRunner(neu1, monitors=['V', 'spike'], inputs=('input', I), dt=0.01)
  runner(duration)
  runner.mon.V = np.where(runner.mon.spike, neu1.V_th, runner.mon.V)
  ax1.plot(runner.mon.ts, runner.mon.V, color=u'#1f77b4', label='delta_T=5')
  ax1.set_ylabel('V (mV)')
  ax1.legend(loc='upper right')

  neu1 = ExpIF(1, delta_T=1.)
  neu1.V[:] = bm.array([-68.])
  runner = bp.DSRunner(neu1, monitors=['V', 'spike'], inputs=('input', I), dt=0.01)
  runner(duration)
  runner.mon.V = np.where(runner.mon.spike, neu1.V_th, runner.mon.V)
  ax2.plot(runner.mon.ts, runner.mon.V, color=u'#ff7f0e', label='delta_T=1')
  ax2.set_ylabel('V (mV)')
  ax2.legend(loc='upper right')

  neu1 = ExpIF(1, delta_T=0.02)
  neu1.V[:] = bm.array([-68.])
  runner = bp.DSRunner(neu1, monitors=['V', 'spike'], inputs=('input', I), dt=0.005)
  runner(duration)
  runner.mon.V = np.where(runner.mon.spike, neu1.V_th, runner.mon.V)
  ax3.plot(runner.mon.ts, runner.mon.V, color=u'#d62728', label='delta_T=0.02')
  ax3.set_ylabel('V (mV)')
  ax3.legend(loc='upper right')

  ax3.set_xlabel('t (ms)')

  plt.tight_layout()
  plt.subplots_adjust(hspace=0.)
  plt.show()


def effect_of_delta_T_v2():
  bm.enable_x64()
  neu1 = ExpIF(3, delta_T=bm.asarray([5., 1., 0.02]), method='exp_euler')
  neu1.V[:] = -68.
  runner = bp.DSRunner(neu1,
                       monitors=['V', 'spike'],
                       inputs=('input', 8.),
                       dt=0.001)
  runner(30.)

  fig, gs = bp.visualize.get_figure(1, 1, 3, 8)
  ax = fig.add_subplot(gs[0, 0])
  runner.mon.V = np.where(runner.mon.spike, neu1.V_th, runner.mon.V)
  ax.plot(runner.mon.ts, runner.mon.V[:, 0], label=r'$\Delta_T$=5.')
  ax.plot(runner.mon.ts, runner.mon.V[:, 1], label=r'$\Delta_T$=1.')
  ax.plot(runner.mon.ts, runner.mon.V[:, 2], label=r'$\Delta_T$=0.02')
  plt.text(9.7, -20, r'$\Delta_T$=0.02')
  plt.text(15.5, -15, r'$\Delta_T$=1')
  plt.text(19, -10, r'$\Delta_T$=5')
  ax.set_ylabel('V [mV]')
  ax.set_xlabel('Time [ms]')
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  # plt.savefig('ExpIF_delta_T.png', transparent=True, dpi=500)
  plt.show()


def dvdt():

  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
  ax = fig.add_subplot(gs[0, 0])
  expif = ExpIF(1)
  Vs = np.linspace(-80, -50, 500)
  y1 = - 1 / expif.tau * (Vs - expif.V_rest)
  x2 = np.ones(500) * expif.V_T
  y2 = np.linspace(-1, 6, 500)

  plt.plot(Vs, np.zeros(500), linewidth=1, color=u'#333333')
  plt.plot(Vs, y1, '--', color='grey')
  plt.plot(x2, y2, '--', color='grey')

  expif = ExpIF(1, delta_T=5.)
  dvdts = expif.derivative(Vs, 0., 0.)
  plt.plot(Vs, dvdts, label=r'$\Delta_T$=5')

  expif = ExpIF(1, delta_T=1.)
  dvdts = expif.derivative(Vs, 0., 0.)
  plt.plot(Vs, dvdts, label=r'$\Delta_T$=1.0')

  expif = ExpIF(1, delta_T=0.2)
  dvdts = expif.derivative(Vs, 0., 0.)
  plt.plot(Vs, dvdts, label=r'$\Delta_T$=0.2')

  expif = ExpIF(1, delta_T=0.05)
  dvdts = expif.derivative(Vs, 0., 0.)
  plt.plot(Vs, dvdts, label=r'$\Delta_T$=0.05')
  plt.text(-54, 0.443, r'$\Delta_T$=0.05')
  plt.text(-56.6, 1.416, r'$\Delta_T$=0.2')
  plt.text(-58.8, 2.72, r'$\Delta_T$=1')
  plt.text(-62.5, 4.02, r'$\Delta_T$=5')

  plt.xlim(-80, -50)
  plt.ylim(-1, 6)
  plt.xlabel(r'$V$ (mV)')
  plt.ylabel('dVdt')
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  # plt.savefig('ExpIF_dVdt_vs_delta_T.png', transparent=True, dpi=500)

  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
  ax = fig.add_subplot(gs[0, 0])

  expif = ExpIF(1)
  Vs = np.linspace(-80, -40, 500)
  y1 = - 1 / expif.tau * (Vs - expif.V_rest)
  y2 = np.linspace(-3, 6, 500)

  plt.plot(Vs, np.zeros(500), linewidth=1, color=u'#333333')
  plt.plot(Vs, y1, '--', color='grey')

  expif = ExpIF(1, delta_T=0.2, V_T=-70)
  x2 = np.ones(500) * expif.V_T
  plt.plot(x2, y2, '--', color='grey')
  dvdts = expif.derivative(Vs, 0., 0.)
  plt.plot(Vs, dvdts, label=r'$V_T$=-70')

  expif = ExpIF(1, delta_T=0.2, V_T=-60)
  x2 = np.ones(500) * expif.V_T
  plt.plot(x2, y2, '--', color='grey')
  dvdts = expif.derivative(Vs, 0., 0.)
  plt.plot(Vs, dvdts, label=r'$V_T$=-60')

  expif = ExpIF(1, delta_T=0.2, V_T=-50)
  x2 = np.ones(500) * expif.V_T
  plt.plot(x2, y2, '--', color='grey')
  dvdts = expif.derivative(Vs, 0., 0.)
  plt.plot(Vs, dvdts, label=r'$V_T$=-50', color=u'#d62728')
  plt.text(-48.94, -1, r'$V_T$=-50')
  plt.text(-58.9, 0.58, r'$V_T$=-60')
  plt.text(-69., 1.92, r'$V_T$=-70')

  plt.xlim(-80, -40)
  plt.ylim(-3, 6)
  plt.xlabel(r'$V$ (mV)')
  plt.ylabel('dVdt')
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  # plt.savefig('ExpIF_dVdt_vs_VT.png', transparent=True, dpi=500)

  plt.show()


def phase_plane():
  bm.enable_x64()

  for I in [0., 20.]:
    fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.)
    ax = fig.add_subplot(gs[0, 0])
    pp = bp.analysis.PhasePlane1D(
      model=ExpIF(1),
      target_vars={'V': [-80, -50]},
      pars_update={'Iext': I},
      resolutions=0.01,
    )
    pp.plot_vector_field()
    pp.plot_fixed_point()
    plt.title(f'Input = {I}')
    plt.xlabel(r'$V$')
    plt.ylim(-2, 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plt.savefig(f'ExpIF_dVdt_I={int(I):d}.png', transparent=True, dpi=500)

  plt.show()


if __name__ == '__main__':
  run_ExpIF()
  effect_of_delta_T_v2()
  dvdt()
  phase_plane()

