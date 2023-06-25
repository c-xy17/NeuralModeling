import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15})
plt.rcParams['font.sans-serif'] = ['Times New Roman']


class GIF(bp.NeuGroupNS):
  def __init__(self, size, V_rest=-70., V_reset=-70., theta_inf=-50., theta_reset=-60.,
               R=20., tau=20., a=0., b=0.01, k1=0.2, k2=0.02, R1=0., R2=1., A1=0.,
               A2=0., **kwargs):
    # 初始化父类时计算了self.num供下文使用
    super(GIF, self).__init__(size=size, **kwargs)

    # 初始化参数
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.theta_inf = theta_inf
    self.theta_reset = theta_reset
    self.R = R
    self.tau = tau
    self.a = a
    self.b = b
    self.k1 = k1
    self.k2 = k2
    self.R1 = R1
    self.R2 = R2
    self.A1 = A1
    self.A2 = A2

    # 初始化变量
    self.V = bm.Variable(bm.zeros(self.num) + V_reset)
    self.theta = bm.Variable(bm.ones(self.num) * theta_inf)
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.I1 = bm.Variable(bm.zeros(self.num))
    self.I2 = bm.Variable(bm.zeros(self.num))

    # 定义积分器
    self.integral = bp.odeint(f=self.derivative, method='exp_auto')

  def dI1(self, I1, t):
    return - self.k1 * I1

  def dI2(self, I2, t):
    return - self.k2 * I2

  def dVth(self, V_th, t, V):
    return self.a * (V - self.V_rest) - self.b * (V_th - self.theta_inf)

  def dV(self, V, t, I1, I2, Iext):
    return (- (V - self.V_rest) + self.R * Iext + self.R * I1 + self.R * I2) / self.tau

  # 将所有微分方程联合为一个，以便同时积分
  @property
  def derivative(self):
    return bp.JointEq([self.dI1, self.dI2, self.dVth, self.dV])

  def update(self,):
    I1, I2, V_th, V = self.integral(self.I1, self.I2, self.theta, self.V, bp.share['t'],
                                    self.input, bp.share['dt'])  # 更新变量I1, I2, V
    spike = self.theta <= V  # 将大于阈值的神经元标记为发放了脉冲
    V = bm.where(spike, self.V_reset, V)  # 将发放了脉冲的神经元V置为V_reset，其余赋值为更新后的V
    I1 = bm.where(spike, self.R1 * I1 + self.A1, I1)  # 按照公式更新发放了脉冲的神经元的I1
    I2 = bm.where(spike, self.R2 * I2 + self.A2, I2)  # 按照公式更新发放了脉冲的神经元的I2
    reset_th = bm.logical_and(V_th < self.theta_reset, spike)  # 判断哪些神经元的V_th需要重置
    V_th = bm.where(reset_th, self.theta_reset, V_th)  # 将需要重置的神经元V_th重置为theta_reset

    # 将更新后的结果赋值给self.*
    self.spike.value = spike
    self.I1.value = I1
    self.I2.value = I2
    self.theta.value = V_th
    self.V.value = V
    self.input[:] = 0.  # 重置外界输入


def run_GIF():
  fig, gs = bp.visualize.get_figure(1, 2, 4, 6)

  # 模拟相位脉冲（phasic spiking）
  group = GIF(10, a=0.005, A1=0., A2=0.)
  runner = bp.DSRunner(group, monitors=['V', 'theta'], inputs=('input', 1.5), dt=0.01)
  runner(500)

  fig.add_subplot(gs[0, 0])
  bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V', zorder=10, show=False)
  bp.visualize.line_plot(runner.mon.ts, runner.mon.theta, legend='theta',
                         title='phasic spiking', show=False)

  # 模拟超极化爆发（hyperpolarization-induced bursting）
  group = GIF(10, a=0.03, A1=10., A2=-0.6)
  runner = bp.DSRunner(group, monitors=['V', 'theta'], inputs=('input', -1), dt=0.01)
  runner(500)

  fig.add_subplot(gs[0, 1])
  bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V', zorder=10, show=False)
  bp.visualize.line_plot(runner.mon.ts, runner.mon.theta, legend='theta',
                         title='hyperpolarization-induced bursting', show=True)


def plot_gallery():
  def _run(ax1, model, duration, I_ext, title=''):
    runner = bp.DSRunner(model,
                         inputs=('input', I_ext, 'iter'),
                         monitors=['V', 'theta'])
    runner.run(duration)
    ts = runner.mon.ts
    ax1.plot(ts, runner.mon.V[:, 0], label=r'$V$', linestyle='-')
    ax1.plot(ts, runner.mon.theta[:, 0], label=r'$\theta$', linestyle='--')
    ax1.set_ylabel('Potential (mV)')
    ax1.set_xlabel(r'$t$ (ms)')
    ax1.set_xlim(-0.1, ts[-1] + 0.1)
    if title: plt.title(title)
    # plt.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

  row, col = 7, 3
  fig, gs = bp.visualize.get_figure(row, col, 3, 5)

  # Tonic Spiking
  ax = fig.add_subplot(gs[0 // col, 0 % col])
  Iext, duration = bp.inputs.section_input((1.5,), (200.,), return_length=True)
  _run(ax, GIF(1), duration, Iext, 'A. Tonic Spiking')

  # Class 1 Excitability
  ax = fig.add_subplot(gs[1 // col, 1 % col])
  Iext, duration = bp.inputs.section_input([1. + 1e-6], [500.], return_length=True)
  _run(ax, GIF(1), duration, Iext, 'B. Class 1 Excitability')

  # Spike Frequency Adaptation
  ax = fig.add_subplot(gs[2 // col, 2 % col])
  Iext, duration = bp.inputs.section_input([2.], [200.], return_length=True)
  _run(ax, GIF(1, a=0.005), duration, Iext, 'C. Spike Frequency Adaptation')

  # Phasic Spiking
  ax = fig.add_subplot(gs[3 // col, 3 % col])
  Iext, duration = bp.inputs.section_input([1.5], [500.], return_length=True)
  _run(ax, GIF(1, a=0.005), duration, Iext, 'D. Phasic Spiking')

  # Accommodation
  ax = fig.add_subplot(gs[4 // col, 4 % col])
  Iext, duration = bp.inputs.section_input([1.5, 0., 0.5, 1., 1.5, 0.],
                                           [100., 500., 100., 100., 100., 100.],
                                           return_length=True)
  _run(ax, GIF(1, a=0.005), duration, Iext, 'E. Accommodation')

  # Threshold Variability
  ax = fig.add_subplot(gs[5 // col, 5 % col])
  Iext, duration = bp.inputs.section_input([1.5, 0., -1.5, 0., 1.5, 0.],
                                           [20., 180., 20., 20., 20., 140.],
                                           return_length=True)
  _run(ax, GIF(1, a=0.005), duration, Iext, 'F. Threshold Variability')

  # Rebound Spiking
  ax = fig.add_subplot(gs[6 // col, 6 % col])
  Iext, duration = bp.inputs.section_input([0., -3.5, 0.], [50., 750., 200.], return_length=True)
  _run(ax, GIF(1, a=0.005), duration, Iext, 'G. Rebound Spiking')

  # Class 2 Excitability
  ax = fig.add_subplot(gs[7 // col, 7 % col])
  Iext, duration = bp.inputs.section_input([2 * (1. + 1e-6)], [200.], return_length=True)
  neu = GIF(1, a=0.005)
  neu.theta[:] = -30.
  _run(ax, neu, duration, Iext, 'H. Class 2 Excitability')

  # Integrator
  ax = fig.add_subplot(gs[8 // col, 8 % col])
  Iext, duration = bp.inputs.section_input([1.5, 0., 1.5, 0., 1.5, 0., 1.5, 0.],
                                           [20., 10., 20., 250., 20., 30., 20., 30.],
                                           return_length=True)
  _run(ax, GIF(1, a=0.005), duration, Iext, 'I. Integrator')

  # Input Bistability
  ax = fig.add_subplot(gs[9 // col, 9 % col])
  Iext, duration = bp.inputs.section_input([1.5, 1.7, 1.5, 1.7],
                                           [100., 400., 100., 400.],
                                           return_length=True)
  _run(ax, GIF(1, a=0.005), duration, Iext, 'J. Input Bistability')

  # Hyperpolarization-induced Spiking
  ax = fig.add_subplot(gs[10 // col, 10 % col])
  Iext, duration = bp.inputs.section_input([-1.], [400.], return_length=True)
  neu = GIF(1, theta_reset=-60., theta_inf=-120.)
  neu.theta[:] = -50.
  _run(ax, neu, duration, Iext, 'K. Hyperpolarization-induced Spiking')

  # Hyperpolarization-induced Bursting
  ax = fig.add_subplot(gs[11 // col, 11 % col])
  Iext, duration = bp.inputs.section_input([-1.], [400.], return_length=True)
  neu = GIF(1, theta_reset=-60., theta_inf=-120., A1=10., A2=-0.6)
  neu.theta[:] = -50.
  _run(ax, neu, duration, Iext, 'L. Hyperpolarization-induced Bursting')

  # Tonic Bursting
  ax = fig.add_subplot(gs[12 // col, 12 % col])
  Iext, duration = bp.inputs.section_input([2.], [500.], return_length=True)
  _run(ax, GIF(1, a=0.005, A1=10., A2=-0.6), duration, Iext, 'M. Tonic Bursting')

  # Phasic Bursting
  ax = fig.add_subplot(gs[13 // col, 13 % col])
  Iext, duration = bp.inputs.section_input([1.5], [500.], return_length=True)
  neu = GIF(1, a=0.005, A1=10., A2=-0.6)
  _run(ax, neu, duration, Iext, 'N. Phasic Bursting')

  # Rebound Bursting
  ax = fig.add_subplot(gs[14 // col, 14 % col])
  Iext, duration = bp.inputs.section_input([0., -3.5, 0.],
                                           [100., 500., 400.],
                                           return_length=True)
  _run(ax, GIF(1, a=0.005, A1=10., A2=-0.6), duration, Iext, 'O. Rebound Bursting')

  # Mixed Mode
  ax = fig.add_subplot(gs[15 // col, 15 % col])
  Iext, duration = bp.inputs.section_input([2.], [500.], return_length=True)
  _run(ax, GIF(1, a=0.005, A1=5., A2=-0.3), duration, Iext, 'P. Mixed Mode')

  # Afterpotentials
  ax = fig.add_subplot(gs[16 // col, 16 % col])
  Iext, duration = bp.inputs.section_input((2., 0.), [15., 185.], return_length=True)
  _run(ax, GIF(1, a=0.005, A1=5., A2=-0.3), duration, Iext, 'Q. Afterpotentials')

  # Basal Bistability
  ax = fig.add_subplot(gs[17 // col, 17 % col])
  Iext, duration = bp.inputs.section_input([5., 0., 5., 0.],
                                           [10., 90., 10., 90.],
                                           return_length=True)
  _run(ax, GIF(1, A1=8., A2=-0.1), duration, Iext, 'R. Basal Bistability')

  # Preferred Frequency
  ax = fig.add_subplot(gs[18 // col, 18 % col])
  Iext, duration = bp.inputs.section_input([5., 0., 4., 0., 5., 0., 4., 0.],
                                           [10., 10., 10., 370., 10., 90., 10., 290.],
                                           return_length=True)
  _run(ax, GIF(1, a=0.005, A1=-3., A2=0.5), duration, Iext, 'S. Preferred Frequency')

  # Spike Latency
  ax = fig.add_subplot(gs[19 // col, 19 % col])
  Iext, duration = bp.inputs.section_input([8., 0.], [2., 48.], return_length=True)
  _run(ax, GIF(1, a=-0.08), duration, Iext, 'T. Spike Latency')

  # plt.savefig('GIF_gallery.pdf', dpi=500, transparent=True)
  plt.show()


def detailed_running():
  def _run(model, duration, I_ext, title=''):
    runner = bp.DSRunner(model,
                         inputs=('input', I_ext, 'iter'),
                         monitors=['V', 'theta', 'I1', 'I2'])
    runner.run(duration)

    fig, gs = bp.visualize.get_figure(5, 1, 0.9, 6.)
    ax1 = fig.add_subplot(gs[:3, 0])
    ax1.plot(runner.mon.ts, runner.mon.V[:, 0], label=r'$V$', linestyle='-')
    ax1.plot(runner.mon.ts, runner.mon.theta[:, 0], label=r'$\theta$', linestyle='--')
    ax1.set_ylabel('Potential (mV)')
    ax1.set_xlim(-0.1, runner.mon.ts[-1] + 0.1)
    plt.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax1 = fig.add_subplot(gs[3:, 0])
    ax1.plot(runner.mon.ts, runner.mon.I1, label=r'$I_1$', linestyle='-')
    ax1.plot(runner.mon.ts, runner.mon.I2, label=r'$I_2$', linestyle='--')
    ax1.plot(runner.mon.ts, bm.as_numpy(I_ext), label=r'$I$', linestyle='dotted')
    ax1.set_xlim(-0.1, runner.mon.ts[-1] + 0.1)
    ax1.set_xlabel(r'$t$ (ms)')
    ax1.set_ylabel('Current')
    plt.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    if title:
      plt.savefig(f'GIF_{title.replace(" ", "-")}.pdf', transparent=True, dpi=500)

  Iext, duration = bp.inputs.section_input((2., 0.), [15., 185.], return_length=True)
  _run(GIF(1, a=0.005, A1=5., A2=-0.3), duration, Iext, 'Afterpotentials')

  Iext, duration = bp.inputs.section_input(
    [5., 0., 4., 0., 5., 0., 4., 0.], [10., 10., 10., 370., 10., 90., 10., 290.], return_length=True)
  _run(GIF(1, a=0.005, A1=-3., A2=0.5), duration, Iext, 'Preferred Frequency')

  Iext, duration = bp.inputs.section_input([2.], [500.], return_length=True)
  _run(GIF(1, a=0.005, A1=10., A2=-0.6), duration, Iext, 'Tonic Bursting')

  Iext, duration = bp.inputs.section_input([0., -3.5, 0.], [50., 750., 200.], return_length=True)
  _run(GIF(1, a=0.005), duration, Iext, 'Rebound Spiking')

  plt.show()


if __name__ == '__main__':
  run_GIF()
  plot_gallery()
  detailed_running()
