import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15})
plt.rcParams['font.sans-serif'] = ['Times New Roman']


class DecisionMakingRateModel(bp.dyn.NeuDyn):
  def __init__(
      self, size, coherence, JE=0.2609, JI=0.0497, Jext=5.2e-4, I0=0.3255,
      gamma=6.41e-4, tau=100., tau_n=2., sigma_n=0.02, a=270., b=108., d=0.154,
      noise_freq=2400., method='exp_auto', **kwargs
  ):
    super(DecisionMakingRateModel, self).__init__(size, **kwargs)

    # 初始化参数
    self.coherence = coherence
    self.JE = JE
    self.JI = JI
    self.Jext = Jext
    self.I0 = I0
    self.gamma = gamma
    self.tau = tau
    self.tau_n = tau_n
    self.sigma_n = sigma_n
    self.a = a
    self.b = b
    self.d = d

    # 初始化变量
    self.s1 = bm.Variable(bm.zeros(self.num) + 0.15)
    self.s2 = bm.Variable(bm.zeros(self.num) + 0.15)
    self.r1 = bm.Variable(bm.zeros(self.num))
    self.r2 = bm.Variable(bm.zeros(self.num))
    self.mu0 = bm.Variable(bm.zeros(self.num))
    self.I1_noise = bm.Variable(bm.zeros(self.num))
    self.I2_noise = bm.Variable(bm.zeros(self.num))

    # 噪声输入的神经元
    self.noise1 = bp.neurons.PoissonGroup(self.num, freqs=noise_freq)
    self.noise2 = bp.neurons.PoissonGroup(self.num, freqs=noise_freq)

    # 定义积分函数
    self.integral = bp.odeint(self.derivative, method=method)

  @property
  def derivative(self):
    return bp.JointEq([self.ds1, self.ds2, self.dI1noise, self.dI2noise])

  def ds1(self, s1, t, s2, mu0):
    I1 = self.Jext * mu0 * (1. + self.coherence / 100.)
    x1 = self.JE * s1 - self.JI * s2 + self.I0 + I1 + self.I1_noise
    r1 = (self.a * x1 - self.b) / (1. - bm.exp(-self.d * (self.a * x1 - self.b)))
    return - s1 / self.tau + (1. - s1) * self.gamma * r1

  def ds2(self, s2, t, s1, mu0):
    I2 = self.Jext * mu0 * (1. - self.coherence / 100.)
    x2 = self.JE * s2 - self.JI * s1 + self.I0 + I2 + self.I2_noise
    r2 = (self.a * x2 - self.b) / (1. - bm.exp(-self.d * (self.a * x2 - self.b)))
    return - s2 / self.tau + (1. - s2) * self.gamma * r2

  def dI1noise(self, I1_noise, t, noise1):
    return (- I1_noise + noise1.spike * bm.sqrt(self.tau_n * self.sigma_n * self.sigma_n)) / self.tau_n

  def dI2noise(self, I2_noise, t, noise2):
    return (- I2_noise + noise2.spike * bm.sqrt(self.tau_n * self.sigma_n * self.sigma_n)) / self.tau_n

  def update(self):
    # 更新噪声神经元以产生新的随机发放
    self.noise1.update()
    self.noise2.update()

    # 更新s1、s2、I1_noise、I2_noise
    integral = self.integral(self.s1, self.s2, self.I1_noise, self.I2_noise, bp.share['t'],
                             mu0=self.mu0, noise1=self.noise1, noise2=self.noise2, dt=bp.share['dt'])
    self.s1.value, self.s2.value, self.I1_noise.value, self.I2_noise.value = integral

    # 用更新后的s1、s2计算r1、r2
    I1 = self.Jext * self.mu0 * (1. + self.coherence / 100.)
    x1 = self.JE * self.s1 + self.JI * self.s2 + self.I0 + I1 + self.I1_noise
    self.r1.value = (self.a * x1 - self.b) / (1. - bm.exp(-self.d * (self.a * x1 - self.b)))

    I2 = self.Jext * self.mu0 * (1. - self.coherence / 100.)
    x2 = self.JE * self.s2 + self.JI * self.s1 + self.I0 + I2 + self.I2_noise
    self.r2.value = (self.a * x2 - self.b) / (1. - bm.exp(-self.d * (self.a * x2 - self.b)))

    # 重置外部输入
    self.mu0[:] = 0.


def run_rate_model_coherence1():
  # 定义各个阶段的时长
  pre_stimulus_period = 100.
  stimulus_period = 2000.
  delay_period = 500.
  coherence = 25.6

  # 生成模型
  dmnet = DecisionMakingRateModel(1, coherence=coherence, noise_freq=2400.)

  # 定义电流随时间的变化
  inputs, total_period = bp.inputs.constant_input([(0., pre_stimulus_period),
                                                   (20., stimulus_period),
                                                   (0., delay_period)])
  # 运行数值模拟
  runner = bp.DSRunner(dmnet, monitors=['s1', 's2', 'r1', 'r2'], inputs=('mu0', inputs, 'iter'))
  runner.run(total_period)

  # 可视化
  fig, gs = bp.visualize.get_figure(2, 1, 2.25, 6)
  ax = fig.add_subplot(gs[0, 0])
  ax.plot(runner.mon.ts, runner.mon.s1, label='s1')
  ax.plot(runner.mon.ts, runner.mon.s2, label='s2')
  ax.axvline(pre_stimulus_period, 0., 1., linestyle='dashed', color=u'#444444')
  ax.axvline(pre_stimulus_period + stimulus_period, 0., 1., linestyle='dashed', color=u'#444444')
  ax.set_ylabel(r'Gating Variable')
  ax.set_xticks([])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.text(1800, 0.55, r'$s_1$')
  plt.text(1800, 0.10, r'$s_2$')
  ax = fig.add_subplot(gs[1, 0])
  ax.plot(runner.mon.ts, runner.mon.r1, label='r1')
  ax.plot(runner.mon.ts, runner.mon.r2, label='r2')
  ax.axvline(pre_stimulus_period, 0., 1., linestyle='dashed', color=u'#444444')
  ax.axvline(pre_stimulus_period + stimulus_period, 0., 1., linestyle='dashed', color=u'#444444')
  ax.set_xlabel(r'$t$ (ms)')
  ax.set_ylabel(r'Firing Rate')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.text(1800, 28, r'$r_1$')
  plt.text(1800, 8, r'$s_2$')
  plt.savefig('decision_making_rate_output_c={}.pdf'.format(coherence), transparent=True, dpi=500)
  # plt.show()


def run_rate_model_coherence2():
  # 定义各个阶段的时长
  pre_stimulus_period = 100.
  stimulus_period = 2000.
  delay_period = 500.
  coherence = -6.4

  # 生成模型
  dmnet = DecisionMakingRateModel(1, coherence=coherence, noise_freq=2400.)

  # 定义电流随时间的变化
  inputs, total_period = bp.inputs.constant_input([(0., pre_stimulus_period),
                                                   (20., stimulus_period),
                                                   (0., delay_period)])
  # 运行数值模拟
  runner = bp.DSRunner(dmnet, monitors=['s1', 's2', 'r1', 'r2'], inputs=('mu0', inputs, 'iter'))
  runner.run(total_period)

  # 可视化
  fig, gs = bp.visualize.get_figure(2, 1, 2.25, 6)
  ax = fig.add_subplot(gs[0, 0])
  ax.plot(runner.mon.ts, runner.mon.s1, label='s1')
  ax.plot(runner.mon.ts, runner.mon.s2, label='s2')
  ax.axvline(pre_stimulus_period, 0., 1., linestyle='dashed', color=u'#444444')
  ax.axvline(pre_stimulus_period + stimulus_period, 0., 1., linestyle='dashed', color=u'#444444')
  ax.set_ylabel(r'Gating Variable')
  ax.set_xticks([])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.text(1800, 0.55, r'$s_2$')
  plt.text(1800, 0.10, r'$s_1$')

  ax = fig.add_subplot(gs[1, 0])
  ax.plot(runner.mon.ts, runner.mon.r1, label='r1')
  ax.plot(runner.mon.ts, runner.mon.r2, label='r2')
  ax.axvline(pre_stimulus_period, 0., 1., linestyle='dashed', color=u'#444444')
  ax.axvline(pre_stimulus_period + stimulus_period, 0., 1., linestyle='dashed', color=u'#444444')
  ax.set_xlabel(r'$t$ (ms)')
  ax.set_ylabel(r'Firing Rate')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.text(1800, 28, r'$r_2$')
  plt.text(1800, 8, r'$s_1$')
  plt.savefig('decision_making_rate_output_c={}.pdf'.format(coherence), transparent=True, dpi=500)
  # plt.show()


def phase_plane():
  # 使用高精度float模式
  bp.math.enable_x64()
  bp.analysis.plotstyle.set_plot_schema(bp.analysis.stability.SADDLE_NODE, marker='*', markersize=15)

  def _analyze(coherence, mu0=20.):
    # 构相平面建分析器
    model = DecisionMakingRateModel(1, coherence=coherence)
    analyzer = bp.analysis.PhasePlane2D(
      model=model,
      target_vars={'s1': [0, 1], 's2': [0, 1]},
      fixed_vars={'I1_noise': 0., 'I2_noise': 0.},
      pars_update={'mu0': mu0},
      resolutions={'s1': 0.002, 's2': 0.002},
    )

    fig, gs = bp.visualize.get_figure(1, 1, 4.5, 4.5)
    ax = fig.add_subplot(gs[0, 0])
    # 画出向量场
    analyzer.plot_vector_field(plot_style=dict(color='lightgrey'))
    # 画出零增长等值线
    analyzer.plot_nullcline(coords=dict(s2='s2-s1'), x_style={'fmt': ':'}, y_style={'fmt': '--'})
    # 画出奇点
    analyzer.plot_fixed_point(tol_aux=2e-10)
    # 画出s1, s2的运动轨迹
    analyzer.plot_trajectory(
      {'s1': [0.1], 's2': [0.1]}, duration=2000., color='darkslateblue', linewidth=2, alpha=0.9,
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title('$c={}, \mu_0={}$'.format(coherence, mu0))
    # plt.savefig('decision_making_phase_plane_c={}_mu={}.pdf'.format(coherence, mu0), transparent=True, dpi=500)
    plt.show()

  _analyze(0, 0)
  _analyze(0, 20)
  _analyze(6.4, 20)
  _analyze(25.6, 20)
  _analyze(100, 20)


if __name__ == '__main__':
  run_rate_model_coherence1()
  run_rate_model_coherence2()
  phase_plane()
