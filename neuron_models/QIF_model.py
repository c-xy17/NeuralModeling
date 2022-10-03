import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15})
plt.rcParams['font.sans-serif'] = ['Times New Roman']


class QIF(bp.dyn.NeuGroup):
  def __init__(self, size, V_rest=-65., V_reset=-68., V_th=-0., V_c=-50.0, a_0=.07, R=1., tau=10., t_ref=5., **kwargs):
    # 初始化父类
    super(QIF, self).__init__(size=size, **kwargs)

    # 初始化参数
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.V_c = V_c
    self.a_0 = a_0
    self.R = R
    self.tau = tau
    self.t_ref = t_ref  # 不应期时长

    # 初始化变量
    self.V = bm.Variable(bm.random.randn(self.num) + V_reset)
    self.input = bm.Variable(bm.zeros(self.num))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)  # 上一次脉冲发放时间
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))  # 是否处于不应期
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))  # 脉冲发放状态

    # 使用指数欧拉方法进行积分
    self.integral = bp.odeint(f=self.derivative, method='exp_auto')

  # 定义膜电位关于时间变化的微分方程
  def derivative(self, V, t, Iext):
    dvdt = (self.a_0 * (V - self.V_rest) * (V - self.V_c) + self.R * Iext) / self.tau
    return dvdt

  def update(self, tdi):
    _t, _dt = tdi.t, tdi.dt
    # 以数组的方式对神经元进行更新
    refractory = (_t - self.t_last_spike) <= self.t_ref  # 判断神经元是否处于不应期
    V = self.integral(self.V, _t, self.input, dt=_dt)  # 根据时间步长更新膜电位
    V = bm.where(refractory, self.V, V)  # 若处于不应期，则返回原始膜电位self.V，否则返回更新后的膜电位V
    spike = V > self.V_th  # 将大于阈值的神经元标记为发放了脉冲
    self.spike[:] = spike  # 更新神经元脉冲发放状态
    self.t_last_spike[:] = bm.where(spike, _t, self.t_last_spike)  # 更新最后一次脉冲发放时间
    self.V[:] = bm.where(spike, self.V_reset, V)  # 将发放了脉冲的神经元膜电位置为V_reset，其余不变
    self.refractory[:] = bm.logical_or(refractory, spike)  # 更新神经元是否处于不应期
    self.input[:] = 0.  # 重置外界输入


def run_QIF():
  # 运行QIF模型
  group = QIF(1)
  runner = bp.DSRunner(group, monitors=['V'], inputs=('input', 6.))
  runner(500)  # 运行时长为500ms
  # 结果可视化
  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
  fig.add_subplot(gs[0, 0])
  plt.plot(runner.mon.ts, runner.mon.V)
  plt.xlabel(r'$t$ (ms)')
  plt.ylabel(r'$V$ (mV)')
  # plt.savefig('QIF_output2.png', transparent=True, dpi=500)
  plt.show()


def QIF_input_threshold():
  def QIF_plot(input, duration, color):
    neu = QIF(1)
    neu.V[:] = bm.array([-68.])
    runner = bp.DSRunner(neu, monitors=['V'], inputs=('input', input))
    runner(duration)
    plt.plot(runner.mon.ts, runner.mon.V, color=color, label='input={}'.format(input))

  inputs = [0., 3., 4., 5.]
  colors = [u'#2ca02c', u'#d62728', u'#1f77b4', u'#ff7f0e']
  dur = 500

  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
  fig.add_subplot(gs[0, 0])
  for i in range(len(inputs)):
    QIF_plot(inputs[i], dur, colors[i])

  plt.text(104, -25, 'input=5.')
  plt.text(462, -25, 'input=4.')
  plt.annotate('input=3.', xy=(144, -61.5), xytext=(110, -45),
               arrowprops=dict(arrowstyle="->"))
  plt.annotate('input=0.', xy=(237.5, -65.15), xytext=(215, -50),
               arrowprops=dict(arrowstyle="->"))
  plt.xlabel(r'$t$ (ms)')
  plt.ylabel(r'$V$ (mV)')
  plt.xlim(-1, 541)
  # plt.savefig('QIF_input_threshold.png', transparent=True, dpi=500)
  plt.show()


def a0_effect():
  duration = 500

  fig, gs = bp.visualize.get_figure(3, 1, 1.5, 6)

  neu1 = QIF(1, a_0=0.005)
  neu1.V[:] = bm.array([-68.])
  runner = bp.DSRunner(neu1, monitors=['V'], inputs=('input', 5.))
  runner(duration)
  ax1 = fig.add_subplot(gs[0, 0])
  ax1.plot(runner.mon.ts, runner.mon.V, color=u'#d62728', label=r'$a_0$=0.005')
  ax1.set_ylabel(r'$V$ (mV)')
  ax1.legend(loc='upper right')
  ax1.spines['top'].set_visible(False)
  ax1.spines['right'].set_visible(False)

  neu1 = QIF(1, a_0=0.045)
  neu1.V[:] = bm.array([-68.])
  runner = bp.DSRunner(neu1, monitors=['V'], inputs=('input', 5.))
  runner(duration)
  ax2 = fig.add_subplot(gs[1, 0])
  ax2.plot(runner.mon.ts, runner.mon.V, color=u'#1f77b4', label=r'$a_0$=0.045')
  ax2.set_ylabel(r'$V$ (mV)')
  ax2.legend(loc='upper right')
  ax2.spines['top'].set_visible(False)
  ax2.spines['right'].set_visible(False)

  neu1 = QIF(1, a_0=0.08)
  neu1.V[:] = bm.array([-68.])
  runner = bp.DSRunner(neu1, monitors=['V'], inputs=('input', 5.))
  runner(duration)
  ax3 = fig.add_subplot(gs[2, 0])
  ax3.plot(runner.mon.ts, runner.mon.V, color=u'#ff7f0e', label=r'$a_0$=0.08')
  ax3.set_ylabel(r'$V$ (mV)')
  ax3.set_xlabel(r'$t$ (ms)')
  ax3.legend(loc='upper right')
  ax3.spines['top'].set_visible(False)
  ax3.spines['right'].set_visible(False)

  # plt.savefig('QIF_a0.png', transparent=True, dpi=500)
  plt.show()


def phase_plane():
  bp.math.enable_x64()

  def phase_plane_analysis(i, model, I_ext, res=0.005):
    fig.sca(axes[i])
    pp = bp.analysis.PhasePlane1D(
      model=model,
      target_vars={'V': [-80, -30]},
      pars_update={'Iext': I_ext},
      resolutions=res
    )
    pp.plot_vector_field()
    pp.plot_fixed_point()
    plt.title('Input = {}'.format(I_ext))
    plt.xlabel(r'$V$')

  fig, gs = bp.visualize.get_figure(1, 3, 4, 4)
  axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
  inputs = [0., 3., 10]  # 设置不同大小的电流输入
  qif = QIF(1)
  for i in range(len(inputs)):
    phase_plane_analysis(i, qif, inputs[i])
  # plt.savefig('QIF_dvdt3.png', transparent=True, dpi=500)
  plt.show()


if __name__ == '__main__':
  pass
  run_QIF()
  QIF_input_threshold()
  a0_effect()
  phase_plane()

