import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


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

  def update(self, _t, _dt):
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


# # 运行QIF模型
# group = QIF(1)
# runner = bp.DSRunner(group, monitors=['V'], inputs=('input', 6.))
# runner(500)  # 运行时长为500ms
# # 结果可视化
# plt.plot(runner.mon.ts, runner.mon.V)
# plt.xlabel('t (ms)')
# plt.ylabel('V (mV)')
# plt.show()


def QIF_plot(input, duration, color):
  neu = QIF(1)
  neu.V[:] = bm.array([-68.])
  runner = bp.DSRunner(neu, monitors=['V'], inputs=('input', input))
  runner(duration)
  plt.plot(runner.mon.ts, runner.mon.V, color=color, label='input={}'.format(input))


inputs = [0., 3., 4., 5.]
# colors = [u'#9467bd', u'#d62728', u'#1f77b4', u'#ff7f0e']
colors = [u'#2ca02c', u'#d62728', u'#1f77b4', u'#ff7f0e']
dur = 500

# for i in range(len(inputs)):
#   QIF_plot(inputs[i], dur, colors[i])
#
# plt.xlabel('t (ms)')
# plt.ylabel('V (mV)')
# plt.legend()
# plt.show()
