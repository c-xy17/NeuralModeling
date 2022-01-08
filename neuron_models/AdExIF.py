import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


class AdExIF(bp.NeuGroup):
  def __init__(self, size, V_rest=-65., V_reset=-68., V_th=-30., V_T=-60., delta_T=1., a=1.,
               b=2.5, tau=10., tau_w=30., tau_ref=2., R=1., **kwargs):
    # 初始化父类
    super(AdExIF, self).__init__(size=size, **kwargs)

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
    self.tau_ref = tau_ref
    self.R = R

    # 初始化变量
    self.V = bm.Variable(bm.random.randn(self.num) + V_reset)
    self.w = bm.Variable(bm.zeros(self.num))
    self.input = bm.Variable(bm.zeros(self.num))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)  # 上一次脉冲发放时间
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))  # 是否处于不应期
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
    refractory = (_t - self.t_last_spike) <= self.tau_ref  # 判断神经元是否处于不应期
    V, w = self.integral(self.V, self.w, _t, self.input, dt=_dt)  # 更新膜电位V和权重值w
    V = bm.where(refractory, self.V, V)  # 若处于不应期，则返回原始膜电位self.V，否则返回更新后的膜电位V
    spike = self.V_th <= V  # 将大于阈值的神经元标记为发放了脉冲
    self.spike.value = spike  # 更新神经元脉冲发放状态
    self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)  # 更新最后一次脉冲发放时间
    self.V.value = bm.where(spike, self.V_reset, V)  # 将发放了脉冲的神经元膜电位置为V_reset，其余不变
    self.w.value = bm.where(spike, w + self.b, w)  # 发放了脉冲的神经元 w = w + b
    self.refractory.value = bm.logical_or(refractory, spike)  # 更新神经元是否处于不应期
    self.input[:] = 0.  # 重置外界输入


# 运行AdExIF模型
group = AdExIF(10)
runner = bp.StructRunner(group, monitors=['V', 'w'], inputs=('input', 11.))
runner(200)  # 运行时长为200ms

# 使用matplotlib.pyplot绘图
# fig, ax1 = plt.subplots()
# ax1.plot(runner.mon.ts, runner.mon.V[:, 0], label='V')
# ax1.set_ylabel('V')
#
# ax2 = ax1.twinx()
# ax2.plot(runner.mon.ts, runner.mon.w[:, 0], color='orange', label='w')
# ax2.set_ylabel('w')

# fig.legend(loc="upper center")
# plt.show()

fig, ax1 = plt.subplots()
bp.visualize.line_plot(runner.mon.ts, runner.mon.w, ax=ax1, ylabel='w', legend='w', show=False)

ax2 = ax1.twinx()
bp.visualize.line_plot(runner.mon.ts, runner.mon.V, ax=ax2, ylabel='V', legend='V', show=True)
