import brainpy as bp
import brainpy.math as bm

# bm.set_platform('cpu')


class LIF1(bp.NeuGroup):
  def __init__(self, size, V_rest=0., V_reset=-5., V_th=20., tau=10., tau_ref=1., **kwargs):
    # 初始化父类
    super(LIF1, self).__init__(size=size, **kwargs)

    # 定义参数
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.tau = tau
    self.tau_ref = tau_ref  # 不应期时长

    # 定义变量
    self.V = bm.Variable(bm.ones(self.num) * V_rest)
    self.input = bm.Variable(bm.zeros(self.num))
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)  # 为方便更新，初始化所有神经元的上次发放时间为-1e7

  @bp.odeint(method='exponential_euler')  # 使用指数欧拉方法进行数值积分
  def integral(self, V, t, Iext):
    dvdt = (-V + self.V_rest + Iext) / self.tau
    return dvdt

  def update(self, _t, _dt):
    for i in range(self.num):  # 遍历每个神经元
      spike = False
      refractory = (_t - self.t_last_spike[i] <= self.tau_ref)
      if not refractory:  # 判断神经元是否发放
        V = self.integral(self.V[i], _t, self.input[i], dt=_dt)
        spike = (V >= self.V_th)  # 判断神经元是否发放
        if spike:
          V = self.V_reset
          self.t_last_spike[i] = _t
          refractory = True
        self.V[i] = V
      self.spike[i] = spike
      self.refractory[i] = refractory
      self.input[i] = 0.  # 重置外部输入


class LIF2(bp.NeuGroup):
  def __init__(self, size, V_rest=0., V_reset=-5., V_th=20., R=1., tau=10., t_refractory=1., **kwargs):
    # 初始化父类
    super(LIF2, self).__init__(size=size, **kwargs)

    # 定义参数
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.R = R
    self.tau = tau
    self.t_refractory = t_refractory  # 不应期时长

    # 定义变量
    self.V = bm.Variable(bm.random.randn(self.num) * 5. + V_reset)
    self.input = bm.Variable(bm.zeros(self.num))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)  # 为方便更新，初始化所有神经元的上次发放时间为-1e7
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))  # 是否出于不应期
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))  # 是否发放了脉冲

    # 使用指数欧拉方法进行数值积分
    self.integral = bp.odeint(f=self.derivative, method='exponential_euler')

  # 微分方程
  def derivative(self, V, t, Iext):
    dvdt = (- (V - self.V_rest) + self.R * Iext) / self.tau
    return dvdt

  def update(self, _t, _dt):
    for i in range(self.num):  # 遍历每个神经元
      if _t - self.t_last_spike[i] <= self.t_refractory:  # 神经元处于不应期
        self.refractory[i] = True
        self.spike[i] = False
      else:
        V = self.integral(self.V[i], _t, self.input[i])  # 膜电位积分
        if V >= self.V_th:  # 膜电位达到阈值，发放脉冲
          self.V[i] = self.V_reset
          self.t_last_spike[i] = _t
          self.spike[i] = True
          self.refractory[i] = True
        else:
          self.V[i] = V
          self.spike[i] = False
          self.refractory[i] = False
      self.input[i] = 0.  # 重置外部输入


class LIF(bp.NeuGroup):
  def __init__(self, size, V_rest=0., V_reset=-5., V_th=20., tau=10.,
               tau_ref=1., method='exponential_euler', **kwargs):
    # initialization
    super(LIF).__init__(size=size, method=method, **kwargs)

    # parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.tau = tau
    self.tau_ref = tau_ref

    # variables
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))

  def derivative(self, V, t, Iext):
    dvdt = (-V + self.V_rest + Iext) / self.tau
    return dvdt

  def update(self, _t, _dt):
    refractory = (_t - self.t_last_spike) <= self.tau_ref
    V = self.integral(self.V, _t, self.input, dt=_dt)
    V = bm.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike[:] = bm.where(spike, _t, self.t_last_spike)
    self.V[:] = bm.where(spike, self.V_reset, V)
    self.refractory[:] = bm.logical_or(refractory, spike)
    self.input[:] = 0.
    self.spike[:] = spike


if __name__ == '__main__':
  group = LIF(10)

  runner = bp.StructRunner(group,  inputs=('input', 26.), monitors=['V'])
  # runner = bp.ReportRunner(group, inputs=('input', 26.), monitors=['V'], report=0.5)
  runner(duration=200.)
  bp.visualize.line_plot(runner.mon.ts, runner.mon.V, show=True)
