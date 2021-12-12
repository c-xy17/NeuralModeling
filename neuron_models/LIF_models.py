import brainpy as bp
import brainpy.math as bm

class LIF(bp.NeuGroup):
  def __init__(self, size, V_rest=0., V_reset=-5., V_th=20., R=1., tau=10., tau_ref=5., **kwargs):
    # 初始化父类
    super(LIF, self).__init__(size=size, **kwargs)

    # 初始化参数
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.R = R
    self.tau = tau
    self.tau_ref = tau_ref  # 不应期时长

    # 初始化变量
    self.V = bm.Variable(bm.random.randn(self.num) * 5. + V_reset)
    self.input = bm.Variable(bm.zeros(self.num))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)  # 上一次脉冲发放时间
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))  # 是否处于不应期
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))  # 脉冲发放状态

    # 使用指数欧拉方法进行积分
    self.integral = bp.odeint(f=self.derivative, method='exponential_euler')

  # 定义膜电位关于时间变化的微分方程
  def derivative(self, V, t, R, Iext):
    dvdt = (-V + self.V_rest + R * Iext) / self.tau
    return dvdt

  def update(self, _t, _dt):
    # 以数组的方式对神经元进行更新
    refractory = (_t - self.t_last_spike) <= self.tau_ref  # 判断神经元是否处于不应期
    V = self.integral(self.V, _t, self.R, self.input, dt=_dt)  # 根据时间步长更新膜电位
    V = bm.where(refractory, self.V, V)  # 若处于不应期，则返回原始膜电位self.V，否则返回更新后的膜电位V
    spike = self.V_th <= V  # 将大于阈值的神经元标记为发放了脉冲
    self.spike[:] = spike  # 更新神经元脉冲发放状态
    self.t_last_spike[:] = bm.where(spike, _t, self.t_last_spike)  # 更新最后一次脉冲发放时间
    self.V[:] = bm.where(spike, self.V_reset, V)  # 将发放了脉冲的神经元膜电位置为V_reset，其余不变
    self.refractory[:] = bm.logical_or(refractory, spike)  # 更新神经元是否处于不应期
    self.input[:] = 0.  # 重置外界输入


if __name__ == '__main__':
    group = LIF(10)
    runner = bp.ReportRunner(group, monitors=['V'], inputs=('input', 22.), report=0.2)
    runner.run(200)
    bp.visualize.line_plot(runner.mon.ts, runner.mon.V, show=True)

