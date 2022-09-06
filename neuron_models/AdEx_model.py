import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np


class AdEx(bp.dyn.NeuGroup):
  def __init__(self, size, V_rest=-65., V_reset=-68., V_th=20., V_T=-60., delta_T=1., a=1.,
               b=2.5, R=1., tau=10., tau_w=30., name=None):
    # 初始化父类
    super(AdEx, self).__init__(size=size, name=name)

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
    self.V = bm.Variable(bm.random.randn(self.num) + V_reset)
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

  def update(self, tdi):
    # 以数组的方式对神经元进行更新
    V, w = self.integral(self.V, self.w, tdi.t, self.input, tdi.dt)  # 更新膜电位V和权重值w
    spike = V > self.V_th  # 将大于阈值的神经元标记为发放了脉冲
    self.spike.value = spike  # 更新神经元脉冲发放状态
    self.V.value = bm.where(spike, self.V_reset, V)  # 将发放了脉冲的神经元膜电位置为V_reset，其余不变
    self.w.value = bm.where(spike, w + self.b, w)  # 发放了脉冲的神经元 w = w + b
    self.input[:] = 0.  # 重置外界输入


# 运行AdEx模型
neu = AdEx(1)
runner = bp.dyn.DSRunner(neu, monitors=['V', 'w', 'spike'], inputs=('input', 9.), dt=0.01)
runner(500)

# 可视化V和w的变化
runner.mon.V = np.where(runner.mon.spike, 20., runner.mon.V)
plt.plot(runner.mon.ts, runner.mon.V, label='V')
plt.plot(runner.mon.ts, runner.mon.w, label='w')
plt.xlabel('t (ms)')
plt.ylabel('V (mV)')

plt.legend(loc='upper right')
plt.show()



group = AdEx(
  size=6,
  a=bm.asarray([0., 0., 0.5, -0.5, 1., -1.]),
  b=bm.asarray([60., 5., 7., 7., 10., 5.]),
  tau=bm.asarray([20., 20., 5., 5., 10., 5.]),
  tau_w=bm.asarray([30., 100., 100., 100., 100., 100.]),
  V_reset=bm.asarray([-55., -55., -51., -47., -60., -60.]),
  R=.5, delta_T=2., V_rest=-70, V_th=-30, V_T=-50
)
group.V.value = group.V_reset

par_I = bm.asarray([65., 65., 65., 65., 55., 25.])
runner = bp.dyn.DSRunner(group, monitors=['V', 'w', 'spike'], inputs=('input', par_I))
runner.run(500.)

runner.mon.V = np.where(runner.mon.spike, 20., runner.mon.V)
names = ['Tonic', 'Adapting', 'Init Bursting', 'Bursting', 'Transient', 'Delayed']
fig, gs = bp.visualize.get_figure(2, 3, 3, 4)
for i_col in range(2):
  for i_row in range(3):
    i = i_col * 3 + i_row
    fig.add_subplot(gs[i_col, i_row])
    plt.plot(runner.mon.ts, runner.mon.V[:, i], label='V')
    plt.plot(runner.mon.ts, runner.mon.w[:, i], label='w')
    plt.title(names[i])
    plt.xlabel('Time [ms]')
    plt.legend()
plt.show()




