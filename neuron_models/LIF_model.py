import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 15})
plt.rcParams['font.sans-serif'] = ['Times New Roman']


class LIF(bp.NeuGroupNS):
  def __init__(self, size, V_rest=0., V_reset=-5., V_th=20., R=1., tau=10., t_ref=5., **kwargs):
    # 初始化父类
    super(LIF, self).__init__(size=size, **kwargs)

    # 初始化参数
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
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
    self.integral = bp.odeint(f=self.derivative, method='exponential_euler')

  # 定义膜电位关于时间变化的微分方程
  def derivative(self, V, t, Iext):
    dvdt = (-V + self.V_rest + self.R * Iext) / self.tau
    return dvdt

  def update(self):
    _t, _dt = bp.share['t'], bp.share['dt']
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


def run_LIF():
  # 运行LIF模型

  group = LIF(1)
  runner = bp.DSRunner(group, monitors=['V'], inputs=('input', 22.))
  runner(200)  # 运行时长为200ms

  # 结果可视化
  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
  ax = fig.add_subplot(gs[0, 0])
  plt.plot(runner.mon.ts, runner.mon.V)
  plt.xlabel(r'$t$ (ms)')
  plt.ylabel(r'$V$ (mV)')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.savefig('LIF_output.pdf', transparent=True, dpi=500)
  # plt.show()


def fi_curve():
  duration = 1000  # 设定仿真时长
  I_cur = np.arange(0, 600, 2)  # 定义大小为10, 20, ..., 1000的100束电流

  neu = LIF(len(I_cur), tau=5., t_ref=5.)  # 定义神经元群
  runner = bp.DSRunner(neu, monitors=['spike'], inputs=('input', I_cur), dt=0.01)  # 设置运行器，其中每一个神经元接收I_cur的一个恒定电流
  runner(duration=duration)  # 运行神经元模型
  f_list = runner.mon.spike.sum(axis=0) / (duration / 1000)  # 计算每个神经元的脉冲发放次数

  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
  ax = fig.add_subplot(gs[0, 0])
  plt.plot(I_cur, f_list)
  plt.xlabel('Input current')
  plt.ylabel('spiking frequency')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.savefig('LIF_fi_curve.pdf', transparent=True, dpi=500)
  # plt.show()


def threshold():
  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
  ax = fig.add_subplot(gs[0, 0])

  duration = 200
  neu1 = LIF(1, t_ref=5.)
  neu1.V[:] = bm.array([-5.])  # 设置V的初始值
  runner = bp.DSRunner(neu1, monitors=['V'], inputs=('input', 20))  # 给neu1一个大小为20的恒定电流
  runner(duration)
  plt.plot(runner.mon.ts, runner.mon.V, label='$I = 20$')
  plt.text(153, 21, r'$I = 20$')

  neu2 = LIF(1, t_ref=5.)
  neu2.V[:] = bm.array([-5.])  # 设置V的初始值
  runner = bp.DSRunner(neu2, monitors=['V'], inputs=('input', 21))  # 给neu2一个大小为21的恒定电流
  runner(duration)
  plt.plot(runner.mon.ts, runner.mon.V, label='$I = 21$')
  plt.text(153, -2, r'$I = 21$')

  plt.xlabel(r'$t$ (ms)')
  plt.ylabel(r'$V$ (mV)')
  plt.ylim(-6, 23)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.savefig('LIF_input_threshold.pdf', transparent=True, dpi=500)
  # plt.show()


if __name__ == '__main__':
  run_LIF()
  fi_curve()
  threshold()
