import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15})
plt.rcParams['font.sans-serif'] = ['Times New Roman']


class LIF(bp.NeuGroupNS):
  def __init__(self, size, V_rest=0., V_reset=-5., V_th=20., R=1., tau=10., t_ref=5.,
               method='exp_auto', **kwargs):
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
    self.E_input = bm.Variable(bm.zeros(self.num))
    self.I_input = bm.Variable(bm.zeros(self.num))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)  # 上一次脉冲发放时间
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))  # 是否处于不应期
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))  # 脉冲发放状态

    # 使用指数欧拉方法进行积分
    self.integral = bp.odeint(f=self.derivative, method=method)

  # 定义膜电位关于时间变化的微分方程
  def derivative(self, V, t, Iext):
    dvdt = (-V + self.V_rest + self.R * Iext) / self.tau
    return dvdt

  def update(self):
    _t, _dt = bp.share['t'], bp.share['dt']
    # 以数组的方式对神经元进行更新
    self.input += self.E_input + self.I_input
    refractory = (_t - self.t_last_spike) <= self.t_ref  # 判断神经元是否处于不应期
    V = self.integral(self.V, _t, self.input, dt=_dt)  # 根据时间步长更新膜电位
    V = bm.where(refractory, self.V, V)  # 若处于不应期，则返回原始膜电位self.V，否则返回更新后的膜电位V
    spike = V > self.V_th  # 将大于阈值的神经元标记为发放了脉冲
    self.spike[:] = spike  # 更新神经元脉冲发放状态
    self.t_last_spike[:] = bm.where(spike, _t, self.t_last_spike)  # 更新最后一次脉冲发放时间
    self.V[:] = bm.where(spike, self.V_reset, V)  # 将发放了脉冲的神经元膜电位置为V_reset，其余不变
    self.refractory[:] = bm.logical_or(refractory, spike)  # 更新神经元是否处于不应期

  def clear_input(self):
    self.input[:] = 0.
    self.E_input[:] = 0.
    self.I_input[:] = 0.


class EINet(bp.Network):
  def __init__(self, num_exc, num_inh, method='exp_auto', **kwargs):
    super(EINet, self).__init__(**kwargs)

    # 搭建神经元
    pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., t_ref=5.)  # 神经元模型需要的参数
    E = LIF(num_exc, **pars, method=method)
    I = LIF(num_inh, **pars, method=method)
    E.V.value = bm.random.randn(num_exc) * 4. - 60.  # 随机初始化膜电位
    I.V.value = bm.random.randn(num_inh) * 4. - 60.  # 随机初始化膜电位
    self.E = E
    self.I = I

    # 搭建神经元连接
    E_pars = dict(g_max=0.3, tau=5.)  # 兴奋性突触需要的参数
    I_pars = dict(g_max=3.2, tau=10.)  # 抑制性突触需要的参数
    self.E2E = bp.synapses.Exponential(E, E, bp.conn.FixedProb(prob=0.02), method=method,
                                       output=bp.synouts.COBA(E=0., target_var='E_input'),
                                       **E_pars)
    self.E2I = bp.synapses.Exponential(E, I, bp.conn.FixedProb(prob=0.02), method=method,
                                       output=bp.synouts.COBA(E=0., target_var='E_input'),
                                       **E_pars)
    self.I2E = bp.synapses.Exponential(I, E, bp.conn.FixedProb(prob=0.02), method=method,
                                       output=bp.synouts.COBA(E=-80., target_var='I_input'),
                                       **I_pars)
    self.I2I = bp.synapses.Exponential(I, I, bp.conn.FixedProb(prob=0.02), method=method,
                                       output=bp.synouts.COBA(E=-80., target_var='I_input'),
                                       **I_pars)


def visualize_current(ts, V, V_th, E_input, I_input, ext_input, duration):
  fig, gs = bp.visualize.get_figure(1, 1, 2.25, 6)
  ax = fig.add_subplot(gs[0, 0])
  ax.plot(ts, E_input, label='E input', color=u'#e62728')
  ax.plot(ts, I_input, label='I input', color=u'#1f77e4')
  # input中不包括外部输入，在此需加上12
  ax.plot(ts, E_input + I_input + ext_input, label='total input', color=u'#2cd02c')
  ax.axhline(0, linestyle='--', color=u'#ff7f0e')
  ax.set_ylabel('Current')
  plt.text(duration / 2, E_input.max() * 0.75, 'E input')
  plt.text(duration / 2, I_input.min() * 0.75, 'I input')
  plt.text(duration + 5, 2., 'Total input')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.set_xlim(-1, duration + 50)
  # plt.savefig('EI_net_example_current.pdf', transparent=True, dpi=500)

  fig, gs = bp.visualize.get_figure(1, 1, 2.25, 6)
  ax = fig.add_subplot(gs[0, 0])
  ax.plot(ts, V)
  ax.axhline(V_th, linestyle='--', color=u'#ff7f0e')
  ax.set_ylabel('V')
  ax.set_xlabel(r'$t$ (ms)')
  ax.set_ylabel('Potential')
  plt.text(duration + 5, V_th - 3, 'Threshold')
  plt.text(duration + 5, (V.min() + V_th) / 2, r'$V$')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.set_xlim(-1, duration + 50)
  # plt.savefig('EI_net_example_potential.pdf', transparent=True, dpi=500)

  plt.show()


# 数值模拟
bm.random.seed(12345)
duration = 200.
net = EINet(3200, 800)
runner = bp.DSRunner(net,
                     monitors=['E.spike', 'I.spike', 'E.E_input', 'E.I_input', 'E.V'],
                     inputs=[('E.input', 12.), ('I.input', 12.)])
runner(duration)

# for i in [199, 100, 21]:
for i in [199, ]:
  visualize_current(runner.mon.ts,
                    runner.mon['E.V'][:, i],
                    net.E.V_th,
                    runner.mon['E.E_input'][:, i],
                    runner.mon['E.I_input'][:, i],
                    12.,
                    duration)
