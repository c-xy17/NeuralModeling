import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt


class LIF(bp.dyn.NeuGroup):
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
    self.E_input_rc = bm.Variable(bm.zeros(self.num))
    self.I_input_rc = bm.Variable(bm.zeros(self.num))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)  # 上一次脉冲发放时间
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))  # 是否处于不应期
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))  # 脉冲发放状态

    # 使用指数欧拉方法进行积分
    self.integral = bp.odeint(f=self.derivative, method=method)

  # 定义膜电位关于时间变化的微分方程
  def derivative(self, V, t, Iext):
    dvdt = (-V + self.V_rest + self.R * Iext) / self.tau
    return dvdt

  def update(self, _t, _dt):
    # 以数组的方式对神经元进行更新
    self.input += self.E_input + self.I_input
    self.E_input_rc.value = self.E_input
    self.I_input_rc.value = self.I_input
    self.E_input[:] = 0
    self.I_input[:] = 0
    refractory = (_t - self.t_last_spike) <= self.t_ref  # 判断神经元是否处于不应期
    V = self.integral(self.V, _t, self.input, dt=_dt)  # 根据时间步长更新膜电位
    V = bm.where(refractory, self.V, V)  # 若处于不应期，则返回原始膜电位self.V，否则返回更新后的膜电位V
    spike = V > self.V_th  # 将大于阈值的神经元标记为发放了脉冲
    self.spike[:] = spike  # 更新神经元脉冲发放状态
    self.t_last_spike[:] = bm.where(spike, _t, self.t_last_spike)  # 更新最后一次脉冲发放时间
    self.V[:] = bm.where(spike, self.V_reset, V)  # 将发放了脉冲的神经元膜电位置为V_reset，其余不变
    self.refractory[:] = bm.logical_or(refractory, spike)  # 更新神经元是否处于不应期
    self.input[:] = 0.  # 重置外界输入


class ExpCOBA(bp.dyn.TwoEndConn):
  def __init__(self, pre, post, conn, g_max=0.02, tau=12., delay_step=2, E=0.,
               method='exp_auto', **kwargs):
    super(ExpCOBA, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input', 'V')

    # 初始化参数
    self.tau = tau
    self.g_max = g_max
    self.delay_step = delay_step
    self.E = E

    # 获取关于连接的信息
    self.pre2post = self.conn.require('pre2post')  # 获取从pre到post的连接信息

    # 初始化变量
    self.g = bm.Variable(bm.zeros(self.post.num))
    self.delay = bm.LengthDelay(self.pre.spike, delay_step)  # 定义一个延迟处理器

    # 定义积分函数
    self.integral = bp.odeint(self.derivative, method=method)

  def derivative(self, g, t):
    dgdt = -g / self.tau
    return dgdt

  def update(self, _t, _dt):
    # 将突触前神经元传来的信号延迟delay_step的时间步长
    delayed_pre_spike = self.delay(self.delay_step)
    self.delay.update(self.pre.spike)

    # 根据连接模式计算各个突触后神经元收到的信号强度
    post_sp = bm.pre2post_event_sum(delayed_pre_spike, self.pre2post, self.post.num, self.g_max)
    # 突触的电导g的更新包括常规积分和突触前脉冲带来的跃变
    self.g.value = self.integral(self.g, _t, dt=_dt) + post_sp

    if self.E < -60:
      self.post.I_input += self.g * (self.E - self.post.V)
    else:
      self.post.E_input += self.g * (self.E - self.post.V)


class EINet(bp.dyn.Network):
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
    E_pars = dict(E=0., g_max=0.3, tau=5.)  # 兴奋性突触需要的参数
    I_pars = dict(E=-80., g_max=3.2, tau=10.)  # 抑制性突触需要的参数
    self.E2E = ExpCOBA(E, E, bp.conn.FixedProb(prob=0.02), **E_pars, method=method)
    self.E2I = ExpCOBA(E, I, bp.conn.FixedProb(prob=0.02), **E_pars, method=method)
    self.I2E = ExpCOBA(I, E, bp.conn.FixedProb(prob=0.02), **I_pars, method=method)
    self.I2I = ExpCOBA(I, I, bp.conn.FixedProb(prob=0.02), **I_pars, method=method)


# 数值模拟
net = EINet(3200, 800)
runner = bp.DSRunner(net,
                     monitors=['E.spike', 'I.spike', 'E.E_input_rc', 'E.I_input_rc', 'E.V'],
                     inputs=[('E.input', 12.), ('I.input', 12.)])
runner(200.)

# 可视化
fig, gs = plt.subplots(2, 1, figsize=(6, 4), sharex='all')

i = 299  # 随机指定一个神经元序号
print(runner.mon['E.spike'][:, i].sum())
gs[0].plot(runner.mon.ts, runner.mon['E.E_input_rc'][:, i], label='E input', color=u'#e62728')
gs[0].plot(runner.mon.ts, runner.mon['E.I_input_rc'][:, i], label='I input', color=u'#1f77e4')
gs[0].plot(runner.mon.ts, runner.mon['E.E_input_rc'][:, i] + runner.mon['E.I_input_rc'][:, i] + 12.,
           label='total input', color=u'#2cd02c')  # input中不包括外部输入，在此需加上12
gs[0].axhline(0, linestyle='--', color=u'#ff7f0e')
gs[0].set_ylabel('input current')
gs[0].legend()

gs[1].plot(runner.mon.ts, runner.mon['E.V'][:, i])
gs[1].axhline(net.E.V_th, linestyle='--', color=u'#ff7f0e')
gs[1].set_ylabel('V')

plt.xlabel('t (ms)')
plt.show()


# 可视化
fig, gs = plt.subplots(2, 1, figsize=(6, 4), sharex='all')

i = 200  # 随机指定一个神经元序号
print(runner.mon['E.spike'][:, i].sum())
gs[0].plot(runner.mon.ts, runner.mon['E.E_input_rc'][:, i], label='E input', color=u'#e62728')
gs[0].plot(runner.mon.ts, runner.mon['E.I_input_rc'][:, i], label='I input', color=u'#1f77e4')
gs[0].plot(runner.mon.ts, runner.mon['E.E_input_rc'][:, i] + runner.mon['E.I_input_rc'][:, i] + 12.,
           label='total input', color=u'#2cd02c')  # input中不包括外部输入，在此需加上12
gs[0].axhline(0, linestyle='--', color=u'#ff7f0e')
gs[0].set_ylabel('input current')
gs[0].legend()

gs[1].plot(runner.mon.ts, runner.mon['E.V'][:, i])
gs[1].axhline(net.E.V_th, linestyle='--', color=u'#ff7f0e')
gs[1].set_ylabel('V')

plt.xlabel('t (ms)')
plt.show()


# 可视化
fig, gs = plt.subplots(2, 1, figsize=(6, 4), sharex='all')

i = 21  # 随机指定一个神经元序号
print(runner.mon['E.spike'][:, i].sum())
gs[0].plot(runner.mon.ts, runner.mon['E.E_input_rc'][:, i], label='E input', color=u'#e62728')
gs[0].plot(runner.mon.ts, runner.mon['E.I_input_rc'][:, i], label='I input', color=u'#1f77e4')
gs[0].plot(runner.mon.ts, runner.mon['E.E_input_rc'][:, i] + runner.mon['E.I_input_rc'][:, i] + 12.,
           label='total input', color=u'#2cc02c')  # input中不包括外部输入，在此需加上12
gs[0].axhline(0, linestyle='--', color=u'#ff7f0e')
gs[0].set_ylabel('input current')
gs[0].legend()

gs[1].plot(runner.mon.ts, runner.mon['E.V'][:, i])
gs[1].axhline(net.E.V_th, linestyle='--', color=u'#ff7f0e')
gs[1].set_ylabel('V')

plt.xlabel('t (ms)')
plt.show()