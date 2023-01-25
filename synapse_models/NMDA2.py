import brainpy as bp
import brainpy.math as bm

from run_synapse import run_syn_NMDA


class NMDA(bp.TwoEndConn):
  def __init__(self, pre, post, conn, g_max=0.02, E=0., c_Mg=1.2, alpha1=2.,
               beta1=0.01, alpha2=1., beta2=0.5, T_0=1., T_dur=0.5, delay_step=2,
               method='exp_auto', **kwargs):
    super(NMDA, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input', 'V')

    # 初始化参数
    self.g_max = g_max
    self.E = E
    self.c_Mg = c_Mg
    self.alpha1 = alpha1
    self.beta1 = beta1
    self.alpha2 = alpha2
    self.beta2 = beta2
    self.T_0 = T_0
    self.T_dur = T_dur
    self.delay_step = delay_step

    # 获取关于连接的信息
    self.pre2post = self.conn.require('pre2post')  # 获取从pre到post的连接信息

    # 初始化变量
    self.x = bm.Variable(bm.zeros(self.post.num))
    self.s = bm.Variable(bm.zeros(self.post.num))
    self.g = bm.Variable(bm.zeros(self.post.num))
    self.b = bm.Variable(bm.zeros(self.post.num))
    self.spike_arrival_time = bm.Variable(bm.ones(self.pre.num) * -1e7)  # 脉冲到达的时间
    self.delay = bm.LengthDelay(self.pre.spike, delay_step)  # 定义一个延迟处理器

    # 定义积分函数
    self.integral = bp.odeint(method=method, f=bp.JointEq(self.ds, self.dx))

  def ds(self, s, t, x):
    return self.alpha1 * x * (1 - s) - self.beta1 * s

  def dx(self, x, t, s, T):
    return self.alpha2 * T * (1 - x) - self.beta2 * x

  def update(self, tdi):
    # 将突触前神经元传来的信号延迟delay_step的时间步长
    delayed_pre_spike = self.delay(self.delay_step)
    self.delay.update(self.pre.spike)

    # 更新脉冲到达的时间，并以此计算T
    self.spike_arrival_time.value = bm.where(delayed_pre_spike, tdi.t, self.spike_arrival_time)
    T = ((tdi.t - self.spike_arrival_time) < self.T_dur) * self.T_0

    # 更新x，s和g
    self.s.value, self.x.value = self.integral(self.s, self.x, tdi.t, T, tdi.dt)
    self.g.value = self.g_max * self.s

    # 更新b
    self.b.value = 1 / (1 + bm.exp(-0.062 * self.post.V) * self.c_Mg / 3.57)

    # 电导模式下计算突触后电流大小
    self.post.input += self.g * self.b * (self.E - self.post.V)


class NMDA_with_DE(bp.dyn.TwoEndConn):
  def __init__(self, pre, post, conn, g_max=0.02, E=0., c_Mg=1.2,
               tau_decay=100., tau_rise=2., delay_step=2,
               method='exp_auto', **kwargs):
    super(NMDA_with_DE, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input', 'V')

    # 初始化参数
    self.E = E
    self.c_Mg = c_Mg
    self.tau_decay = tau_decay
    self.tau_rise = tau_rise
    self.g_max = g_max
    self.delay_step = delay_step

    # 获取关于连接的信息
    self.pre2post = self.conn.require('pre2post')  # 获取从pre到post的连接信息

    # 初始化变量
    self.x = bm.Variable(bm.zeros(self.post.num))
    self.s = bm.Variable(bm.zeros(self.post.num))
    self.g = bm.Variable(bm.zeros(self.post.num))
    self.b = bm.Variable(bm.zeros(self.post.num))
    self.spike_arrival_time = bm.Variable(bm.ones(self.pre.num) * -1e7)  # 脉冲到达的时间
    self.delay = bm.LengthDelay(self.pre.spike, delay_step)  # 定义一个延迟处理器

    # 定义积分函数
    self.int_x = bp.odeint(method=method, f=lambda x, t: -x / self.tau_rise)
    self.int_s = bp.odeint(method=method, f=lambda s, t, x: -s / self.tau_decay + x)

  def update(self, tdi):
    # 将突触前神经元传来的信号延迟delay_step的时间步长
    delayed_pre_spike = self.delay(self.delay_step)
    self.delay.update(self.pre.spike)

    # 根据连接模式计算各个突触后神经元收到的信号强度
    post_sp = bm.pre2post_event_sum(delayed_pre_spike, self.pre2post, self.post.num, 1.)
    # 更新x，s和g
    self.x.value = self.int_x(self.x, tdi.t, tdi.dt) + post_sp
    self.s.value = self.int_s(self.s, tdi.t, self.x, tdi.dt)
    self.g.value = self.g_max * self.s

    # 更新b
    self.b.value = 1 / (1 + bm.exp(-0.062 * self.post.V) * self.c_Mg / 3.57)

    # 电导模式下计算突触后电流大小
    self.post.input += self.g * self.b * (self.E - self.post.V)


if __name__ == '__main__':
  run_syn_NMDA(NMDA,
               title='NMDA Synapse (Kinetic)',
               sp_times=[25, 50, 75, 100, 160],)

  run_syn_NMDA(NMDA_with_DE,
               title='NMDA Synapse (Phenomenological)',
               sp_times=[25, 50, 75, 100, 160],)
