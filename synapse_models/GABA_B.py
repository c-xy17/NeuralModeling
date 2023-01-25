import brainpy as bp
import brainpy.math as bm

from run_synapse import run_syn_GABAb


class GABAb(bp.TwoEndConn):
  def __init__(self, pre, post, conn, g_max=1., E=-95., alpha=0.09, beta=0.0012,
               T_0=0.5, T_dur=0.5, k1=0.18, k2=0.034, K_d=0.1, delay_step=2,
               method='exp_auto', **kwargs):
    super(GABAb, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input', 'V')

    # 初始化参数
    self.g_max = g_max
    self.E = E
    self.alpha = alpha
    self.beta = beta
    self.T_0 = T_0
    self.T_dur = T_dur
    self.k1 = k1
    self.k2 = k2
    self.K_d = K_d
    self.delay_step = delay_step

    # 获取关于连接的信息
    self.pre2post = self.conn.require('pre2post')  # 获取从pre到post的连接信息

    # 初始化变量
    self.r = bm.Variable(bm.zeros(self.post.num))
    self.G = bm.Variable(bm.zeros(self.post.num))
    self.s = bm.Variable(bm.zeros(self.post.num))
    self.g = bm.Variable(bm.zeros(self.post.num))
    self.spike_arrival_time = bm.Variable(bm.ones(self.pre.num) * -1e7)  # 脉冲到达的时间
    self.delay = bm.LengthDelay(self.pre.spike, delay_step)  # 定义一个延迟处理器

    # 定义积分函数
    self.integral = bp.odeint(method=method, f=bp.JointEq(self.dr, self.dG))

  def dr(self, r, t, T):
    return self.alpha * T * (1 - r) - self.beta * r

  def dG(self, G, t, r):
    return self.k1 * r - self.k2 * G

  def update(self, tdi):
    # 将突触前神经元传来的信号延迟delay_step的时间步长
    delayed_pre_spike = self.delay(self.delay_step)
    self.delay.update(self.pre.spike)

    # 更新脉冲到达的时间，并以此计算T
    self.spike_arrival_time.value = bm.where(delayed_pre_spike, tdi.t, self.spike_arrival_time)
    T = ((tdi.t - self.spike_arrival_time) < self.T_dur) * self.T_0

    # 更新r，G，s和g
    self.r.value, self.G.value = self.integral(self.r, self.G, tdi.t, T, tdi.dt)
    self.s.value = bm.power(self.G, 4) / (bm.power(self.G, 4) + self.K_d)
    self.g.value = self.g_max * self.s

    # 电导模式下计算突触后电流大小
    self.post.input += self.g * (self.E - self.post.V)


if __name__ == '__main__':
  run_syn_GABAb(GABAb, run_duration=1000., Iext=0., title='GABA$_\mathrm{B}$ Synapse Model')

