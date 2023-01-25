import brainpy as bp
import brainpy.math as bm

from run_synapse import run_syn


# first-order kinetics
class AMPA(bp.TwoEndConn):
  def __init__(self, pre, post, conn, g_max=0.02, E=0., alpha=0.98, beta=0.18,
               delay_step=2, method='exp_auto', **kwargs):
    super(AMPA, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input', 'V')

    # 初始化参数
    self.g_max = g_max
    self.E = E
    self.alpha = alpha
    self.beta = beta
    self.delay_step = delay_step

    # 获取关于连接的信息
    self.pre2post = self.conn.require('pre2post')  # 获取从pre到post的连接信息

    # 初始化变量
    self.s = bm.Variable(bm.zeros(self.post.num))
    self.g = bm.Variable(bm.zeros(self.post.num))
    self.delay = bm.LengthDelay(self.pre.spike, delay_step)  # 定义一个延迟处理器

    # 定义积分函数
    self.integral = bp.odeint(method=method, f=lambda s, t: - self.beta * s)

  def update(self, tdi):
    # 将突触前神经元传来的信号延迟delay_step的时间步长
    delayed_pre_spike = self.delay(self.delay_step)
    self.delay.update(self.pre.spike)

    # 根据连接模式计算各个突触后神经元收到的信号强度
    post_sp = bm.pre2post_event_sum(delayed_pre_spike, self.pre2post, self.post.num, 1.)

    # 更新s和g
    self.s.value = self.integral(self.s, tdi.t, tdi.dt) + self.alpha * post_sp
    self.g.value = self.g_max * self.s

    # 电导模式下计算突触后电流大小
    self.post.input += self.g * (self.E - self.post.V)


if __name__ == '__main__':
  run_syn(AMPA, title='AMPA Synapse Model')
