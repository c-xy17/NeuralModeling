import brainpy as bp
import brainpy.math as bm

from run_synapse import run_delta_syn


class VoltageJump(bp.TwoEndConn):
  def __init__(self, pre, post, conn, g_max=1., delay_step=2, E=0., **kwargs):
    super().__init__(pre=pre, post=post, conn=conn, **kwargs)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input', 'V')

    # 初始化参数
    self.g_max = g_max
    self.delay_step = delay_step
    self.E = E

    # 获取关于连接的信息
    self.pre2post = self.conn.require('pre2post')  # 获取从pre到post的连接信息

    # 初始化变量
    self.g = bm.Variable(bm.zeros(self.post.num))
    self.delay = bm.LengthDelay(self.pre.spike, delay_step)  # 定义一个延迟处理器

  def update(self, tdi):
    # 将突触前神经元传来的信号延迟delay_step的时间步长
    delayed_pre_spike = self.delay(self.delay_step)
    self.delay.update(self.pre.spike)

    # 根据连接模式计算各个突触后神经元收到的信号强度
    post_sp = bm.pre2post_event_sum(delayed_pre_spike, self.pre2post, self.post.num, self.g_max)
    self.g.value = post_sp

    # 计算突触后电流
    self.post.V += self.g  # E - V_rest


if __name__ == '__main__':
  run_delta_syn(VoltageJump, title='Delta Synapse Model', g_max=2.)
