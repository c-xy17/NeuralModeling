import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


class STP(bp.dyn.TwoEndConn):
  def __init__(self, pre, post, conn, g_max=0.1, U=0.15, tau_f=1500., tau_d=200.,
               tau=8., E=0., delay_step=2, method='exp_auto', **kwargs):
    super(STP, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input', 'V')

    # 初始化参数
    self.tau_d = tau_d
    self.tau_f = tau_f
    self.tau = tau
    self.U = U
    self.g_max = g_max
    self.E = E
    self.delay_step = delay_step

    # 获取每个连接的突触前神经元pre_ids和突触后神经元post_ids
    self.pre_ids, self.post_ids = self.conn.require('pre_ids', 'post_ids')

    self.x = bm.Variable(bm.ones(self.pre.num))
    self.u = bm.Variable(bm.zeros(self.pre.num))
    self.g = bm.Variable(bm.zeros(self.pre.num))
    self.delay = bm.LengthDelay(self.g, delay_step)  # 定义一个处理g的延迟器

    # 定义积分函数
    self.integral = bp.odeint(method=method, f=self.derivative)

  @property
  def derivative(self):
    du = lambda u, t: - u / self.tau_f
    dx = lambda x, t: (1 - x) / self.tau_d
    dI = lambda I, t: -I / self.tau
    return bp.JointEq([du, dx, dI])  # 将三个微分方程联合求解

  def update(self, _t, _dt):

    # 将突触前神经元传来的信号延迟delay_step的时间步长
    delayed_g = self.delay(self.delay_step)

    # 计算突触后电流
    post_g = bm.syn2post(delayed_g, self.post_ids, self.post.num)
    # self.post.input += post_g * (self.E - self.post.V)
    self.post.input += post_g

    # 更新各个变量
    syn_sps = bm.pre2syn(self.pre.spike, self.pre_ids)  # 哪些突触前神经元产生了脉冲
    g, u, x = self.integral(self.g, self.u, self.x, _t, dt=_dt)  # 计算积分后的g, u, x
    u = bm.where(syn_sps, u + self.U * (1 - self.u), u)  # 更新后的u
    x = bm.where(syn_sps, x - u * self.x, x)  # 更新后的x
    g = bm.where(syn_sps, self.g + self.g_max * u * self.x, self.g)  # 更新后的g
    self.u.value = u
    self.x.value = x
    self.g.value = g

    # 更新延迟器
    self.delay.update(self.g)


def run_STP():
  neu1 = bp.dyn.LIF(1)
  neu2 = bp.dyn.LIF(1)
  syn1 = STP(neu1, neu2, bp.connect.All2All(), U=0.1, tau_d=10, tau_f=100.)
  net = bp.dyn.Network(pre=neu1, syn=syn1, post=neu2)

  runner = bp.dyn.DSRunner(net,
                           inputs=[('pre.input', 28.)],
                           monitors=['syn.g', 'syn.u', 'syn.x'])
  runner.run(150.)

  # 可视化
  fig, gs = bp.visualize.get_figure(2, 1, 3, 7)

  fig.add_subplot(gs[0, 0])
  plt.plot(runner.mon.ts, runner.mon['syn.u'][:, 0], label='u')
  plt.plot(runner.mon.ts, runner.mon['syn.x'][:, 0], label='x')
  plt.legend()

  fig.add_subplot(gs[1, 0])
  plt.plot(runner.mon.ts, runner.mon['syn.g'][:, 0], label='I')
  plt.legend()

  plt.xlabel('t (ms)')
  plt.show()
  

run_STP()