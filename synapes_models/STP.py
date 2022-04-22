import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


class STP(bp.dyn.TwoEndConn):
  def __init__(self, pre, post, conn, g_max=0.1, U=0.15, tau_f=1500., tau_d=200.,
               tau=8., E=1., delay_step=2, method='exp_auto', **kwargs):
    super(STP, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input', 'V_rest')

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

    # 初始化变量
    num = len(self.pre_ids)
    self.x = bm.Variable(bm.ones(num))
    self.u = bm.Variable(bm.zeros(num))
    self.g = bm.Variable(bm.zeros(num))
    self.delay = bm.LengthDelay(self.g, delay_step)  # 定义一个处理g的延迟器

    # 定义积分函数
    self.integral = bp.odeint(method=method, f=self.derivative)

  @property
  def derivative(self):
    du = lambda u, t: - u / self.tau_f
    dx = lambda x, t: (1 - x) / self.tau_d
    dg = lambda g, t: -g / self.tau
    return bp.JointEq([du, dx, dg])  # 将三个微分方程联合求解

  def update(self, _t, _dt):
    # 将g的计算延迟delay_step的时间步长
    delayed_g = self.delay(self.delay_step)

    # 计算突触后电流
    post_g = bm.syn2post(delayed_g, self.post_ids, self.post.num)
    self.post.input += post_g * (self.E - self.post.V_rest)

    # 更新各个变量
    syn_sps = bm.pre2syn(self.pre.spike, self.pre_ids)  # 哪些突触前神经元产生了脉冲
    u, x, g = self.integral(self.u, self.x, self.g, _t)  # 计算积分后的u, x, g
    u = bm.where(syn_sps, u + self.U * (1 - self.u), u)  # 更新后的u
    x = bm.where(syn_sps, x - u * self.x, x)  # 更新后的x
    g = bm.where(syn_sps, g + self.g_max * u * self.x, g)  # 更新后的g
    self.u.value = u
    self.x.value = x
    self.g.value = g

    # 更新延迟器
    self.delay.update(self.g)


def run_STP(title=None, **kwargs):
  # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
  neu1 = bp.dyn.LIF(1)
  neu2 = bp.dyn.LIF(1)
  syn = STP(neu1, neu2, bp.connect.All2All(), **kwargs)
  net = bp.dyn.Network(pre=neu1, syn=syn, post=neu2)

  # 分段电流
  inputs, dur = bp.inputs.section_input(values=[22., 0., 22., 0.],
                                   durations=[200., 200., 25., 75.],
                                   return_length=True)
  # 运行模拟
  runner = bp.dyn.DSRunner(net,
                           inputs=[('pre.input', inputs, 'iter')],
                           monitors=['syn.u', 'syn.x', 'syn.g'])
  runner.run(dur)

  # 可视化
  fig, gs = plt.subplots(2, 1, figsize=(6, 4.5))

  plt.sca(gs[0])
  plt.plot(runner.mon.ts, runner.mon['syn.x'][:, 0], label='x')
  plt.plot(runner.mon.ts, runner.mon['syn.u'][:, 0], label='u')
  plt.legend(loc='center right')
  if title: plt.title(title)

  plt.sca(gs[1])
  plt.plot(runner.mon.ts, runner.mon['syn.g'][:, 0], label='g', color=u'#d62728')
  plt.legend(loc='center right')

  plt.xlabel('t (ms)')
  plt.tight_layout()
  plt.show()
  

# 短时程易化
run_STP(title='STF', U=0.1, tau_d=15., tau_f=200.)
# 短时程抑制
run_STP(title='STD', U=0.4, tau_d=200., tau_f=15.)
