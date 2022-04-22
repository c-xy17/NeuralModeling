import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


class STDP(bp.dyn.TwoEndConn):
  def __init__(self, pre, post, conn, tau_s=10., tau_t=10., tau=8., delta_As=0.5,
               delta_At=0.5, E=1., delay_step=0, method='exp_auto', **kwargs):
    super(STDP, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
    self.check_pre_attrs('spike')
    self.check_post_attrs('spike', 'input', 'V_rest')

    # 初始化参数
    self.tau_s = tau_s
    self.tau_t = tau_t
    self.tau = tau
    self.delta_As = delta_As
    self.delta_At = delta_At
    self.E = E
    self.delay_step = delay_step

    # 获取每个连接的突触前神经元pre_ids和突触后神经元post_ids
    self.pre_ids, self.post_ids = self.conn.require('pre_ids', 'post_ids')

    # 初始化变量
    num = len(self.pre_ids)
    self.As = bm.Variable(bm.zeros(num))
    self.At = bm.Variable(bm.zeros(num))
    self.w = bm.Variable(bm.ones(num))
    self.g = bm.Variable(bm.zeros(num))
    self.delay = bm.LengthDelay(self.pre.spike, delay_step)  # 定义一个延迟处理器

    # 定义积分函数
    self.integral = bp.odeint(method=method, f=self.derivative)

  @property
  def derivative(self):
    dAs = lambda As, t: - As / self.tau_s
    dAt = lambda At, t: - At / self.tau_t
    dg = lambda g, t: -g / self.tau
    return bp.JointEq([dAs, dAt, dg])  # 将三个微分方程联合求解

  def update(self, _t, _dt):
    # 将g的计算延迟delay_step的时间步长
    delayed_g = self.delay(self.delay_step)

    # 计算突触后电流
    post_g = bm.syn2post(delayed_g, self.post_ids, self.post.num)
    self.post.input += post_g * (self.E - self.post.V_rest)

    # 更新各个变量
    pre_spikes = bm.pre2syn(self.pre.spike, self.pre_ids)  # 哪些突触前神经元产生了脉冲
    post_spikes = bm.pre2syn(self.post.spike, self.post_ids)  # 哪些突触后神经元产生了脉冲

    # 计算积分后的As, At, g
    self.As.value, self.At.value, self.g.value = self.integral(self.As, self.At, self.g, _t)

    # if (pre spikes)
    As = bm.where(pre_spikes, self.As + self.delta_As, self.As)
    self.w.value = bm.where(pre_spikes, self.w - self.At, self.w)
    # if (post spikes)
    At = bm.where(post_spikes, self.At + self.delta_At, self.At)
    self.w.value = bm.where(post_spikes, self.w + self.As, self.w)
    self.As.value = As
    self.At.value = At

    # 更新完w后再更新g
    self.g.value = bm.where(pre_spikes, self.g + self.w, self.g)

    # 更新延迟器
    self.delay.update(self.g)


def run_STDP(I_pre, I_post, dur, **kwargs):
  # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
  pre = bp.dyn.LIF(1)
  post = bp.dyn.LIF(1)
  syn = STDP(pre, post, bp.connect.All2All(), **kwargs)
  net = bp.dyn.Network(pre=pre, syn=syn, post=post)

  # 运行模拟
  runner = bp.dyn.DSRunner(net,
                           inputs=[('pre.input', I_pre, 'iter'), ('post.input', I_post, 'iter')],
                           monitors=['pre.spike', 'post.spike', 'syn.g', 'syn.w', 'syn.As', 'syn.At'])
  runner(dur)

  # 可视化
  fig, gs = plt.subplots(5, 1, gridspec_kw={'height_ratios': [2, 1, 1, 2, 2]}, figsize=(6, 8))

  plt.sca(gs[0])
  plt.plot(runner.mon.ts, runner.mon['syn.g'][:, 0], label='$g$', color=u'#d62728')

  plt.sca(gs[1])
  plt.plot(runner.mon.ts, runner.mon['pre.spike'][:, 0], label='pre spike', color='springgreen')
  plt.legend(loc='center right')

  plt.sca(gs[2])
  plt.plot(runner.mon.ts, runner.mon['post.spike'][:, 0], label='post spike', color='seagreen')

  plt.sca(gs[3])
  plt.plot(runner.mon.ts, runner.mon['syn.w'][:, 0], label='$w$')

  plt.sca(gs[4])
  plt.plot(runner.mon.ts, runner.mon['syn.As'][:, 0], label='$A_s$', color='coral')
  plt.plot(runner.mon.ts, runner.mon['syn.At'][:, 0], label='$A_t$', color='gold')

  for i in range(4):
    gs[i].set_xticks([])
  for i in range(1, 3):
    gs[i].set_yticks([])
  for i in range(5):
    gs[i].legend(loc='upper right')

  plt.xlabel('t (ms)')
  plt.tight_layout()
  plt.subplots_adjust(hspace=0.)
  plt.show()


# 设置输入给pre和post的电流
duration = 300.
I_pre, _ = bp.inputs.constant_input([(0, 5), (30, 15),
                                     (0, 15), (30, 15),
                                     (0, 15), (30, 15),
                                     (0, 98), (30, 15),  # switch order: t_interval=98ms
                                     (0, 15), (30, 15),
                                     (0, 15), (30, 15),
                                     (0, duration - 155 - 98)])
I_post, _ = bp.inputs.constant_input([(0, 10), (30, 15),
                                      (0, 15), (30, 15),
                                      (0, 15), (30, 15),
                                      (0, 90), (30, 15),  # switch order: t_interval=90ms
                                      (0, 15), (30, 15),
                                      (0, 15), (30, 15),
                                      (0, duration - 160 - 90)])

run_STDP(I_pre, I_post, duration)