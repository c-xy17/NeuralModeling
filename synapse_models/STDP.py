import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15})
plt.rcParams['font.sans-serif'] = ['Times New Roman']

class STDP(bp.dyn.TwoEndConn):
  def __init__(self, pre, post, conn, tau_s=16.8, tau_t=33.7, tau=8., A1=0.96,
               A2=0.53, E=1., delay_step=0, method='exp_auto', **kwargs):
    super(STDP, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
    self.check_pre_attrs('spike')
    self.check_post_attrs('spike', 'input', 'V_rest')

    # 初始化参数
    self.tau_s = tau_s
    self.tau_t = tau_t
    self.tau = tau
    self.A1 = A1
    self.A2 = A2
    self.E = E
    self.delay_step = delay_step

    # 获取每个连接的突触前神经元pre_ids和突触后神经元post_ids
    self.pre_ids, self.post_ids = self.conn.require('pre_ids', 'post_ids')

    # 初始化变量
    num = len(self.pre_ids)
    self.Apre = bm.Variable(bm.zeros(num))
    self.Apost = bm.Variable(bm.zeros(num))
    self.w = bm.Variable(bm.ones(num))
    self.g = bm.Variable(bm.zeros(num))
    self.delay = bm.LengthDelay(self.g, delay_step)  # 定义一个延迟处理器

    # 定义积分函数
    self.integral = bp.odeint(method=method, f=self.derivative)

  @property
  def derivative(self):
    dApre = lambda Apre, t: - Apre / self.tau_s
    dApost = lambda Apost, t: - Apost / self.tau_t
    dg = lambda g, t: -g / self.tau
    return bp.JointEq([dApre, dApost, dg])  # 将三个微分方程联合求解

  def update(self, tdi):
    # 将g的计算延迟delay_step的时间步长
    delayed_g = self.delay(self.delay_step)

    # 计算突触后电流
    post_g = bm.syn2post(delayed_g, self.post_ids, self.post.num)
    self.post.input += post_g * (self.E - self.post.V_rest)

    # 更新各个变量
    pre_spikes = bm.pre2syn(self.pre.spike, self.pre_ids)  # 哪些突触前神经元产生了脉冲
    post_spikes = bm.pre2syn(self.post.spike, self.post_ids)  # 哪些突触后神经元产生了脉冲

    # 计算积分后的Apre, Apost, g
    self.Apre.value, self.Apost.value, self.g.value = self.integral(self.Apre, self.Apost, self.g, tdi.t, tdi.dt)

    # if (pre spikes)
    Apre = bm.where(pre_spikes, self.Apre + self.A1, self.Apre)
    self.w.value = bm.where(pre_spikes, self.w - self.Apost, self.w)
    # if (post spikes)
    Apost = bm.where(post_spikes, self.Apost + self.A2, self.Apost)
    self.w.value = bm.where(post_spikes, self.w + self.Apre, self.w)
    self.Apre.value = Apre
    self.Apost.value = Apost

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
  runner = bp.dyn.DSRunner(
    net,
    inputs=[('pre.input', I_pre, 'iter'), ('post.input', I_post, 'iter')],
    monitors=['pre.spike', 'post.spike', 'syn.g', 'syn.w', 'syn.Apre', 'syn.Apost']
  )
  runner(dur)

  # 可视化
  fig, gs = bp.visualize.get_figure(8, 1, 0.8, 10)

  ax = fig.add_subplot(gs[0:2, 0])
  plt.plot(runner.mon.ts, runner.mon['syn.g'][:, 0], label='$g$', color=u'#d62728')
  plt.xticks([])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.legend(loc='center right')

  ax = fig.add_subplot(gs[2, 0])
  plt.plot(runner.mon.ts, runner.mon['pre.spike'][:, 0], label='pre.spike', color='springgreen')
  plt.xticks([])
  plt.yticks([])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.legend(loc='center right')

  ax = fig.add_subplot(gs[3, 0])
  plt.plot(runner.mon.ts, runner.mon['post.spike'][:, 0], label='post.spike', color='seagreen')
  plt.xticks([])
  plt.yticks([])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.legend(loc='center right')

  ax = fig.add_subplot(gs[4:6, 0])
  plt.plot(runner.mon.ts, runner.mon['syn.w'][:, 0], label='$w$')
  plt.xticks([])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.legend(loc='center right')

  ax = fig.add_subplot(gs[6:8, 0])
  plt.plot(runner.mon.ts, runner.mon['syn.Apre'][:, 0], label='$A_{\mathrm{pre}}$', color='coral')
  plt.plot(runner.mon.ts, runner.mon['syn.Apost'][:, 0], label='$A_{\mathrm{post}}$', color='gold', linestyle='--')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.legend(loc='center right')

  plt.xlabel(r'$t$ (ms)')
  plt.savefig('STDP_output.pdf',
              transparent=True, dpi=500)
  plt.show()


if __name__ == '__main__':
  # 设置输入给pre和post的电流
  duration = 300.
  I_pre = bp.inputs.section_input([0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0],
                                  [5, 15, 15, 15, 15, 15, 98, 15, 15, 15, 15, 15, duration - 255])
  I_post = bp.inputs.section_input([0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0],
                                   [10, 15, 15, 15, 15, 15, 90, 15, 15, 15, 15, 15, duration - 250])

  run_STDP(I_pre, I_post, duration)
