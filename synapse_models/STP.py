import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 15})
plt.rcParams['font.sans-serif'] = ['Times New Roman']

class STP(bp.synapses.TwoEndConn):
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

  def update(self):
    # 将g的计算延迟delay_step的时间步长
    delayed_g = self.delay(self.delay_step)

    # 计算突触后电流
    post_g = bm.syn2post(delayed_g, self.post_ids, self.post.num)
    self.post.input += post_g * (self.E - self.post.V_rest)

    # 更新各个变量
    syn_sps = bm.pre2syn(self.pre.spike, self.pre_ids)  # 哪些突触前神经元产生了脉冲
    u, x, g = self.integral(self.u, self.x, self.g, bp.share['t'], bp.share['dt'])  # 计算积分后的u, x, g
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
  neu1 = bp.neurons.LIF(1)
  neu2 = bp.neurons.LIF(1)
  syn = STP(neu1, neu2, bp.connect.All2All(), **kwargs)
  net = bp.Network(pre=neu1, syn=syn, post=neu2)

  # 分段电流
  inputs, dur = bp.inputs.section_input(values=[22., 0., 22., 0.],
                                        durations=[200., 200., 25., 75.],
                                        return_length=True)
  # 运行模拟
  runner = bp.DSRunner(net,
                           inputs=[('pre.input', inputs, 'iter')],
                           monitors=['syn.u', 'syn.x', 'syn.g'])
  runner.run(dur)

  # 可视化
  # fig, gs = plt.subplots(2, 1, figsize=(6, 4.5))
  fig, gs = bp.visualize.get_figure(2, 1)
  ax = fig.add_subplot(gs[0, 0])
  plt.plot(runner.mon.ts, runner.mon['syn.x'][:, 0], label=r'$x$')
  plt.plot(runner.mon.ts, runner.mon['syn.u'][:, 0], label=r'$u$', linestyle='--')
  plt.legend(loc='center right')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  if title: plt.title(title)

  ax = fig.add_subplot(gs[1, 0])
  plt.plot(runner.mon.ts, runner.mon['syn.g'][:, 0], label=r'$g$', color=u'#d62728')
  plt.legend(loc='center right')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  plt.xlabel(r'$t$ (ms)')
  # plt.tight_layout()
  plt.savefig(f'{title}_output.pdf',
              transparent=True, dpi=500)
  plt.show()


def run_STP2(title=None, **kwargs):
  # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
  neu1 = bp.neurons.SpikeTimeGroup(1,
                                   times=[100, 150, 200, 250, 300, 350, 400, 450, 500, 1000],
                                   indices=[0] * 10)
  neu2 = bp.neurons.LIF(1)
  syn = STP(neu1, neu2, bp.connect.All2All(), **kwargs)
  net = bp.Network(pre=neu1, syn=syn, post=neu2)

  # 分段电流
  inputs, dur = bp.inputs.section_input(values=[22., 0., 22., 0.],
                                        durations=[200., 200., 25., 75.],
                                        return_length=True)
  # 运行模拟
  runner = bp.DSRunner(net,
                           monitors=['syn.u', 'syn.x', 'syn.g', 'pre.spike'])
  runner.run(1200)

  # 可视化
  fig, ax = plt.subplots(figsize=(8, 2.5))
  plt.plot(runner.mon.ts, runner.mon['pre.spike'][:, 0], label='pre.spike')
  ax.spines['right'].set_color('none')
  ax.spines['top'].set_color('none')
  plt.legend()
  plt.xlabel(r'$t$ (ms)')
  plt.ylabel('Pre Spike')
  plt.savefig(f'../img/STP_illustration1.pdf',
              transparent=True, dpi=500)
  plt.show()

  fig, ax = plt.subplots(figsize=(8, 3))
  # ax = fig.add_subplot(gs[1:, 0])
  g = runner.mon['syn.g'][:, 0] * 10
  noise_g = g + np.random.normal(0., 0.005, g.shape)
  plt.plot(runner.mon.ts, noise_g, label='Data')
  plt.plot(runner.mon.ts, g, label='Fit', color=u'#ff7f0e')
  plt.legend()
  plt.xlabel(r'$t$ (ms)')
  plt.ylabel('Post Voltage (mV)')
  ax.spines['right'].set_color('none')
  ax.spines['top'].set_color('none')
  plt.savefig(f'../img/STP_illustration2.pdf',
              transparent=True, dpi=500)
  plt.show()


if __name__ == '__main__':
  # 短时程易化
  run_STP(title='STF', U=0.1, tau_d=15., tau_f=200.)
  # 短时程抑制
  run_STP(title='STD', U=0.4, tau_d=200., tau_f=15.)

# if __name__ == '__main__':
#   run_STP2(title='STD', U=0.4, tau_d=200., tau_f=15.)
