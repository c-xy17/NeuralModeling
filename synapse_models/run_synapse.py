import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

from neuron_models.FRNeuron import FR


def run_syn_LIF(syn_model, run_duration=30., **kwargs):
  # 定义突触前神经元、突触后神经元和突触连接，并构建网络
  pre_neu = bp.dyn.LIF(5)
  post_neu = bp.dyn.LIF(3)
  syn = syn_model(pre_neu, post_neu, conn=bp.conn.All2All(), **kwargs)
  net = bp.dyn.Network(pre=pre_neu, syn=syn, post=post_neu)

  # 运行模拟
  runner = bp.DSRunner(net, monitors=['pre.V', 'syn.g', 'post.V'], inputs=('pre.input', 35.))
  runner(run_duration)

  # 只选取第0个突触后神经元可视化
  plt.plot(runner.mon.ts, runner.mon['syn.g'][:, 0])
  plt.xlabel('t (ms)')
  plt.ylabel('g')
  plt.show()


def run_syn(syn_model, title, run_duration=200., Iext=None, **kwargs):
  # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
  neu1 = bp.dyn.HH(1)
  neu2 = bp.dyn.HH(1)
  syn1 = syn_model(neu1, neu2, conn=bp.connect.All2All(), **kwargs)
  net = bp.dyn.Network(pre=neu1, syn=syn1, post=neu2)

  # 运行模拟
  if Iext is None:
    # 生成一个脉冲序列的电流输入
    Iext = bp.inputs.spike_input(sp_times=[20, 60, 100, 125, 150], sp_lens=2., sp_sizes=6.,
                                 duration=run_duration)
  runner = bp.dyn.DSRunner(net, inputs=[('pre.input', Iext, 'iter')],
                           monitors=['pre.V', 'post.V', 'syn.g'])
  runner.run(run_duration)

  # 可视化
  fig, gs = plt.subplots(2, 1, figsize=(6, 4.5))
  plt.sca(gs[0])
  plt.plot(runner.mon.ts, runner.mon['pre.V'], label='pre-V')
  plt.plot(runner.mon.ts, runner.mon['post.V'], label='PSP')
  plt.legend(loc='upper right')
  plt.title(title)

  plt.sca(gs[1])
  plt.plot(runner.mon.ts, runner.mon['syn.g'], label='g', color=u'#d62728')
  plt.legend(loc='upper right')

  plt.tight_layout()
  plt.show()


def run_syn2(syn_model, title, run_duration=100., Iext=5., **kwargs):
  # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
  neu1 = bp.dyn.HH(1)
  neu2 = bp.dyn.HH(1)
  syn1 = syn_model(neu1, neu2, conn=bp.connect.All2All(), **kwargs)
  net = bp.dyn.Network(pre=neu1, syn=syn1, post=neu2)

  # 运行模拟
  runner = bp.dyn.DSRunner(net, inputs=[('pre.input', Iext)], monitors=['pre.V', 'post.V', 'syn.g'])
  runner.run(run_duration)

  # 可视化
  fig, gs = plt.subplots(2, 1, figsize=(6, 4.5))
  plt.sca(gs[0])
  plt.plot(runner.mon.ts, runner.mon['pre.V'], label='pre-V')
  plt.plot(runner.mon.ts, runner.mon['post.V'], label='post-V')
  plt.legend(loc='upper right')
  plt.title(title)

  plt.sca(gs[1])
  plt.plot(runner.mon.ts, runner.mon['syn.g'], label='g', color=u'#d62728')
  plt.legend(loc='upper right')

  plt.tight_layout()
  plt.show()


def run_syn_NMDA(syn_model, title, run_duration=200., Iext=None, **kwargs):
  # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
  neu1 = bp.dyn.HH(1)
  neu2 = bp.dyn.HH(1)
  syn1 = syn_model(neu1, neu2, conn=bp.connect.All2All(), **kwargs)
  net = bp.dyn.Network(pre=neu1, post=neu2, syn=syn1)

  # 运行模拟
  if Iext is None:
    # 生成一个脉冲序列的电流输入
    Iext = bp.inputs.spike_input(sp_times=[20, 60, 100, 125, 150], sp_lens=2., sp_sizes=6.,
                                 duration=run_duration)
  post_Iext = bp.inputs.spike_input(sp_times=[130,], sp_lens=2., sp_sizes=6.,
                                 duration=run_duration)
  runner = bp.dyn.DSRunner(net,
                           inputs=[('pre.input', Iext, 'iter'), ('post.input', post_Iext, 'iter')],
                           monitors=['pre.V', 'post.V', 'syn.g', 'syn.b', 'post.input'])
  runner.run(run_duration)

  # 可视化
  fig, gs = plt.subplots(3, 1, figsize=(6, 6))
  plt.sca(gs[0])
  plt.plot(runner.mon.ts, runner.mon['pre.V'], label='pre-V')
  plt.plot(runner.mon.ts, runner.mon['post.V'], label='post-V')
  plt.legend(loc='upper right')
  plt.title(title)

  plt.sca(gs[1])
  plt.plot(runner.mon.ts, runner.mon['syn.g'], label='g', color=u'#d62728')
  gs[1].set_ylabel('g', rotation='horizontal')
  twinx = gs[1].twinx()
  twinx.plot(runner.mon.ts, runner.mon['syn.b'], label='b', color=u'#2ca02c')
  twinx.set_ylabel('b', rotation='horizontal')
  lines, labels = gs[1].get_legend_handles_labels()
  lines2, labels2 = twinx.get_legend_handles_labels()
  gs[1].legend(lines + lines2, labels + labels2, loc='upper right')

  plt.sca(gs[2])
  plt.plot(runner.mon.ts, runner.mon['post.input'], label='PSC', color=u'#2cc0e0')
  plt.legend(loc='upper right')

  plt.tight_layout()
  plt.show()


def run_syn_GABAb(syn_model, title, run_duration=200., Iext=None, **kwargs):
  # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
  neu1 = bp.dyn.HH(1)
  neu2 = bp.dyn.HH(1)
  syn1 = syn_model(neu1, neu2, conn=bp.connect.All2All(), **kwargs)
  net = bp.dyn.Network(pre=neu1, syn=syn1, post=neu2)

  # 运行模拟
  if Iext is None:
    # 生成一个脉冲序列的电流输入
    Iext = bp.inputs.spike_input(sp_times=[20, 60, 100, 125, 150], sp_lens=2., sp_sizes=6.,
                                 duration=run_duration)
  runner = bp.dyn.DSRunner(net,
                           inputs=[('pre.input', Iext)],
                           monitors=['pre.V', 'post.V', 'syn.r', 'syn.G', 'syn.g'])
  runner.run(run_duration)

  # 可视化
  fig, gs = plt.subplots(2, 1, figsize=(6, 4.5))
  plt.sca(gs[0])
  plt.plot(runner.mon.ts, runner.mon['pre.V'], label='pre-V')
  plt.plot(runner.mon.ts, runner.mon['post.V'], label='post-V')
  plt.legend(loc='upper right')
  plt.title(title)

  plt.sca(gs[1])
  plt.plot(runner.mon.ts, runner.mon['syn.r'], label='r', color=u'#d62728')
  plt.plot(runner.mon.ts, runner.mon['syn.G']/4, label='G/4', color='lime')
  twinx = gs[1].twinx()
  twinx.plot(runner.mon.ts, runner.mon['syn.g'], label='g', color=u'#2ca02c')
  twinx.set_ylabel('g', rotation='horizontal')
  lines, labels = gs[1].get_legend_handles_labels()
  lines2, labels2 = twinx.get_legend_handles_labels()
  gs[1].legend(lines + lines2, labels + labels2, loc='upper right')

  plt.tight_layout()
  plt.show()


def run_syn_GJ(syn_model, title, run_duration=100., Iext=7.5, **kwargs):
  # 定义神经元组和突触连接，并构建神经网络
  neu = bp.dyn.HH(2)
  syn = syn_model(neu, neu, conn=bp.connect.All2All(include_self=False), **kwargs)  # include_self=False: 自己和自己没有连接
  net = bp.dyn.Network(syn=syn, neu=neu)

  # 运行模拟
  runner = bp.dyn.DSRunner(net,
                           inputs=[('neu.input', bm.array([Iext, 0.]))],
                           monitors=['neu.V', 'syn.current'])
  runner.run(run_duration)

  # 可视化
  fig, gs = plt.subplots(2, 1, figsize=(6, 4.5))
  plt.sca(gs[0])
  plt.plot(runner.mon.ts, runner.mon['neu.V'][:, 0], label='neu0-V')
  plt.plot(runner.mon.ts, runner.mon['neu.V'][:, 1], label='neu1-V')
  plt.legend(loc='upper right')
  plt.title(title)

  plt.sca(gs[1])
  plt.plot(runner.mon.ts, runner.mon['syn.current'][:, 0],
           label='neu0-current', color=u'#48d688')
  plt.plot(runner.mon.ts, runner.mon['syn.current'][:, 1],
           label='neu1-current', color=u'#d64888')
  plt.legend(loc='upper right')

  plt.tight_layout()
  plt.show()


def run_FR(syn_model, I_pre, dur, **kwargs):
  # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
  pre = FR(2)
  post = FR(1)
  syn = syn_model(pre, post, conn=bp.conn.All2All(), **kwargs)
  net = bp.dyn.Network(pre=pre, post=post, syn=syn)

  # 运行模拟
  runner = bp.dyn.DSRunner(net,
                           # inputs=[('pre.input', I_pre.T, 'iter'), ('post.input', I2, 'iter')],
                           inputs=[('pre.input', I_pre.T, 'iter')],
                           monitors=['pre.r', 'post.r', 'syn.w'])
  runner(dur)

  # 可视化
  fig, gs = plt.subplots(3, 1, figsize=(6, 6))

  plt.sca(gs[0])
  plt.plot(runner.mon.ts, runner.mon['pre.r'][:, 0], label='pre0 $r$', color=u'#ff7f0e')
  plt.plot(runner.mon.ts, runner.mon['pre.r'][:, 1], label='pre1 $r$', color=u'#1f77b4')

  plt.sca(gs[1])
  plt.plot(runner.mon.ts, runner.mon['post.r'], label='post $r$', color=u'#d62728')

  plt.sca(gs[2])
  plt.plot(runner.mon.ts, runner.mon['syn.w'][:, 0], label='$w0$', color=u'#ff7f0e')
  plt.plot(runner.mon.ts, runner.mon['syn.w'][:, 1], label='$w1$', color=u'#1f77b4')

  for i in range(2):
    gs[i].set_xticks([])
  for i in range(3):
    gs[i].legend(loc='center right')

  plt.xlabel('t (ms)')
  plt.tight_layout()
  plt.subplots_adjust(hspace=0.)
  plt.show()

  def run_FR(syn_model, I_pre, dur, **kwargs):
    # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
    pre = FR(2)
    post = FR(1)
    syn = syn_model(pre, post, conn=bp.conn.All2All())
    net = bp.dyn.Network(pre=pre, post=post, syn=syn, **kwargs)

    # 运行模拟
    runner = bp.dyn.DSRunner(net,
                             # inputs=[('pre.input', I_pre.T, 'iter'), ('post.input', I2, 'iter')],
                             inputs=[('pre.input', I_pre.T, 'iter')],
                             monitors=['pre.r', 'post.r', 'syn.w'])
    runner(dur)

    # 可视化
    fig, gs = plt.subplots(3, 1, figsize=(6, 6))

    plt.sca(gs[0])
    plt.plot(runner.mon.ts, runner.mon['pre.r'][:, 0], label='pre0 $r$', color=u'#ff7f0e')
    plt.plot(runner.mon.ts, runner.mon['pre.r'][:, 1], label='pre1 $r$', color=u'#1f77b4')

    plt.sca(gs[1])
    plt.plot(runner.mon.ts, runner.mon['post.r'], label='post $r$', color=u'#d62728')

    plt.sca(gs[2])
    plt.plot(runner.mon.ts, runner.mon['syn.w'][:, 0], label='$w0$', color=u'#ff7f0e')
    plt.plot(runner.mon.ts, runner.mon['syn.w'][:, 1], label='$w1$', color=u'#1f77b4')

    for i in range(2):
      gs[i].set_xticks([])
    for i in range(3):
      gs[i].legend(loc='center right')

    plt.xlabel('t (ms)')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.)
    plt.show()


def run_BCM(syn_model, I_pre, dur, **kwargs):
  # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
  pre = FR(2)
  post = FR(1)
  syn = syn_model(pre, post, conn=bp.conn.All2All(), **kwargs)
  net = bp.dyn.Network(pre=pre, post=post, syn=syn)

  # 运行模拟
  runner = bp.dyn.DSRunner(net,
                           # inputs=[('pre.input', I_pre.T, 'iter'), ('post.input', I2, 'iter')],
                           inputs=[('pre.input', I_pre.T, 'iter')],
                           monitors=['pre.r', 'post.r', 'syn.w', 'syn.theta_M'])
  runner(dur)

  # 可视化
  fig, gs = plt.subplots(3, 1, figsize=(6, 6))

  plt.sca(gs[0])
  plt.plot(runner.mon.ts, runner.mon['pre.r'][:, 0], label='pre0 $r$', color=u'#ff7f0e')
  plt.plot(runner.mon.ts, runner.mon['pre.r'][:, 1], label='pre1 $r$', color=u'#1f77b4')

  plt.sca(gs[1])
  plt.plot(runner.mon.ts, runner.mon['post.r'], label='post $r$', color=u'#d62728')
  plt.plot(runner.mon.ts, runner.mon['syn.theta_M'], label='$\\theta_\mathrm{M}$', color='gold')

  plt.sca(gs[2])
  plt.plot(runner.mon.ts, runner.mon['syn.w'][:, 0], label='$w0$', color=u'#ff7f0e')
  plt.plot(runner.mon.ts, runner.mon['syn.w'][:, 1], label='$w1$', color=u'#1f77b4')

  for i in range(2):
    gs[i].set_xticks([])
  for i in range(3):
    gs[i].legend(loc='center right')

  plt.xlabel('t (ms)')
  plt.tight_layout()
  plt.subplots_adjust(hspace=0.)
  plt.show()
