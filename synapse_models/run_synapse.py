import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

from neuron_models.FRNeuron import FR

plt.rcParams.update({"font.size": 15})
plt.rcParams['font.sans-serif'] = ['Times New Roman']

def run_syn_LIF(syn_model, run_duration=30., **kwargs):
  # 定义突触前神经元、突触后神经元和突触连接，并构建网络
  pre_neu = bp.neurons.LIF(5)
  post_neu = bp.neurons.LIF(3)
  syn = syn_model(pre_neu, post_neu, conn=bp.conn.All2All(), **kwargs)
  net = bp.dyn.Network(pre=pre_neu, syn=syn, post=post_neu)

  # 运行模拟
  runner = bp.dyn.DSRunner(net,
                           monitors=['pre.V', 'syn.g', 'post.V'],
                           inputs=('pre.input', 35.))
  runner(run_duration)

  # 只选取第0个突触后神经元可视化
  plt.plot(runner.mon.ts, runner.mon['syn.g'][:, 0])
  plt.xlabel(r'$t$ (ms)')
  plt.ylabel(r'$g$')
  plt.show()


def run_delta_syn(syn_model, title, run_duration=200., **kwargs):
  # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
  neu1 = bp.neurons.SpikeTimeGroup(1,
                                   times=[20, 60, 100, 140, 180],
                                   indices=[0, 0, 0, 0, 0])
  neu2 = bp.neurons.HH(1, V_initializer=bp.init.Constant(-70.68))
  syn1 = syn_model(neu1, neu2, conn=bp.connect.All2All(), **kwargs)
  net = bp.dyn.Network(pre=neu1, syn=syn1, post=neu2)

  # 构建一个模拟器
  runner = bp.dyn.DSRunner(
    net,
    monitors=['pre.spike', 'post.V', 'syn.g']
  )
  runner.run(run_duration)

  # 可视化
  fig, gs = bp.visualize.get_figure(3, 1, 1.5, 6.)

  ax = fig.add_subplot(gs[0, 0])
  plt.plot(runner.mon.ts, runner.mon['pre.spike'], label='pre.spike')
  plt.legend(loc='upper right')
  plt.title(title)
  plt.xticks([])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  ax = fig.add_subplot(gs[1, 0])
  plt.plot(runner.mon.ts, runner.mon['syn.g'], label=r'$g$', color=u'#d62728')
  plt.legend(loc='upper right')
  plt.xticks([])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  ax = fig.add_subplot(gs[2, 0])
  plt.plot(runner.mon.ts, runner.mon['post.V'], label='post.V')
  plt.legend(loc='upper right')
  plt.xlabel(r'$t$ (ms)')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  # plt.savefig('../img/DeltaSynapse.pdf', transparent=True, dpi=500)
  plt.show()



def run_syn(syn_model, title, run_duration=200., sp_times=(10, 20, 30), **kwargs):
  # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
  neu1 = bp.neurons.SpikeTimeGroup(1, times=sp_times, indices=[0] * len(sp_times))
  neu2 = bp.neurons.HH(1, V_initializer=bp.init.Constant(-70.68))
  syn1 = syn_model(neu1, neu2, conn=bp.connect.All2All(), **kwargs)
  net = bp.dyn.Network(pre=neu1, syn=syn1, post=neu2)

  # 运行模拟
  runner = bp.dyn.DSRunner(net, monitors=['pre.spike', 'post.V', 'syn.g', 'post.input'])
  runner.run(run_duration)

  # 可视化
  fig, gs = bp.visualize.get_figure(7, 1, 0.8, 6.)

  ax = fig.add_subplot(gs[0, 0])
  plt.plot(runner.mon.ts, runner.mon['pre.spike'], label='pre.spike')
  plt.legend(loc='upper right')
  plt.title(title)
  plt.xticks([])
  ax.spines['top'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['right'].set_visible(False)

  ax = fig.add_subplot(gs[1:3, 0])
  plt.plot(runner.mon.ts, runner.mon['syn.g'], label=r'$g$', color=u'#d62728')
  plt.legend(loc='upper right')
  plt.xticks([])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  ax = fig.add_subplot(gs[3:5, 0])
  plt.plot(runner.mon.ts, runner.mon['post.input'], label='PSC', color=u'#d62728')
  plt.legend(loc='upper right')
  plt.xticks([])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  ax = fig.add_subplot(gs[5:7, 0])
  plt.plot(runner.mon.ts, runner.mon['post.V'], label='post.V')
  plt.legend(loc='upper right')
  plt.xlabel(r'$t$ (ms)')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  # plt.savefig('../img/DeltaSynapse.pdf', transparent=True, dpi=500)
  plt.show()


def run_syn2(syn_model, title, run_duration=100., Iext=5., **kwargs):
  # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
  neu1 = bp.neurons.HH(1, V_initializer=bp.init.Constant(-70.))
  neu2 = bp.neurons.HH(1, V_initializer=bp.init.Constant(-70.))
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


def run_syn_NMDA(syn_model, title, run_duration=200., sp_times=(10, 20, 30), **kwargs):
  # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
  neu1 = bp.neurons.SpikeTimeGroup(1, times=sp_times, indices=[0] * len(sp_times))
  neu2 = bp.neurons.HH(1, V_initializer=bp.init.Constant(-70.68))
  syn1 = syn_model(neu1, neu2, conn=bp.connect.All2All(), **kwargs)
  net = bp.dyn.Network(pre=neu1, post=neu2, syn=syn1)

  # 运行模拟
  post_Iext = bp.inputs.spike_input(sp_times=[130],
                                    sp_lens=2.,
                                    sp_sizes=6.,
                                    duration=run_duration)
  runner = bp.dyn.DSRunner(net,
                           inputs=[('post.input', post_Iext, 'iter')],
                           monitors=['pre.spike', 'post.V', 'syn.g', 'syn.b', 'post.input'])
  runner.run(run_duration)

  # 可视化
  fig, gs = bp.visualize.get_figure(9, 1, 0.6, 5)
  ax = fig.add_subplot(gs[0, 0])
  plt.plot(runner.mon.ts, runner.mon['pre.spike'], label='pre.spike')
  plt.legend(loc='upper right')
  plt.xticks([])
  plt.title(title)
  ax.spines['top'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['right'].set_visible(False)

  ax = fig.add_subplot(gs[1:3, 0])
  plt.plot(runner.mon.ts, runner.mon['post.V'], label='post.V')
  plt.xticks([])
  plt.legend(loc='upper right')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  ax = fig.add_subplot(gs[3:5, 0])
  plt.plot(runner.mon.ts, runner.mon['syn.g'], label=r'$g$', color=u'#d62728')
  plt.xticks([])
  plt.legend(loc='upper right')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  ax = fig.add_subplot(gs[5:7, 0])
  plt.plot(runner.mon.ts, runner.mon['syn.b'], label=r'$b$', color=u'#2ca02c')
  plt.xticks([])
  plt.legend(loc='upper right')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  ax = fig.add_subplot(gs[7:9, 0])
  plt.plot(runner.mon.ts, runner.mon['post.input'], label='PSC', color=u'#2cc0e0')
  plt.legend(loc='upper right')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.xlabel(r'$t$ [ms]')
  # plt.savefig('../img/Synapse.pdf', transparent=True, dpi=500)
  plt.show()


def run_syn_GABAb(syn_model, title, run_duration=200., Iext=0., **kwargs):
  # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
  neu1 = bp.neurons.SpikeTimeGroup(1, [10., ], [0.])
  neu2 = bp.neurons.HH(1, V_initializer=bp.init.Constant(-70.))
  syn1 = syn_model(neu1, neu2, conn=bp.connect.All2All(), **kwargs)
  net = bp.dyn.Network(pre=neu1, syn=syn1, post=neu2)

  runner = bp.dyn.DSRunner(net,
                           monitors=['pre.spike', 'syn.r', 'syn.G', 'syn.g'],
                           dt=0.01)
  runner.run(run_duration)

  # 可视化
  fig, gs = bp.visualize.get_figure(3, 1, 2.3, 7.5)
  ax = fig.add_subplot(gs[0, 0])
  plt.plot(runner.mon.ts, runner.mon['pre.spike'], label='pre.spike')
  plt.title(title)
  plt.xticks([])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.legend()

  ax = fig.add_subplot(gs[1, 0])
  plt.plot(runner.mon.ts, runner.mon['syn.r'], label=r'$r$', color=u'#d62728', linestyle='--')
  plt.plot(runner.mon.ts, runner.mon['syn.G'] / 4, label=r'$G/4$', color='lime')
  plt.legend()
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.xticks([])

  ax = fig.add_subplot(gs[2, 0])
  plt.plot(runner.mon.ts, runner.mon['syn.g'], label=r'$g$', color=u'#2ca02c')
  plt.legend()
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.xlabel(r'$t$ [ms]')
  # plt.savefig('../img/Synapse.pdf', transparent=True, dpi=500)
  plt.show()


def run_syn_GJ(syn_model, title, run_duration=100., Iext=7.5, **kwargs):
  # 定义神经元组和突触连接，并构建神经网络
  neu = bp.neurons.HH(2, V_initializer=bp.init.Constant(-70.68))
  syn = syn_model(neu, neu, conn=bp.connect.All2All(include_self=False), **kwargs)  # include_self=False: 自己和自己没有连接
  net = bp.dyn.Network(syn=syn, neu=neu)

  # 运行模拟
  Iext = bm.array([Iext, 0.])
  runner = bp.dyn.DSRunner(net,
                           inputs=[('neu.input', Iext)],
                           monitors=['neu.V', 'neu.input'])
  runner.run(run_duration)

  # 可视化
  fig, gs = bp.visualize.get_figure(2, 1, 2.25, 6)
  ax = fig.add_subplot(gs[0, 0])
  plt.plot(runner.mon.ts, runner.mon['neu.V'][:, 0], label='neu0-V')
  plt.plot(runner.mon.ts, runner.mon['neu.V'][:, 1], label='neu1-V', linestyle='--')
  plt.legend(loc='upper right')
  plt.ylabel('Potential (mV)')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.title(title)
  plt.xticks([])

  ax = fig.add_subplot(gs[1, 0])
  runner.mon['neu.input'] = runner.mon['neu.input'] - bm.as_ndarray(Iext)
  plt.plot(runner.mon.ts, runner.mon['neu.input'][:, 0],
           label='neu0-current', color=u'#48d688')
  plt.plot(runner.mon.ts, runner.mon['neu.input'][:, 1],
           label='neu1-current', color=u'#d64888', linestyle='--')
  plt.legend(loc='upper right')
  plt.ylabel('Current')
  plt.xlabel(r'$t$ [ms]')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  # plt.savefig('../img/GJ_output.pdf', transparent=True, dpi=500)
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

  plt.xlabel(r'$t$ (ms)')
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
  fig, gs = bp.visualize.get_figure(3, 1)

  ax = fig.add_subplot(gs[0, 0])
  plt.plot(runner.mon.ts, runner.mon['pre.r'][:, 0], label='pre0 $r$', color=u'#ff7f0e')
  plt.plot(runner.mon.ts, runner.mon['pre.r'][:, 1], label='pre1 $r$', color=u'#1f77b4', linestyle='--')
  plt.legend(loc='center right')
  plt.xticks([])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  ax = fig.add_subplot(gs[1, 0])
  plt.plot(runner.mon.ts, runner.mon['post.r'], label='post $r$', color=u'#d62728')
  plt.plot(runner.mon.ts, runner.mon['syn.theta_M'], label='$\\theta_\mathrm{M}$', color='gold', linestyle='--')
  plt.legend(loc='center right')
  plt.xticks([])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  ax = fig.add_subplot(gs[2, 0])
  plt.plot(runner.mon.ts, runner.mon['syn.w'][:, 0], label='$w0$', color=u'#ff7f0e')
  plt.plot(runner.mon.ts, runner.mon['syn.w'][:, 1], label='$w1$', color=u'#1f77b4', linestyle='--')
  plt.legend(loc='center right')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  plt.xlabel(r'$t$ (ms)')
  # plt.savefig('../img/BCM_output2.pdf', transparent=True, dpi=500)
  plt.show()

