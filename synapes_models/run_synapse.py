import brainpy as bp
import matplotlib.pyplot as plt


def run_syn(syn_model, run_duration=30., **kwargs):
	# 定义突触前神经元、突触后神经元和突触连接
	pre_neu = bp.dyn.LIF(5)
	post_neu = bp.dyn.LIF(3)
	syn = syn_model(pre_neu, post_neu, conn=bp.conn.All2All(), **kwargs)

	# 构建网络并运行模拟
	net = bp.dyn.Network(pre=pre_neu, syn=syn, post=post_neu)
	runner = bp.DSRunner(net, monitors=['pre.V', 'syn.g', 'post.V'], inputs=('pre.input', 35.))
	runner(run_duration)

	# 只选取第0个突触后神经元可视化
	plt.plot(runner.mon.ts, runner.mon['syn.g'][:, 0])
	plt.xlabel('t (ms)')
	plt.ylabel('g')
	plt.show()



# neu1 = bp.dyn.HH(1)
# neu2 = bp.dyn.HH(1)
# syn1 = bp.dyn.ExpCOBA(neu1, neu2, bp.connect.All2All(), E=0.)
# net = bp.dyn.Network(pre=neu1, syn=syn1, post=neu2)
#
# runner = bp.dyn.DSRunner(net, inputs=[('pre.input', 5.)], monitors=['pre.V', 'post.V', 'syn.g'])
# runner.run(150.)
#
# fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
# fig.add_subplot(gs[0, 0])
# plt.plot(runner.mon.ts, runner.mon['pre.V'], label='pre-V')
# plt.plot(runner.mon.ts, runner.mon['post.V'], label='post-V')
# plt.legend()
#
# fig.add_subplot(gs[1, 0])
# plt.plot(runner.mon.ts, runner.mon['syn.g'], label='g')
# plt.legend()
# plt.show()
