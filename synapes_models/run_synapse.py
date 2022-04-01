import brainpy as bp
import matplotlib.pyplot as plt


def run_syn(syn_model, run_duration=30., **kwargs):
	pre_neu = bp.dyn.LIF(5)
	post_neu = bp.dyn.LIF(3)

	syn = syn_model(pre_neu, post_neu, conn=bp.conn.All2All(), **kwargs)

	net = bp.dyn.Network(pre=pre_neu, syn=syn, post=post_neu)
	runner = bp.DSRunner(net, monitors=['pre.V', 'syn.g', 'post.V'], inputs=('pre.input', 35.))
	runner(run_duration)

	plt.plot(runner.mon.ts, runner.mon['syn.g'])
	plt.show()
