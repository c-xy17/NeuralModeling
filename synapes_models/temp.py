import brainpy as bp
import matplotlib.pyplot as plt

neu1 = bp.dyn.LIF(1)
neu2 = bp.dyn.LIF(1)
syn1 = bp.dyn.STP(neu1, neu2, bp.connect.All2All(), U=0.1, tau_d=10, tau_f=100.)
net = bp.dyn.Network(pre=neu1, syn=syn1, post=neu2)

runner = bp.dyn.DSRunner(net, inputs=[('pre.input', 28.)], monitors=['syn.I', 'syn.u', 'syn.x'])
runner.run(150.)

 # plot
fig, gs = bp.visualize.get_figure(2, 1, 3, 7)

fig.add_subplot(gs[0, 0])
plt.plot(runner.mon.ts, runner.mon['syn.u'][:, 0], label='u')
plt.plot(runner.mon.ts, runner.mon['syn.x'][:, 0], label='x')
plt.legend()

fig.add_subplot(gs[1, 0])
plt.plot(runner.mon.ts, runner.mon['syn.I'][:, 0], label='I')
plt.legend()

plt.xlabel('Time (ms)')
plt.show()