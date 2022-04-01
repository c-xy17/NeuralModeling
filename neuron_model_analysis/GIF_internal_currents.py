import brainpy as bp
import numpy as np
import matplotlib.pyplot as plt

from neuron_models.GIF_model import GIF


def run(i, model, duration, I_ext, title):
  runner = bp.DSRunner(model, inputs=('input', I_ext, 'iter'), monitors=['V', 'theta', 'I1', 'I2'])
  runner.run(duration)

  ts = runner.mon.ts

  ax1 = fig.add_subplot(gs[0])
  ax1.title.set_text(title)

  ax1.plot(ts, runner.mon.V[:, 0], label='V')
  ax1.plot(ts, runner.mon.theta[:, 0], label='$\Theta$')
  ax1.set_ylabel('Membrane potential')
  ax1.set_xlim(-0.1, ts[-1] + 0.1)
  plt.legend(loc=(0.85, 0.45))

  ax2 = ax1.twinx()
  ax2.plot(ts, I_ext, color='turquoise', label='$I$')
  ax2.set_xlabel('Time (ms)')
  ax2.set_ylabel('External current')
  ax2.set_xlim(-0.1, ts[-1] + 0.1)
  ax2.set_ylim(-5., 20.)
  plt.legend(loc=(0.85, 0.35))

  ax3 = fig.add_subplot(gs[1])
  ax3.plot(runner.mon.ts, runner.mon.I1, color=u'#f7022a', label='I1')
  ax3.plot(runner.mon.ts, runner.mon.I2, color='lime', label='I2')
  ax3.set_ylabel('Internal currents')
  ax3.set_xlabel('t (ms)')

  plt.legend(loc=(0.85, 0.3))
  plt.subplots_adjust(hspace=0.)
  plt.show()


# Iext, duration = bp.inputs.constant_current([(2., 15.), (0, 185.)])
# neu = GIF(1, a=0.005, A1=5., A2=-0.3)
# runner = bp.DSRunner(neu, monitors=['V', 'theta', 'I1', 'I2'], inputs=('input', Iext, 'iter'))
# runner(duration)
#
# fig, ax1 = plt.subplots()
# ax1.plot(runner.mon.ts, runner.mon.V, color=u'#d62728', label='V')
# ax1.plot(runner.mon.ts, runner.mon.theta, color=u'#ff7f0e', label='$\Theta$')
#
# ax2 = ax1.twinx()
# ax2.plot(runner.mon.ts, runner.mon.I1, color=u'#1f77b4', label='I1')
# ax2.plot(runner.mon.ts, runner.mon.I2, color=u'#2ca02c', label='I2')
#
# fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
# plt.title('Afterpotentials')
# plt.show()

fig, gs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(7, 6), sharex='all')

Iext, duration = bp.inputs.constant_current([(0, 50.), (-3.5, 750.), (0., 200.)])
neu = GIF(1, a=0.005)
run(6, neu, duration, Iext, 'Rebound Spiking')
