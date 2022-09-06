import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

from neuron_models.ExpIF_model import ExpIF


def visualize1():
  duration = 200
  I = 6.

  fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6), sharex=True, sharey=True)

  neu1 = ExpIF(1, delta_T=5.)
  neu1.V[:] = bm.array([-68.])
  runner = bp.DSRunner(neu1, monitors=['V', 'spike'], inputs=('input', I), dt=0.01)
  runner(duration)
  runner.mon.V = np.where(runner.mon.spike, neu1.V_th, runner.mon.V)
  ax1.plot(runner.mon.ts, runner.mon.V, color=u'#1f77b4', label='delta_T=5')
  ax1.set_ylabel('V (mV)')
  ax1.legend(loc='upper right')

  neu1 = ExpIF(1, delta_T=1.)
  neu1.V[:] = bm.array([-68.])
  runner = bp.DSRunner(neu1, monitors=['V', 'spike'], inputs=('input', I), dt=0.01)
  runner(duration)
  runner.mon.V = np.where(runner.mon.spike, neu1.V_th, runner.mon.V)
  ax2.plot(runner.mon.ts, runner.mon.V, color=u'#ff7f0e', label='delta_T=1')
  ax2.set_ylabel('V (mV)')
  ax2.legend(loc='upper right')

  neu1 = ExpIF(1, delta_T=0.02)
  neu1.V[:] = bm.array([-68.])
  runner = bp.DSRunner(neu1, monitors=['V', 'spike'], inputs=('input', I), dt=0.005)
  runner(duration)
  runner.mon.V = np.where(runner.mon.spike, neu1.V_th, runner.mon.V)
  ax3.plot(runner.mon.ts, runner.mon.V, color=u'#d62728', label='delta_T=0.02')
  ax3.set_ylabel('V (mV)')
  ax3.legend(loc='upper right')

  ax3.set_xlabel('t (ms)')

  plt.tight_layout()
  plt.subplots_adjust(hspace=0.)
  plt.show()


def visualize2():
  bm.enable_x64()
  neu1 = ExpIF(3, delta_T=bm.asarray([5., 1., 0.02]), method='exp_euler')
  neu1.V[:] = -68.
  runner = bp.DSRunner(neu1,
                       monitors=['V', 'spike'],
                       inputs=('input', 8.),
                       dt=0.001)
  runner(30.)

  fig, gs = bp.visualize.get_figure(1, 1, 3, 8)
  ax = fig.add_subplot(gs[0, 0])
  runner.mon.V = np.where(runner.mon.spike, neu1.V_th, runner.mon.V)
  ax.plot(runner.mon.ts, runner.mon.V[:, 0], label=r'$\Delta_T$=5.')
  ax.plot(runner.mon.ts, runner.mon.V[:, 1], label=r'$\Delta_T$=1.')
  ax.plot(runner.mon.ts, runner.mon.V[:, 2], label=r'$\Delta_T$=0.02')
  ax.set_ylabel('V [mV]')
  ax.set_xlabel('Time [ms]')
  ax.legend(loc='upper right')
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  plt.show()


if __name__ == '__main__':
    visualize2()

