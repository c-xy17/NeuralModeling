import brainpy as bp
import numpy as np
import matplotlib.pyplot as plt

from neuron_models.GIF_model import GIF

bp.math.enable_x64()


def subplot(i, neu, title=None, input=('input', 1.), dur=200):
  plt.subplot(3, 3, i)
  runner = bp.DSRunner(neu, monitors=['V', 'theta'], inputs=input)
  runner(dur)
  bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='$V$', show=False)
  bp.visualize.line_plot(runner.mon.ts, runner.mon.theta, legend='$\Theta$', show=False)
  plt.title(title)


def run(i, model, duration, I_ext, title):
  runner = bp.DSRunner(model, inputs=('input', I_ext, 'iter'), monitors=['V', 'theta'])
  runner.run(duration)

  ts = runner.mon.ts

  ax1 = fig.add_subplot(gs[int(i / col), i % col])
  ax1.title.set_text(title)

  ax1.plot(ts, runner.mon.V[:, 0], label='V')
  ax1.plot(ts, runner.mon.theta[:, 0], label='$\Theta$')
  ax1.set_xlabel('t (ms)')
  if i % col == 0:
    ax1.set_ylabel('Membrane potential')
  ax1.set_xlim(-0.1, ts[-1] + 0.1)
  plt.legend(fontsize=7.5, loc=(0.8, 0.5))

  ax2 = ax1.twinx()
  ax2.plot(ts, I_ext, color='turquoise', label='$I$')
  ax2.set_xlabel('Time (ms)')
  if (i+1) % col == 0:
    ax2.set_ylabel('External input')
  ax2.set_xlim(-0.1, ts[-1] + 0.1)
  ax2.set_ylim(-5., 20.)
  plt.legend(fontsize=7.5, loc=(0.8, 0.35))


row, col = 5, 4
fig, gs = bp.visualize.get_figure(row, col, 4, 6)

Iext, duration = bp.inputs.constant_current([(1.5, 200.)])
neu = GIF(1)
run(0, neu, duration, Iext, 'Tonic Spiking')

Iext, duration = bp.inputs.constant_current([(1.+1e-6, 500.)])
neu = GIF(1)
run(1, neu, duration, Iext, 'Class 1')

Iext, duration = bp.inputs.constant_current([(2., 200.)])
neu = GIF(1, a=0.005)
run(2, neu, duration, Iext, 'Spike Frequency Adaptation')

Iext, duration = bp.inputs.constant_current([(1.5, 500.)])
neu = GIF(1, a=0.005)
run(3, neu, duration, Iext, 'Phasic Spiking')

Iext, duration = bp.inputs.constant_current([(1.5, 100.), (0, 500.), (0.5, 100.),
                                             (1., 100.), (1.5, 100.), (0., 100.)])
neu = GIF(1, a=0.005)
run(4, neu, duration, Iext, 'Accomodation')

Iext, duration = bp.inputs.constant_current([(1.5, 25.), (0., 175.), (-1.5, 25.),
                                             (0., 25.), (1.5, 25.), (0., 125.)])
neu = GIF(1, a=0.005)
run(5, neu, duration, Iext, 'Threshold Variability')

Iext, duration = bp.inputs.constant_current([(0, 50.), (-3.5, 750.), (0., 200.)])
neu = GIF(1, a=0.005)
run(6, neu, duration, Iext, 'Rebound Spiking')

Iext, duration = bp.inputs.constant_current([(2 * (1. + 1e-6), 200.)])
neu = GIF(1, a=0.005)
neu.theta[:] = -30.
run(7, neu, duration, Iext, 'Class 2')

Iext, duration = bp.inputs.constant_current([(1.5, 20.), (0., 10.), (1.5, 20.), (0., 250.),
                                             (1.5, 20.), (0., 30.), (1.5, 20.), (0., 30.)])
neu = GIF(1, a=0.005)
run(8, neu, duration, Iext, 'Integrator')

Iext, duration = bp.inputs.constant_current([(1.5, 100.), (1.7, 400.), (1.5, 100.), (1.7, 400.)])
neu = GIF(1, a=0.005)
run(9, neu, duration, Iext, 'Input Bistability')

Iext, duration = bp.inputs.constant_current([(-1., 400.)])
neu = GIF(1, theta_reset=-60., theta_inf=-120.)
neu.theta[:] = -50.
run(10, neu, duration, Iext, 'Hyperpolarization-induced Spiking')

Iext, duration = bp.inputs.constant_current([(-1., 400.)])
neu = GIF(1, theta_reset=-60., theta_inf=-120., A1=10., A2=-0.6)
neu.theta[:] = -50.
run(11, neu, duration, Iext, 'Hyperpolarization-induced Bursting')

Iext, duration = bp.inputs.constant_current([(2., 500.)])
neu = GIF(1, a=0.005, A1=10., A2=-0.6)
run(12, neu, duration, Iext, 'Tonic Bursting')

Iext, duration = bp.inputs.constant_current([(1.5, 500.)])
neu = GIF(1, a=0.005, A1=10., A2=-0.6)
run(13, neu, duration, Iext, 'Phasic Bursting')

Iext, duration = bp.inputs.constant_current([(0, 100.), (-3.5, 500.), (0., 400.)])
neu = GIF(1, a=0.005, A1=10., A2=-0.6)
run(14, neu, duration, Iext, 'Rebound Bursting')

Iext, duration = bp.inputs.constant_current([(2., 500.)])
neu = GIF(1, a=0.005, A1=5., A2=-0.3)
run(15, neu, duration, Iext, 'Mixed Mode')

Iext, duration = bp.inputs.constant_current([(2., 15.), (0, 185.)])
neu = GIF(1, a=0.005, A1=5., A2=-0.3)
run(16, neu, duration, Iext, 'Afterpotentials')

Iext, duration = bp.inputs.constant_current([(5., 10.), (0., 90.), (5., 10.), (0., 90.)])
neu = GIF(1, A1=8., A2=-0.1)
run(17, neu, duration, Iext, 'Basal Bistability')

Iext, duration = bp.inputs.constant_current([(5., 5.), (0., 5.),  (4., 5.), (0., 385.),
                                             (5., 5.), (0., 45.), (4., 5.), (0., 345.)])
neu = GIF(1, a=0.005, A1=-3., A2=0.5)
run(18, neu, duration, Iext, 'Preferred Frequency')

Iext, duration = bp.inputs.constant_current([(8., 2.), (0, 48.)])
neu = GIF(1, a=-0.08)
run(19, neu, duration, Iext, 'Spike Latency')

plt.show()
