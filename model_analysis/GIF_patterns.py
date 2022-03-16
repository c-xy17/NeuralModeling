import brainpy as bp
import numpy as np
import matplotlib.pyplot as plt

from neuron_models.GIF_model import GIF


def subplot(i, neu, title=None, input=('input', 1.), dur=200):
  plt.subplot(3, 3, i)
  runner = bp.DSRunner(neu, monitors=['V', 'V_th'], inputs=input)
  runner(dur)
  bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='$V$', show=False)
  bp.visualize.line_plot(runner.mon.ts, runner.mon.V_th, legend='$V_\mathrm{th}$', show=False)
  plt.title(title)


def run(i, model, duration, I_ext):
  runner = bp.StructRunner(model,
                           inputs=('input', I_ext, 'iter'),
                           monitors=['V', 'V_th'])
  runner.run(duration)

  ts = runner.mon.ts

  ax1 = fig.add_subplot(gs[int(i/col), i%col])
  #ax1.title.set_text(f'{mode}')

  ax1.plot(ts, runner.mon.V[:, 0], label='V')
  ax1.plot(ts, runner.mon.V_th[:, 0], label='V_th')
  ax1.set_xlabel('Time (ms)')
  ax1.set_ylabel('Membrane potential')
  ax1.set_xlim(-0.1, ts[-1] + 0.1)
  plt.legend()

  ax2 = ax1.twinx()
  ax2.plot(ts, I_ext, color='turquoise', label='input')
  ax2.set_xlabel('Time (ms)')
  ax2.set_ylabel('External input')
  ax2.set_xlim(-0.1, ts[-1] + 0.1)
  ax2.set_ylim(-5., 20.)
  # plt.legend(loc='lower left')


row, col = 3, 3
fig, gs = bp.visualize.get_figure(row, col, 4, 6)

Iext, duration = bp.inputs.constant_current([(1.5, 200.)])
neu = GIF(1)
run(0, neu, duration, Iext)

Iext, duration = bp.inputs.constant_current([(2., 200.)])
neu = GIF(1, a=0.005)
run(1, neu, duration, Iext)

plt.tight_layout()
plt.show()
