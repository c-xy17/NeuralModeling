import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

from neuron_models.ExpIF_model import ExpIF


duration = 200
I = 6.

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6), sharex=True, sharey=True)

neu1 = ExpIF(1, delta_T=5.)
neu1.V[:] = bm.array([-68.])
runner = bp.DSRunner(neu1, monitors=['V'], inputs=('input', I), dt=0.01)
runner(duration)
ax1.plot(runner.mon.ts, runner.mon.V, color=u'#1f77b4', label='delta_T=5')
ax1.set_ylabel('V (mV)')
ax1.legend(loc='upper right')

neu1 = ExpIF(1, delta_T=1.)
neu1.V[:] = bm.array([-68.])
runner = bp.DSRunner(neu1, monitors=['V'], inputs=('input', I), dt=0.01)
runner(duration)
ax2.plot(runner.mon.ts, runner.mon.V, color=u'#ff7f0e', label='delta_T=1')
ax2.set_ylabel('V (mV)')
ax2.legend(loc='upper right')

neu1 = ExpIF(1, delta_T=0.02)
neu1.V[:] = bm.array([-68.])
runner = bp.DSRunner(neu1, monitors=['V'], inputs=('input', I), dt=0.005)
runner(duration)
ax3.plot(runner.mon.ts, runner.mon.V, color=u'#d62728', label='delta_T=0.02')
ax3.set_ylabel('V (mV)')
ax3.legend(loc='upper right')

ax3.set_xlabel('t (ms)')

plt.tight_layout()
plt.subplots_adjust(hspace=0.)
plt.show()
