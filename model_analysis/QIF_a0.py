import brainpy as bp
import brainpy.math as bm
from neuron_models.QIF_model import QIF

import matplotlib.pyplot as plt

duration = 500

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 6), sharex=True)

neu1 = QIF(1, a_0=0.005)
neu1.V[:] = bm.array([-68.])
runner = bp.StructRunner(neu1, monitors=['V'], inputs=('input', 5.))
runner(duration)
ax1.plot(runner.mon.ts, runner.mon.V, color=u'#d62728', label='a_0=0.005')
ax1.set_ylabel('V')
ax1.legend(loc='upper right')

neu1 = QIF(1, a_0=0.045)
neu1.V[:] = bm.array([-68.])
runner = bp.StructRunner(neu1, monitors=['V'], inputs=('input', 5.))
runner(duration)
ax2.plot(runner.mon.ts, runner.mon.V, color=u'#1f77b4', label='a_0=0.045')
ax2.set_ylabel('V')
ax2.legend(loc='upper right')

neu1 = QIF(1, a_0=0.08)
neu1.V[:] = bm.array([-68.])
runner = bp.StructRunner(neu1, monitors=['V'], inputs=('input', 5.))
runner(duration)
ax3.plot(runner.mon.ts, runner.mon.V, color=u'#ff7f0e', label='a_0=0.08')
ax3.set_ylabel('V')
ax3.set_xlabel('Time (ms)')
ax3.legend(loc='upper right')

plt.tight_layout()
plt.show()
