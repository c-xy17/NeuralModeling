import brainpy as bp
import matplotlib.pyplot as plt

from network_models.decision_making_rate_model import DecisionMakingRateModel


dmnet = DecisionMakingRateModel(1, coherence=-12.8)

runner = bp.DSRunner(dmnet, monitors=['s1', 's2'], inputs=('mu0', 30.), dt=0.02)
runner.run(1000.)

plt.plot(runner.mon.ts, runner.mon.s1, label='s1')
plt.plot(runner.mon.ts, runner.mon.s2, label='s2')

plt.xlabel('t (ms)')
plt.ylabel('population activity')
plt.legend()
plt.show()

