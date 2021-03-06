import matplotlib.pyplot as plt
import numpy as np

from neuron_models.ExpIF_model import ExpIF
from neuron_models.QIF_model import QIF


plt.figure(figsize=(12, 4.5))
plt.subplot(1, 2, 1)

expif = ExpIF(1)
Vs = np.linspace(-80, -50, 500)
y1 = - 1 / expif.tau * (Vs - expif.V_rest)
x2 = np.ones(500) * expif.V_T
y2 = np.linspace(-1, 6, 500)

plt.plot(Vs, np.zeros(500), linewidth=1, color=u'#333333')
plt.plot(Vs, y1, '--', color='grey')
plt.plot(x2, y2, '--', color='grey')

expif = ExpIF(1, delta_T=5.)
dvdts = expif.derivative(Vs, 0., 0.)
plt.plot(Vs, dvdts, label='delta_T=5')

expif = ExpIF(1, delta_T=1.)
dvdts = expif.derivative(Vs, 0., 0.)
plt.plot(Vs, dvdts, label='delta_T=1')

expif = ExpIF(1, delta_T=0.2)
dvdts = expif.derivative(Vs, 0., 0.)
plt.plot(Vs, dvdts, label='delta_T=0.2')

expif = ExpIF(1, delta_T=0.05)
dvdts = expif.derivative(Vs, 0., 0.)
plt.plot(Vs, dvdts, label='delta_T=0.05')

plt.xlim(-80, -50)
plt.ylim(-1, 6)
plt.xlabel('V')
plt.ylabel('dV/dt')
plt.legend()


plt.subplot(1, 2, 2)

expif = ExpIF(1)
Vs = np.linspace(-80, -40, 500)
y1 = - 1 / expif.tau * (Vs - expif.V_rest)
y2 = np.linspace(-3, 6, 500)

plt.plot(Vs, np.zeros(500), linewidth=1, color=u'#333333')
plt.plot(Vs, y1, '--', color='grey')


expif = ExpIF(1, delta_T=0.2, V_T=-70)
x2 = np.ones(500) * expif.V_T
plt.plot(x2, y2, '--', color='grey')
dvdts = expif.derivative(Vs, 0., 0.)
plt.plot(Vs, dvdts, label='V_T=-70')

expif = ExpIF(1, delta_T=0.2, V_T=-60)
x2 = np.ones(500) * expif.V_T
plt.plot(x2, y2, '--', color='grey')
dvdts = expif.derivative(Vs, 0., 0.)
plt.plot(Vs, dvdts, label='V_T=-60')

expif = ExpIF(1, delta_T=0.2, V_T=-50)
x2 = np.ones(500) * expif.V_T
plt.plot(x2, y2, '--', color='grey')
dvdts = expif.derivative(Vs, 0., 0.)
plt.plot(Vs, dvdts, label='V_T=-50', color=u'#d62728')

plt.xlim(-80, -40)
plt.ylim(-3, 6)
plt.xlabel('V')
plt.ylabel('dV/dt')
plt.legend()
plt.show()
