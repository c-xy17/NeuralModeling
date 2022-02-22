import matplotlib.pyplot as plt
import numpy as np

from neuron_models.QIF_model import QIF

qif = QIF(1)
Vs = np.linspace(-80, -30, 200)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey='col')

# I_ext = 0
ax = axes[0]
dvdts = qif.derivative(Vs, 0., 0.)
ax.plot(Vs, np.zeros(200), '--', color=u'#333333')
ax.plot(Vs, dvdts)
ax.set_title('External Input = 0')
ax.set_xlim(-80, -30)
ax.set_xlabel('V')
ax.set_ylabel('dV/dt')

# I_ext = 3
ax = axes[1]
dvdts = qif.derivative(Vs, 0., 3.)
ax.plot(Vs, np.zeros(200), '--', color=u'#333333')
ax.plot(Vs, dvdts)
ax.set_title('External Input = 3')
ax.set_xlim(-80, -30)
ax.set_xlabel('V')

# I_ext = 10
ax = axes[2]
dvdts = qif.derivative(Vs, 0., 10.)
ax.plot(Vs, np.zeros(200), '--', color=u'#333333')
ax.plot(Vs, dvdts)
ax.set_title('External Input = 10')
ax.set_xlim(-80, -30)
ax.set_xlabel('V')

ax.get_shared_y_axes().join(axes[0], axes[1], axes[2])
plt.show()
