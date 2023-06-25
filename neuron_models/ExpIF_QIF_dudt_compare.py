import brainpy as bp
import matplotlib.pyplot as plt
import numpy as np

from neuron_models.ExpIF_model import ExpIF
from neuron_models.QIF_model import QIF

bp.math.enable_x64()


def phase_plane_analysis(i, model, I_ext, res=0.005):
  fig.sca(axes[i])
  pp = bp.analysis.PhasePlane1D(
    model=model,
    target_vars={'V': [-80, -45]},
    pars_update={'Iext': I_ext},
    resolutions=res
  )
  pp.plot_vector_field()
  pp.plot_fixed_point()
  plt.ylim(-2, 10)
  plt.title('Input = {}'.format(I_ext))


fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey='all')  # 设置子图并共享y轴
inputs = [0., 20.]  # 设置不同大小的电流输入
expif = ExpIF(1, delta_T=2., V_T=-54.03)
# qif = QIF(1)

# phase_plane_analysis(0, qif, inputs[0])
phase_plane_analysis(0, expif, inputs[0])
phase_plane_analysis(1, expif, inputs[1])

plt.tight_layout()
plt.show()

expif = ExpIF(1, delta_T=2., V_T=-54.03)
qif = QIF(1)
Vs = np.linspace(-80, -45, 200)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey='all')

# I_ext = 0
ax = axes[0]
dvdts = qif.derivative(Vs, 0., 0.)
ax.plot(Vs, dvdts, color=u'#d62728', label='QIF')
dvdts = expif.derivative(Vs, 0., 0.)
ax.plot(Vs, np.zeros(200), '--', color=u'#333333')
ax.plot(Vs, dvdts, label='ExpIF')
ax.set_title('External Input = 0')
ax.set_xlim(-80, -45)
ax.set_ylim(-3, 10)
ax.set_xlabel('V')
ax.set_ylabel('dV/dt')
ax.legend()

ax = axes[1]
dvdts = expif.derivative(Vs, 0., 20.)
ax.plot(Vs, np.zeros(200), '--', color=u'#333333')
ax.plot(Vs, dvdts)
ax.set_title('External Input = 10')
ax.set_xlim(-80, -45)
ax.set_xlabel('V')

# ax = axes[2]
# dvdts = expif.derivative(Vs, 0., 20.)
# ax.plot(Vs, np.zeros(200), '--', color=u'#333333')
# ax.plot(Vs, dvdts)
# ax.set_title('External Input = 20')
# ax.set_xlim(-80, -45)
# ax.set_xlabel('V')

# ax.get_shared_y_axes().join(axes[0], axes[1], axes[2])
ax.get_shared_y_axes().join(axes[0], axes[1])
plt.show()
