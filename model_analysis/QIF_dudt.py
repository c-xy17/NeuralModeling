import matplotlib.pyplot as plt
import numpy as np

from neuron_models.QIF_model import QIF

# qif = QIF(1)
# Vs = np.linspace(-80, -30, 200)
#
# fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey='col')
#
# # I_ext = 0
# ax = axes[0]
# dvdts = qif.derivative(Vs, 0., 0.)
# ax.plot(Vs, np.zeros(200), '--', color=u'#333333')
# ax.plot(Vs, dvdts)
# ax.set_title('External Input = 0')
# ax.set_xlim(-80, -30)
# ax.set_xlabel('V')
# ax.set_ylabel('dV/dt')
#
# # I_ext = 3
# ax = axes[1]
# dvdts = qif.derivative(Vs, 0., 3.)
# ax.plot(Vs, np.zeros(200), '--', color=u'#333333')
# ax.plot(Vs, dvdts)
# ax.set_title('External Input = 3')
# ax.set_xlim(-80, -30)
# ax.set_xlabel('V')
#
# # I_ext = 10
# ax = axes[2]
# dvdts = qif.derivative(Vs, 0., 10.)
# ax.plot(Vs, np.zeros(200), '--', color=u'#333333')
# ax.plot(Vs, dvdts)
# ax.set_title('External Input = 10')
# ax.set_xlim(-80, -30)
# ax.set_xlabel('V')
#
# ax.get_shared_y_axes().join(axes[0], axes[1], axes[2])
# plt.show()

import brainpy as bp

bp.math.enable_x64()

def phase_plane_analysis(i, model, I_ext, res=0.005):
  fig.sca(axes[i])
  pp = bp.analysis.PhasePlane1D(
    model=model,
    target_vars={'V': [-80, -30]},
    pars_update={'Iext': I_ext},
    resolutions=res
  )
  pp.plot_vector_field()
  pp.plot_fixed_point()
  plt.title('Input = {}'.format(I_ext))

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey='all')  # 设置子图并共享y轴
inputs = [0., 3., 10]  # 设置不同大小的电流输入
qif = QIF(1)

for i in range(len(inputs)):
  phase_plane_analysis(i, qif, inputs[i])

# plt.subplots_adjust(wspace=0.)
plt.tight_layout()
plt.show()
