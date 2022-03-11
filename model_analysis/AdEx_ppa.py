import brainpy as bp
import numpy as np
import matplotlib.pyplot as plt

from AdEx_patterns import AdEx

bp.math.enable_x64()


def ppa2d(group, title, v_range=None, w_range=None, Iext=65., duration=400):
	v_range = [-70., -40.] if not v_range else v_range
	w_range = [-10., 50.] if not w_range else w_range

	# 使用BrainPy中的相平面分析工具
	phase_plane_analyzer = bp.analysis.PhasePlane2D(
		model=group,
		target_vars={'V': v_range, 'w': w_range},  # 待分析变量
		pars_update={'Iext': Iext},  # 需要更新的变量
		resolutions=0.05
	)

	# 画出V, w的零增长曲线
	phase_plane_analyzer.plot_nullcline()
	# 画出固定点
	phase_plane_analyzer.plot_fixed_point()
	# 画出向量场
	phase_plane_analyzer.plot_vector_field(plot_style=dict(color='lightgrey', density=1.))

	runner = bp.DSRunner(group, monitors=['V', 'w', 'spike'], inputs=('input', Iext))
	runner(duration)
	spike = runner.mon.spike.squeeze()
	s_idx = np.where(spike)[0]  # 找到所有发放动作电位对应的index
	s_idx = np.concatenate(([0], s_idx, [len(spike) - 1]))  # 加上起始点和终止点的index
	# 分段画出V, w的变化轨迹
	for i in range(len(s_idx) - 1):
		plt.plot(runner.mon.V[s_idx[i]: s_idx[i + 1]], runner.mon.w[s_idx[i]: s_idx[i + 1]], color='darkslateblue')

	# 画出虚线 x = V_reset
	plt.plot([group.V_reset, group.V_reset], w_range, '--', color='grey', zorder=-1)

	plt.xlim(v_range)
	plt.ylim(w_range)
	plt.title(title)
	plt.show()


# y_style = dict(color='lightcoral', alpha=.7, )
# fmt = y_style.pop('fmt', '.')
# # Tonic Spiking
# plt.plot(np.linspace(-70, -40, 500), np.zeros(500), fmt, **y_style)
# ppa2d(AdEx(1, tau=20., a=0., tau_w=30., b=60., V_reset=-55.),
#       title='Tonic Spiking', w_range=[-5, 75.])

# # Adaptation
# plt.plot(np.linspace(-70, -40, 500), np.zeros(500), fmt, **y_style)
# ppa2d(AdEx(1, tau=20., a=0., tau_w=100., b=5., V_reset=-55.),
#       title='Adaptation', w_range=[-5, 45.])
#
# # Initial Bursting
# ppa2d(AdEx(1, tau=5., a=0.5, tau_w=100., b=7., V_reset=-51.),
#       title='Initial Bursting', w_range=[-5, 50.])
#
# # Bursting
# ppa2d(AdEx(1, tau=5., a=-0.5, tau_w=100., b=7., V_reset=-47.),
#       title='Bursting', w_range=[-5, 60.])
#
# # Transient Spiking
# ppa2d(AdEx(1, tau=10., a=1., tau_w=100., b=10., V_reset=-60.),
#       title='Transient Spiking', w_range=[-5, 60.], Iext=55.)
#
# # Delayed Spiking
# ppa2d(AdEx(1, tau=5., a=-1., tau_w=100., b=5., V_reset=-60.),
#       title='Delayed Spiking', w_range=[-30, 20.], Iext=25.)
