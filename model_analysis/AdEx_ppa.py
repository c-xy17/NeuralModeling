import brainpy as bp
import numpy as np
import matplotlib.pyplot as plt

from AdEx_patterns import AdEx

bp.math.enable_x64()


def ppa2d(group, title, v_range=None, w_range=None, Iext=65., duration=400):
	v_range = [-70., -40.] if not v_range else v_range
	w_range = [-10., 50.] if not w_range else w_range

	phase_plane_analyzer = bp.analysis.PhasePlane2D(
		model=group,
		target_vars={'V': v_range, 'w': w_range},  # 待分析变量
		pars_update={'Iext': Iext},  # 需要更新的变量
		resolutions=0.05
	)

	# 画出V, w的零增长曲线
	res = phase_plane_analyzer.plot_nullcline(with_return=True, tol_nullcline=1e-3)
	# 画出固定点
	phase_plane_analyzer.plot_fixed_point()
	# 画出向量场
	phase_plane_analyzer.plot_vector_field(plot_style=dict(color='lightgrey', density=1.))

	runner = bp.DSRunner(group, monitors=['V', 'w', 'spike'], inputs=('input', Iext))
	runner(duration)
	s_idx = np.where(runner.mon.spike.squeeze())[0]  # 找到所有发放动作电位对应的index
	s_idx = np.concatenate(([0], s_idx))
	# 分段画出V, w的变化轨迹
	for i in range(len(s_idx) - 1):
		plt.plot(runner.mon.V[s_idx[i]: s_idx[i + 1]], runner.mon.w[s_idx[i]: s_idx[i + 1]], color='darkslateblue')

	# 画出虚线 x = V_reset
	plt.plot([group.V_reset, group.V_reset], w_range, '--', color='grey', zorder='-1')

	plt.xlim(v_range)
	plt.ylim(w_range)
	plt.title(title)
	plt.show()


ppa2d(AdEx(1, tau=20., a=1e-2, tau_w=30., b=60., V_reset=-55.),
      title='tonic spiking', w_range=[-5, 75.],
      v_range=[-70., -40.])
