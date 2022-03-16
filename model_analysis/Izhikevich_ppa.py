import brainpy as bp
import numpy as np
import matplotlib.pyplot as plt

from neuron_models.Izhikevich_model import Izhikevich

bp.math.enable_x64()


def ppa2d(group, title, v_range, u_range, Iext=10., duration=200):

	# 使用BrainPy中的相平面分析工具
	phase_plane_analyzer = bp.analysis.PhasePlane2D(
		model=group,
		target_vars={'V': v_range, 'u': u_range},  # 待分析变量
		pars_update={'Iext': Iext},  # 需要更新的变量
		resolutions=0.05
	)

	# 画出V, w的零增长曲线
	phase_plane_analyzer.plot_nullcline()
	# 画出固定点
	phase_plane_analyzer.plot_fixed_point()
	# 画出向量场
	phase_plane_analyzer.plot_vector_field(plot_style=dict(color='lightgrey', density=1.))

	runner = bp.DSRunner(group, monitors=['V', 'u', 'spike'], inputs=('input', Iext))
	runner(duration)
	spike = runner.mon.spike.squeeze()
	s_idx = np.where(spike)[0]  # 找到所有发放动作电位对应的index
	s_idx = np.concatenate(([0], s_idx, [len(spike) - 1]))  # 加上起始点和终止点的index
	# 分段画出V, w的变化轨迹
	for i in range(len(s_idx) - 1):
		plt.plot(runner.mon.V[s_idx[i]: s_idx[i + 1]], runner.mon.u[s_idx[i]: s_idx[i + 1]], color='darkslateblue')

	# 画出虚线 x = V_reset
	plt.plot([group.c, group.c], u_range, '--', color='grey', zorder=-1)

	plt.xlim(v_range)
	plt.ylim(u_range)
	plt.title(title)
	plt.show()


# Bursting
def reset():
	return Izhikevich(1, a=0.02, b=0.2, c=-65., d=2.)


# izhi = reset()
# I = 5.
# runner = bp.DSRunner(izhi, monitors=['V', 'u'], inputs=('input', I))
# runner(200.)
# bp.visualize.line_plot(runner.mon.ts, runner.mon.V, show=False)
# bp.visualize.line_plot(runner.mon.ts, runner.mon.u, show=True)
#
# izhi = reset()
# ppa2d(izhi, title='Bursting', v_range=[-80., -40.], u_range=[-20., 5.], Iext=I)
