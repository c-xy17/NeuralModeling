import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

from neuron_models.HindmarshRose_model import HindmarshRose

bp.math.enable_x64()


model = HindmarshRose(1, s=0.5)

# 定义分析器
phase_plane_analyzer = bp.analysis.PhasePlane2D(
	model=model,
	target_vars={'x': [-2., 3.], 'y': [-13., 2.]},  # 待分析变量
	fixed_vars={'z': 0.07},                          # 固定变量
	pars_update={'Iext': 0.},                       # 需要更新的变量
	resolutions=0.01
)

plt.figure(figsize=(8, 5))

# 画出V, y的零增长曲线
phase_plane_analyzer.plot_nullcline()
# phase_plane_analyzer.plot_nullcline(x_style=dict(color='cornflowerblue', fmt='--', linewidth=2, alpha=0.7),
#                                     y_style=dict(color='lightcoral', fmt='--', linewidth=2, alpha=0.7))

# 画出固定点
phase_plane_analyzer.plot_fixed_point()

# 画出向量场
phase_plane_analyzer.plot_vector_field(plot_style=dict(color='lightgrey'))

# 画出V, y的变化轨迹
phase_plane_analyzer.plot_trajectory(
	{'x': [-0.7], 'y': [-3.]},
	duration=100., color='darkslateblue', linewidth=2, alpha=0.9,
	show=True
)
