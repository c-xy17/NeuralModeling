import brainpy as bp
import brainpy.math as bm

from neuron_models.Izhikevich_model import Izhikevich
from Izhikevich_ppa import ppa2d

bp.math.enable_x64()

a, b = 0.02, 0.2


@bp.odeint
def dV(V, t, u, Iext):
	return 0.04 * V * V + 5 * V + 140 - u + Iext


@bp.odeint
def du(u, t, V):
	return a * (b * V - u)


# model = Izhikevich(1)
#
# # 定义分析器
# bif = bp.analysis.Bifurcation2D(
# 	model=model,
# 	target_vars={'V': [-75., -45.], 'u': [-17., -7.]},  # 设置变量的分析范围
# 	target_pars={'Iext': [0., 6.]},  # 设置参数的范围
# 	resolutions={'Iext': 0.02}  # 设置分辨率
# )
#
# # 进行分析
# res = bif.plot_bifurcation(show=True)

def pattern_and_ppa(model, v_range, u_range, Iext, dur=400):
	model.V.value = bm.ones_like(model.V) * model.c
	model.u.value = bm.ones_like(model.u) * model.V * model.b
	runner = bp.DSRunner(model, monitors=['V', 'u'], inputs=('input', Iext))
	runner(dur)
	bp.visualize.line_plot(runner.mon.ts, runner.mon.V, title='I = {}'.format(Iext), show=False)
	bp.visualize.line_plot(runner.mon.ts, runner.mon.u, show=True)

	model = Izhikevich(1, c=-68)
	model.V.value = bm.ones_like(model.V) * model.c
	model.u.value = bm.ones_like(model.u) * model.V * model.b
	ppa2d(model, 'I = {}'.format(Iext), v_range, u_range, Iext, dur)


I = 3.7
v_range = [-80., -45.]
u_range = [-20., 0.]
model = Izhikevich(1, c=-68)
pattern_and_ppa(model, v_range, u_range, I)
