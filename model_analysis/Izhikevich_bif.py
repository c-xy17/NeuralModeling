import brainpy as bp

bp.math.enable_x64()

from neuron_models.Izhikevich_model import Izhikevich

a, b = 0.02, 0.2


@bp.odeint
def dV(V, t, u, Iext):
	return 0.04 * V * V + 5 * V + 140 - u + Iext


@bp.odeint
def du(u, t, V):
	return a * (b * V - u)


model = Izhikevich(1)

bif = bp.analysis.Bifurcation2D(
	model=[dV, du],
	target_vars={'V': [-80., -40.], 'u': [-20., 5.]},
	target_pars={'Iext': [0., 10.]},
	resolutions={'Iext': 0.01}
)
res = bif.plot_bifurcation(show=False, select_candidates='fx-nullcline')
bif.show_figure()
