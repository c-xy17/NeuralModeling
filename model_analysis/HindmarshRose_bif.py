import brainpy as bp

from neuron_models.HindmarshRose_model import HindmarshRose

bp.math.enable_x64()

model = HindmarshRose(1)

bif = bp.analysis.Bifurcation2D(
    model=model,
    target_vars={'x': [-2, 2], 'y': [-20, 5]},
    fixed_vars={'z': 1.8},
    target_pars={'Iext': [0., 2.5]},
    resolutions={'Iext': 0.01}
)
bif.plot_bifurcation(show=True)
