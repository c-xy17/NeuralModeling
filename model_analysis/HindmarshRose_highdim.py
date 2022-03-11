import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

from neuron_models.HindmarshRose_model import HindmarshRose

bp.math.enable_x64()


model = HindmarshRose(1)

finder = bp.analysis.SlowPointFinder(f_cell=model.derivative)
finder.find_fps_with_gd_method(
    candidates=bm.random.random((1000, 2)), tolerance=1e-5, num_batch=200,
    opt_setting=dict(method=bm.optimizers.Adam,
                     lr=bm.optimizers.ExponentialDecay(0.01, 1, 0.9999)),
)
finder.filter_loss(1e-5)
finder.keep_unique()

finder.fixed_points()
