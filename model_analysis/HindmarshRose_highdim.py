import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

from neuron_models.HindmarshRose_model import HindmarshRose

bp.math.enable_x64()


model = HindmarshRose(1)


def f_cell(x):
  x, y, z = x
  res = model.derivative(x, y, z, 0, 2.)
  return bm.asarray(res)


finder = bp.analysis.SlowPointFinder(f_cell=f_cell)
finder.find_fps_with_gd_method(
    candidates=bm.random.random((1000, 3)), tolerance=1e-5, num_batch=200,
    opt_setting=dict(method=bp.optim.Adam,
                     lr=bp.optim.ExponentialDecay(0.01, 1, 0.9999)),
)
finder.filter_loss(1e-5)
finder.keep_unique()

print(finder.fixed_points)
