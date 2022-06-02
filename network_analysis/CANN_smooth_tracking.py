import brainpy as bp
import brainpy.math as bm

from network_models.CANN import CANN1D

cann = CANN1D(num=512, k=8.1)


dur1, dur2, dur3 = 20., 20., 20.
num1 = int(dur1 / bm.get_dt())
num2 = int(dur2 / bm.get_dt())
num3 = int(dur3 / bm.get_dt())
position = bm.zeros(num1 + num2 + num3)
position[num1: num1 + num2] = bm.linspace(0., 12., num2)
position[num1 + num2:] = 12.
position = position.reshape((-1, 1))
Iext = cann.get_stimulus_by_pos(position)
runner = bp.dyn.DSRunner(cann,
                         inputs=('input', Iext, 'iter'),
                         monitors=['u'],
                         dyn_vars=cann.vars())
runner.run(dur1 + dur2 + dur3)
bp.visualize.animate_1D(
  dynamical_vars=[{'ys': runner.mon.u, 'xs': cann.x, 'legend': 'u'},
                  {'ys': Iext, 'xs': cann.x, 'legend': 'Iext'}],
  frame_step=5,
  frame_delay=50,
  show=True,
  # save_path='../../images/cann-tracking.gif'
)