import brainpy as bp
import brainpy.math as bm

from network_models.CANN import CANN1D

cann = CANN1D(num=512, k=0.1)


I1 = cann.get_stimulus_by_pos(0.)
Iext, duration = bp.inputs.section_input(values=[0., I1, 0.],
                                         durations=[1., 8., 8.],
                                         return_length=True)
runner = bp.dyn.DSRunner(cann,
                         inputs=['input', Iext, 'iter'],
                         monitors=['u'],
                         dyn_vars=cann.vars())
runner.run(duration)

bp.visualize.animate_1D(
  dynamical_vars=[{'ys': runner.mon.u, 'xs': cann.x, 'legend': 'u'},
                  {'ys': Iext, 'xs': cann.x, 'legend': 'Iext'}],
  frame_step=1,
  frame_delay=100,
  show=True,
  # save_path='../../images/cann-encoding.gif'
)

cann.k = 8.1

dur1, dur2, dur3 = 10., 30., 0.
num1 = int(dur1 / bm.get_dt())
num2 = int(dur2 / bm.get_dt())
num3 = int(dur3 / bm.get_dt())
Iext = bm.zeros((num1 + num2 + num3,) + cann.size)
Iext[:num1] = cann.get_stimulus_by_pos(0.5)
Iext[num1:num1 + num2] = cann.get_stimulus_by_pos(0.)
Iext[num1:num1 + num2] += 0.1 * cann.A * bm.random.randn(num2, *cann.size)

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
  # save_path='../../images/cann-decoding.gif'
)