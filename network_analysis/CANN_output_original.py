import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt

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

# 定义函数
def plot_animate(frame_step=5, frame_delay=50):
  bp.visualize.animate_1D(dynamical_vars=[{'ys': runner.mon.u, 'xs': cann.x, 'legend': 'u'},
                                          {'ys': Iext, 'xs': cann.x, 'legend': 'Iext'}],
                          frame_step=frame_step, frame_delay=frame_delay,
                          show=True)

# 调用函数
plot_animate(frame_step=1, frame_delay=100)

cann = CANN1D(num=512, k=8.1)

dur1, dur2, dur3 = 10., 30., 0.
num1 = int(dur1 / bm.get_dt())
num2 = int(dur2 / bm.get_dt())
num3 = int(dur3 / bm.get_dt())
Iext = np.zeros((num1 + num2 + num3,) + cann.size)
Iext[:num1] = cann.get_stimulus_by_pos(0.5)
Iext[num1:num1 + num2] = cann.get_stimulus_by_pos(0.)
Iext[num1:num1 + num2] += 0.1 * cann.A * np.random.randn(num2, *cann.size)

runner = bp.dyn.DSRunner(cann,
                         inputs=['input', Iext, 'iter'],
                         monitors=['u'],
                         dyn_vars=cann.vars())
runner.run(dur1 + dur2 + dur3)
plot_animate()
