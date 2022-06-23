import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

from network_models.CANN import CANN1D

# 可视化
def plot_response(ax, t):
  ts = int(t / bm.get_dt())
  I, u = Iext[ts], runner.mon.u[ts]
  ax.plot(cann.x, I, label='Iext')
  ax.plot(cann.x, u, label='U')
  ax.set_title('t = {} ms'.format(t))
  ax.set_xlabel('x')
  ax.legend()

# 生成CANN
cann = CANN1D(num=512, k=0.1)

# 生成外部刺激，从第2到12ms，持续10ms
dur1, dur2, dur3 = 2., 10., 10.
I1 = cann.get_stimulus_by_pos(0.)
Iext, duration = bp.inputs.section_input(values=[0., I1, 0.],
                                         durations=[dur1, dur2, dur3],
                                         return_length=True)
noise = bm.random.normal(0., 1., (int(duration / bm.get_dt()), len(I1)))
Iext += noise

# 运行数值模拟
runner = bp.dyn.DSRunner(cann,
                         inputs=['input', Iext, 'iter'],
                         monitors=['u'],
                         dyn_vars=cann.vars())
runner.run(duration)

# 可视化
fig, gs = plt.subplots(1, 2, figsize=(12, 4.5), sharey='all')

plot_response(gs[0], t=10.)
plot_response(gs[1], t=20.)

plt.tight_layout()
plt.show()

# bp.visualize.animate_1D(
#   dynamical_vars=[{'ys': runner.mon.u, 'xs': cann.x, 'legend': 'u'},
#                   {'ys': Iext, 'xs': cann.x, 'legend': 'Iext'}],
#   frame_step=1,
#   frame_delay=40,
#   show=True,
#   # save_path='../../images/cann-encoding.gif'
# )

# cann.k = 8.1
#
# dur1, dur2, dur3 = 10., 30., 0.
# num1 = int(dur1 / bm.get_dt())
# num2 = int(dur2 / bm.get_dt())
# num3 = int(dur3 / bm.get_dt())
# Iext = bm.zeros((num1 + num2 + num3,) + cann.size)
# Iext[:num1] = cann.get_stimulus_by_pos(0.5)
# Iext[num1:num1 + num2] = cann.get_stimulus_by_pos(0.)
# Iext[num1:num1 + num2] += 0.1 * cann.A * bm.random.randn(num2, *cann.size)
#
# runner = bp.dyn.DSRunner(cann,
#                          inputs=('input', Iext, 'iter'),
#                          monitors=['u'],
#                          dyn_vars=cann.vars())
# runner.run(dur1 + dur2 + dur3)
# bp.visualize.animate_1D(
#   dynamical_vars=[{'ys': runner.mon.u, 'xs': cann.x, 'legend': 'u'},
#                   {'ys': Iext, 'xs': cann.x, 'legend': 'Iext'}],
#   frame_step=5,
#   frame_delay=50,
#   show=True,
#   # save_path='../../images/cann-decoding.gif'
# )