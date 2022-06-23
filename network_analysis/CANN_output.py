import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

from network_models.CANN import CANN1D

# 生成CANN
cann = CANN1D(num=512, k=0.1)

# 生成外部刺激，从第2到12ms，持续10ms
dur1, dur2, dur3 = 2., 10., 10.
I1 = cann.get_stimulus_by_pos(0.)
Iext, duration = bp.inputs.section_input(values=[0., I1, 0.],
                                         durations=[dur1, dur2, dur3],
                                         return_length=True)
# 运行数值模拟
runner = bp.dyn.DSRunner(cann,
                         inputs=['input', Iext, 'iter'],
                         monitors=['u'],
                         dyn_vars=cann.vars())
runner.run(duration)

# 可视化
def plot_response(ax, t):
  ts = int(t / bm.get_dt())
  I, u = Iext[ts], runner.mon.u[ts]
  ax.plot(cann.x, I, label='Iext')
  ax.plot(cann.x, u, label='U')
  ax.set_title('t = {} ms'.format(t))
  ax.set_xlabel('x')
  ax.legend()

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

