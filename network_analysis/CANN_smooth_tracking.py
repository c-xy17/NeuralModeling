import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

from network_models.CANN import CANN1D

def plot_response(ax, t):
  ts = int(t / bm.get_dt())
  I, u = Iext[ts], runner.mon.u[ts]
  ax.plot(cann.x, I, label='Iext')
  ax.plot(cann.x, u, label='U')
  ax.set_title('t = {} ms'.format(t))
  ax.set_xlabel('x')
  ax.legend()


cann = CANN1D(num=512, k=8.1)

# 定义随时间变化的外部刺激
dur1, dur2, dur3 = 10., 10., 20
num1 = int(dur1 / bm.get_dt())
num2 = int(dur2 / bm.get_dt())
num3 = int(dur3 / bm.get_dt())
position = bm.zeros(num1 + num2 + num3)
position[num1: num1 + num2] = bm.linspace(0., 1.5 * bm.pi, num2)
position[num1 + num2:] = 1.5 * bm.pi
position = position.reshape((-1, 1))
Iext = cann.get_stimulus_by_pos(position)

# 运行模拟
runner = bp.dyn.DSRunner(cann,
                         inputs=['input', Iext, 'iter'],
                         monitors=['u'],
                         dyn_vars=cann.vars())
runner.run(dur1 + dur2 + dur3)

# 可视化
fig, gs = plt.subplots(2, 2, figsize=(12, 9), sharey='all')

plot_response(gs[0, 0], t=10.)
plot_response(gs[0, 1], t=15.)
plot_response(gs[1, 0], t=20.)
plot_response(gs[1, 1], t=30.)

plt.tight_layout()
plt.show()

plt.savefig('E:\\2021-2022RA\\神经计算建模实战\\NeuralModeling\\'
            'images_network_models\\CANN_smooth_tracking.png')

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
#   # save_path='../../images/cann-tracking.gif'
# )