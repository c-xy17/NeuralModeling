import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15})
plt.rcParams['font.sans-serif'] = ['Times New Roman']


class HindmarshRose(bp.dyn.NeuGroup):
  def __init__(self, size, a=1., b=3., c=1., d=5., r=0.001, s=4., x_r=-1.6,
               theta=1.0, **kwargs):
    # 初始化父类
    super(HindmarshRose, self).__init__(size=size, **kwargs)

    # 初始化参数
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    self.r = r
    self.s = s
    self.theta = theta
    self.x_r = x_r

    # 初始化变量
    self.x = bm.Variable(bm.random.randn(self.num) + x_r)
    self.y = bm.Variable(bm.ones(self.num) * -10.)
    self.z = bm.Variable(bm.ones(self.num) * 1.7)
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))  # 脉冲发放状态

    # 定义积分器
    self.integral = bp.odeint(f=self.derivative, method='exp_auto')

  def dx(self, x, t, y, z, Iext):
    return y - self.a * x * x * x + self.b * x * x - z + Iext

  def dy(self, y, t, x):
    return self.c - self.d * x * x - y

  def dz(self, z, t, x):
    return self.r * (self.s * (x - self.x_r) - z)

  # 将两个微分方程联合为一个，以便同时积分
  @property
  def derivative(self):
    return bp.JointEq([self.dx, self.dy, self.dz])

  def update(self, tdi):
    x, y, z = self.integral(self.x, self.y, self.z, tdi.t, self.input, tdi.dt)  # 更新变量x, y, z
    self.spike.value = bm.logical_and(x >= self.theta, self.x < self.theta)  # 判断神经元是否发放脉冲
    self.x.value = x
    self.y.value = y
    self.z.value = z
    self.input[:] = 0.  # 重置外界输入


def run_HindmarshRose():
  group = HindmarshRose(10)
  runner = bp.DSRunner(group, monitors=['x', 'y', 'z'], inputs=('input', 2.), dt=0.01)
  runner(20)
  runner(1000)  # 再运行100ms
  plt.figure(figsize=(6, 4))
  bp.visualize.line_plot(runner.mon.ts, runner.mon.x, legend='x', show=False)
  bp.visualize.line_plot(runner.mon.ts, runner.mon.y, legend='y', show=False)
  bp.visualize.line_plot(runner.mon.ts, runner.mon.z, legend='z', show=True)


def bursting_firing():
  duration = 1200
  group = HindmarshRose(1)
  runner = bp.DSRunner(group, monitors=['x', 'y', 'z'], inputs=('input', 2.), dt=0.01)
  runner(duration)  # 再运行100ms

  def visualize(mon, duration, xim, text_pos):
    fig, gs = bp.visualize.get_figure(3, 1, 1.5, 6)
    ax = fig.add_subplot(gs[0, 0])
    plt.plot(mon.ts, mon.x, label='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    plt.text(text_pos, (mon.x.max() + mon.x.min()) / 2, r'$x$')
    ax.set_xlim(*xim)

    ax = fig.add_subplot(gs[1, 0])
    plt.plot(mon.ts, mon.y, label='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    plt.text(text_pos, (mon.y.max() + mon.y.min()) / 2, r'$y$')
    ax.set_xlim(*xim)

    ax = fig.add_subplot(gs[2, 0])
    plt.plot(mon.ts, mon.z, label='z')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.text(text_pos, (mon.z.max() + mon.z.min()) / 2, r'$z$')
    ax.set_xlim(*xim)
    ax.set_xlabel(r'$t$ (ms)')

  visualize(runner.mon, duration=duration, xim=(-1, duration + 50), text_pos=duration)
  plt.savefig('HindmarshRoseModel_output1.pdf', transparent=True, dpi=500)

  from jax.tree_util import tree_map
  visualize(tree_map(lambda a: a[2000:20000], runner.mon), duration=200,
            xim=(19, 200 + 8), text_pos=25)
  plt.savefig('HindmarshRoseModel_output2.pdf', transparent=True, dpi=500)
  plt.show()


def phase_plane_analysis():
  bp.math.enable_x64()

  group = HindmarshRose(1)
  group.x[:] = 1.
  runner = bp.DSRunner(group, monitors=['x', 'y', 'z'], inputs=('input', 2.), dt=0.01)
  runner(80)
  runner(80)  # 再运行100ms
  fig, gs = bp.visualize.get_figure(2, 1, 2.25, 6)
  ax = fig.add_subplot(gs[0, 0])
  ax.set_ylabel(r'$x$')
  ax.set_xlim(80, 160)
  plt.plot(runner.mon.ts, runner.mon.x, color='tab:blue')
  plt.axvline(117.8, linestyle='--', color='gray')
  plt.axvline(135.05, linestyle='--', color='gray')
  plt.axvline(136.07, linestyle='--', color='gray')
  plt.text(118.1, -0.85, '1')
  plt.text(136.57, 1.608, '3')
  plt.text(137.34, -0.85, "1'")
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  ax = fig.add_subplot(gs[1, 0])
  plt.plot(runner.mon.ts, runner.mon.y, color='tab:orange')
  ax.set_ylabel(r'$y$')
  ax.set_xlim(80, 160)
  plt.axvline(117.8, linestyle='--', color='gray')
  plt.axvline(135.05, linestyle='--', color='gray')
  plt.axvline(136.07, linestyle='--', color='gray')
  plt.text(118.1, -4.32, '1')
  plt.text(132.1, 0.22, '2')
  plt.text(136.72, -6.21, '3')
  plt.text(137.34, -4.32, "1'")
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.savefig('HindmarshRose_output123.pdf', transparent=True, dpi=500)

  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
  ax = fig.add_subplot(gs[0, 0])
  # 定义分析器
  model = HindmarshRose(1)
  phase_plane_analyzer = bp.analysis.PhasePlane2D(
    model=model,
    target_vars={'x': [-2, 3], 'y': [-13., 2.]},  # 待分析变量
    fixed_vars={'z': 1.8},  # 固定变量
    pars_update={'Iext': 2.},  # 需要更新的变量
    resolutions=0.01
  )
  # 画出向量场
  phase_plane_analyzer.plot_vector_field(plot_style=dict(color='lightgrey'))
  # 画出V, y的零增长曲线
  phase_plane_analyzer.plot_nullcline()
  # 画出固定点
  phase_plane_analyzer.plot_fixed_point()
  # 画出V, y的变化轨迹
  phase_plane_analyzer.plot_trajectory({'x': [1.], 'y': [0.]}, duration=100.,
                                       color='darkslateblue', linewidth=2, alpha=0.9)
  ax.get_legend().remove()
  plt.text(-1.185, -3.96, '1')
  plt.text(0.135, 0.92, '2')
  plt.text(1.836, -6.48, '3')
  plt.text(1.846, -0.78, 'x nullcline')
  plt.text(1.676, -10.9, 'y nullcline')
  plt.text(-0.805, -6.07, 'Trajectory')
  plt.annotate('unstable focus', xy=(0.6703567413201327, -1.246890802266034),
               xytext=(-0.415, -3), arrowprops=dict(arrowstyle="->"))
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.savefig('HindmarshRose_ppa123.pdf', transparent=True, dpi=500)
  plt.show()


def phase_plane_analysis_v2():
  bp.math.enable_x64()

  def analysis(x_range, y_range, z, extra_f=None, name=None):
    fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
    ax = fig.add_subplot(gs[0, 0])
    # 定义分析器
    model = HindmarshRose(1)
    phase_plane_analyzer = bp.analysis.PhasePlane2D(
      model=model,
      target_vars={'x': x_range, 'y': y_range},  # 待分析变量
      fixed_vars={'z': z},  # 固定变量
      pars_update={'Iext': 2.},  # 需要更新的变量
      resolutions=0.01
    )
    # 画出向量场
    phase_plane_analyzer.plot_vector_field(plot_style=dict(color='lightgrey'))
    # 画出V, y的零增长曲线
    phase_plane_analyzer.plot_nullcline()
    # 画出固定点
    phase_plane_analyzer.plot_fixed_point()
    # 画出V, y的变化轨迹
    phase_plane_analyzer.plot_trajectory({'x': [1.], 'y': [0.]}, duration=100.,
                                         color='darkslateblue', linewidth=2, alpha=0.9)
    ax.get_legend().remove()
    if extra_f: extra_f()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if name:
      plt.savefig(name, transparent=True, dpi=500)

  def f():
    plt.text(1.846, -0.78, 'x nullcline')
    plt.text(1.676, -10.9, 'y nullcline')
    plt.text(-0.805, -6.07, 'Trajectory')
    plt.annotate('unstable focus', xy=(0.6703567413201327, -1.246890802266034),
                 xytext=(-0.415, -3), arrowprops=dict(arrowstyle="->"))

  analysis(x_range=[-2, 3], y_range=[-15., 2.], z=1.8, extra_f=f)
  plt.show()

  def f():
    plt.text(-1.0334, -3.441, 'y nullcline')
    plt.text(-1.05, -4.863, 'x nullcline')
    plt.text(-0.9539, -4.355, 'Trajectory')

  analysis(x_range=[-1.2, -0.8], y_range=[-5., -3.], z=1.8, extra_f=f,
           name='HindmarshRose_I=1.8-v2.pdf')
  plt.show()

  def f():
    plt.text(1.846, -0.78, 'x nullcline')
    plt.text(1.676, -10.9, 'y nullcline')
    plt.text(-0.805, -6.07, 'Trajectory')
    plt.annotate('unstable focus', xy=(0.6040054794020179, -0.8241129286685903),
                 xytext=(-0.415, -3), arrowprops=dict(arrowstyle="->"))
    plt.annotate('saddle node', xy=(-0.9521776294506809, -3.5332112027188076),
                 xytext=(-1.856, -2.07), arrowprops=dict(arrowstyle="->"))
    plt.annotate('stable node', xy=(-1.6518281033863245, -12.642680671164262),
                 xytext=(-1.125, -11.58), arrowprops=dict(arrowstyle="->"))

  analysis(x_range=[-2, 3], y_range=[-15., 2.], z=2.05, extra_f=f,
           name='HindmarshRose_I=2.05-v1.pdf')
  plt.show()

  def f():
    plt.text(-0.9831, -3.209, 'y nullcline')
    plt.text(-1.178, -4.863, 'x nullcline')
    plt.text(-0.9530, -4.095, 'Trajectory')
    plt.annotate('saddle node', xy=(-0.9521957298446758, -3.5333742767571485),
                 xytext=(-0.9179, -3.755), arrowprops=dict(arrowstyle="->"))

  analysis(x_range=[-1.2, -0.8], y_range=[-5., -3.], z=2.05, extra_f=f,
           name='HindmarshRose_I=2.05-v2.pdf')
  plt.show()


def bifurcation_analysis():
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


def bursting_analysis():
  group = HindmarshRose(1)
  group.x[:] = -1.6046805
  group.y[:] = -11.875047
  group.z[:] = -0.01800228
  inputs, duration = bp.inputs.section_input([0., 2., 0.], [40, 10, 200], return_length=True)
  runner = bp.DSRunner(group, monitors=['x', 'y', 'z'], inputs=('input', inputs, 'iter'), dt=0.01)
  runner.run(duration)
  fig, gs = bp.visualize.get_figure(3, 1, 1.5, 6)
  ax = fig.add_subplot(gs[0, 0])
  plt.plot(runner.mon.ts, runner.mon.x, color='tab:blue')
  ax.set_ylabel(r'$x$')
  ax = fig.add_subplot(gs[1, 0])
  plt.plot(runner.mon.ts, runner.mon.y, color='tab:orange')
  ax.set_ylabel(r'$y$')
  ax = fig.add_subplot(gs[2, 0])
  plt.plot(runner.mon.ts, runner.mon.z, color='tab:brown')
  ax.set_ylabel(r'$z$')
  plt.show()


if __name__ == '__main__':
  run_HindmarshRose()
  bursting_firing()
  phase_plane_analysis()
  phase_plane_analysis_v2()
  bifurcation_analysis()
  bursting_analysis()

