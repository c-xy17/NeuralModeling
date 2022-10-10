import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 15})
plt.rcParams['font.sans-serif'] = ['Times New Roman']


class Izhikevich(bp.dyn.NeuGroup):
  def __init__(self, size, a=0.02, b=0.20, c=-65., d=2., tau_ref=0.,
               V_th=30., **kwargs):
    # 初始化父类
    super(Izhikevich, self).__init__(size=size, **kwargs)

    # 初始化参数
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    self.V_th = V_th
    self.tau_ref = tau_ref

    # 初始化变量
    self.V = bm.Variable(bm.random.randn(self.num) - 65.)
    self.u = bm.Variable(self.V * b)
    self.input = bm.Variable(bm.zeros(self.num))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)  # 上一次脉冲发放时间
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))  # 是否处于不应期
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))  # 脉冲发放状态

    # 定义积分器
    self.integral = bp.odeint(f=self.derivative, method='exp_auto')

  def dV(self, V, t, u, Iext):
    return 0.04 * V * V + 5 * V + 140 - u + Iext

  def du(self, u, t, V):
    return self.a * (self.b * V - u)

  # 将两个微分方程联合为一个，以便同时积分
  @property
  def derivative(self):
    return bp.JointEq([self.dV, self.du])

  def update(self, tdi):
    _t, _dt = tdi.t, tdi.dt
    V, u = self.integral(self.V, self.u, _t, self.input, dt=_dt)  # 更新变量V, u
    refractory = (_t - self.t_last_spike) <= self.tau_ref  # 判断神经元是否处于不应期
    V = bm.where(refractory, self.V, V)  # 若处于不应期，则返回原始膜电位self.V，否则返回更新后的膜电位V
    u = bm.where(refractory, self.u, u)  # u同理
    spike = V > self.V_th  # 将大于阈值的神经元标记为发放了脉冲
    self.spike.value = spike  # 更新神经元脉冲发放状态
    self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)  # 更新最后一次脉冲发放时间
    self.V.value = bm.where(spike, self.c, V)  # 将发放了脉冲的神经元的V置为c，其余不变
    self.u.value = bm.where(spike, u + self.d, u)  # 将发放了脉冲的神经元的u增加d，其余不变
    self.refractory.value = bm.logical_or(refractory, spike)  # 更新神经元是否处于不应期
    self.input[:] = 0.  # 重置外界输入

  def reset_state(self, batch_size=None):
    self.V.value = bm.ones_like(self.V) * self.c
    self.u.value = bm.ones_like(self.u) * self.V * self.b
    self.input.value = bm.zeros_like(self.input)
    self.t_last_spike[:] = -1e7
    self.refractory.value = bm.zeros_like(self.refractory)
    self.spike.value = bm.zeros_like(self.spike)


def run_Izhkevich():
  # 运行Izhikevich模型
  group = Izhikevich(10)
  runner = bp.DSRunner(group, monitors=['V', 'u'], inputs=('input', 10.))
  runner(300)
  bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V', show=False)
  bp.visualize.line_plot(runner.mon.ts, runner.mon.u, legend='u', show=True)


def Izhkevich_patterns():
  def subplot(izhi, title, input=('input', 10.), duration=200):
    fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
    ax = fig.add_subplot(gs[0, 0])
    runner = bp.DSRunner(izhi, monitors=['V', 'u'], inputs=input)
    runner(duration)
    plt.plot(runner.mon.ts, runner.mon.V)
    plt.plot(runner.mon.ts, runner.mon.u)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.text(duration + 5, runner.mon.V[-1, 0], r'$V$')
    plt.text(duration + 5, runner.mon.u[-1, 0], r'$u$')
    plt.xlim(-1, duration + 10)
    plt.xlabel(r'$t$ (ms)')
    plt.savefig(f'Izhkevich_pattern_{title.replace(" ", "-")}.pdf', transparent=True, dpi=500)

  subplot(Izhikevich(1, d=8.), title='Regular Spiking')
  subplot(Izhikevich(1, c=-55., d=4.), title='Intrinsic Bursting')
  subplot(Izhikevich(1, a=0.1), title='Fast Spiking')
  subplot(Izhikevich(1, c=-50.), title='Chattering (Bursting)')
  input5 = bp.inputs.section_input(values=[-30, 3.5], durations=[50, 150])
  subplot(Izhikevich(1, b=0.2), title='Rebound Bursting', input=('input', input5, 'iter'))
  subplot(Izhikevich(1, b=0.25), title='Low Threshold Spiking')
  plt.show()


def bifurcation_analysis():
  bp.math.enable_x64()

  # 定义分析器
  plt.figure('V', (6, 4.5), )
  plt.figure('u', (6, 4.5), )
  bif = bp.analysis.Bifurcation2D(
    model=Izhikevich(1),
    target_vars={'V': [-75., -45.], 'u': [-17., -7.]},  # 设置变量的分析范围
    target_pars={'Iext': [0., 5.]},  # 设置参数的范围
    resolutions={'Iext': bm.concatenate([bm.arange(0, 3.95, 0.01),
                                         bm.arange(3.95, 4.05, 0.001),
                                         bm.arange(4.05, 5.0, 0.1)])}  # 设置分辨率
  )
  # 进行分析
  bif.plot_bifurcation()
  plt.figure('V')
  plt.tight_layout()
  ax = plt.gca()
  ax.get_legend().remove()
  plt.savefig('Izhikevich_bif_V.png', dpi=500, transparent=True)

  plt.figure('u')
  plt.tight_layout()
  ax = plt.gca()
  ax.get_legend().remove()
  plt.savefig('Izhikevich_bif_u.png', dpi=500, transparent=True)
  plt.show()


def _ppa2d(model, v_range, u_range, Iext=10., duration=200, extra_fun=None):
  model.reset_state()
  runner = bp.DSRunner(model, monitors=['V', 'u'], inputs=('input', Iext))
  runner.run(duration)
  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
  ax = fig.add_subplot(gs[0, 0])
  ax.plot(runner.mon.ts, runner.mon.V)
  ax.plot(runner.mon.ts, runner.mon.u)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.text(duration + 5, runner.mon.V[-1, 0], r'$V$')
  plt.text(duration + 5, runner.mon.u[-1, 0], r'$u$')
  plt.xlim(-1, duration + 15)
  plt.savefig(f'Izhkevich_pattern_I={Iext}.pdf', transparent=True, dpi=500)

  model.reset_state()
  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
  ax = fig.add_subplot(gs[0, 0])
  # 使用BrainPy中的相平面分析工具
  phase_plane_analyzer = bp.analysis.PhasePlane2D(
    model=model,
    target_vars={'V': v_range, 'u': u_range},  # 待分析变量
    pars_update={'Iext': Iext},  # 需要更新的变量
    resolutions=0.01
  )
  # 画出向量场
  phase_plane_analyzer.plot_vector_field(plot_style=dict(color='lightgrey', density=1.))
  # 画出V, w的零增长曲线
  phase_plane_analyzer.plot_nullcline()
  # 画出固定点
  phase_plane_analyzer.plot_fixed_point(tol_unique=1e-1)
  # 分段画出V, w的变化轨迹
  runner = bp.DSRunner(model, monitors=['V', 'u', 'spike'], inputs=('input', Iext))
  runner(duration)
  spike = runner.mon.spike.squeeze()
  s_idx = np.where(spike)[0]  # 找到所有发放动作电位对应的index
  s_idx = np.concatenate(([0], s_idx, [len(spike) - 1]))  # 加上起始点和终止点的index
  for i in range(len(s_idx) - 1):
    plt.plot(runner.mon.V[s_idx[i]: s_idx[i + 1]],
             runner.mon.u[s_idx[i]: s_idx[i + 1]],
             color='darkslateblue')
  # 画出虚线 x = V_reset
  plt.plot([model.c, model.c], u_range, '--', color='grey', zorder=-1)
  plt.xlim(v_range)
  plt.ylim(u_range)
  ax.get_legend().remove()
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  if extra_fun:    extra_fun()
  plt.savefig(f'Izhkevich_phase_plane_I={Iext}.pdf', transparent=True, dpi=500)
  plt.show()


def phase_plane_analysis():
  bp.math.enable_x64()

  def f():
    plt.text(-76.5, -3.7, 'V nullcline')
    plt.text(-76.5, -16.3, 'u nullcline')
    plt.text(-60.8, -14.5, 'Trajectory')
    plt.annotate('stable focus',
                 xy=(-62.7388249681836, -12.547720273309872),
                 xytext=(-67.5, -10),
                 arrowprops=dict(arrowstyle="->"))
    plt.annotate('saddle node',
                 xy=(-57.27984674554154, -11.459999999998665),
                 xytext=(-60, -7.3),
                 arrowprops=dict(arrowstyle="->"))

  _ppa2d(Izhikevich(1, c=-68), [-80., -45.], [-20., 0.], Iext=3.7, duration=400, extra_fun=f)

  def f():
    plt.text(-76.5, -3.7, 'V nullcline')
    plt.text(-76.5, -16.3, 'u nullcline')
    plt.text(-60.8, -14.5, 'Trajectory')
    plt.annotate('stable focus',
                 xy=(-61.5826134542462, -12.316334160065736),
                 xytext=(-67.5, -10),
                 arrowprops=dict(arrowstyle="->"))
    plt.annotate('saddle node',
                 xy=(-58.437980797678, -11.689999999998701),
                 xytext=(-60, -7.3),
                 arrowprops=dict(arrowstyle="->"))

  _ppa2d(Izhikevich(1, c=-68), [-80., -45.], [-20., 0.], Iext=3.9, duration=400, extra_fun=f)


if __name__ == '__main__':
  # run_Izhkevich()
  # Izhkevich_patterns()
  bifurcation_analysis()
  # phase_plane_analysis()
