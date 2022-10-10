import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 15})
plt.rcParams['font.sans-serif'] = ['Times New Roman']


class AdEx(bp.dyn.NeuGroup):
  def __init__(self, size, V_rest=-65., V_reset=-68., V_th=20., V_T=-60., delta_T=1., a=1.,
               b=2.5, R=1., tau=10., tau_w=30., name=None):
    # 初始化父类
    super(AdEx, self).__init__(size=size, name=name)

    # 初始化参数
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.V_T = V_T
    self.delta_T = delta_T
    self.a = a
    self.b = b
    self.tau = tau
    self.tau_w = tau_w
    self.R = R

    # 初始化变量
    self.V = bm.Variable(bm.random.randn(self.num) + V_reset)
    self.w = bm.Variable(bm.zeros(self.num))
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))  # 脉冲发放状态

    # 定义积分器
    self.integral = bp.odeint(f=self.derivative, method='exp_auto')

  def dV(self, V, t, w, Iext):
    _tmp = self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
    dVdt = (- V + self.V_rest + _tmp - self.R * w + self.R * Iext) / self.tau
    return dVdt

  def dw(self, w, t, V):
    dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
    return dwdt

  # 将两个微分方程联合为一个，以便同时积分
  @property
  def derivative(self):
    return bp.JointEq([self.dV, self.dw])

  def update(self, tdi):
    # 以数组的方式对神经元进行更新
    V, w = self.integral(self.V, self.w, tdi.t, self.input, tdi.dt)  # 更新膜电位V和权重值w
    spike = V > self.V_th  # 将大于阈值的神经元标记为发放了脉冲
    self.spike.value = spike  # 更新神经元脉冲发放状态
    self.V.value = bm.where(spike, self.V_reset, V)  # 将发放了脉冲的神经元膜电位置为V_reset，其余不变
    self.w.value = bm.where(spike, w + self.b, w)  # 发放了脉冲的神经元 w = w + b
    self.input[:] = 0.  # 重置外界输入

  def reset_state(self, batch_size=None):
    self.V[:] = self.V_reset
    self.w[:] = 0.
    self.input[:] = 0.
    self.spike[:] = False


def run_AdEx_model():
  # 运行AdEx模型
  neu = AdEx(2)
  neu.V[:] = bm.asarray([-68.79061, -66.51926])
  runner = bp.dyn.DSRunner(neu, monitors=['V', 'w', 'spike'], inputs=('input', 9.), dt=0.01)
  runner(400)

  # 可视化V和w的变化
  runner.mon.V = np.where(runner.mon.spike, 20., runner.mon.V)

  for i in range(2):
    fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
    ax = fig.add_subplot(gs[0, 0])
    plt.plot(runner.mon.ts, runner.mon.V[:, i], label='V')
    plt.plot(runner.mon.ts, runner.mon.w[:, i], label='w')
    plt.xlabel(r'$t$ (ms)')
    plt.ylabel(r'$V$ (mV)')
    plt.text(-24, 0, r'$w$')
    plt.text(-24, -68, r'$V$')
    plt.xlim(-30, 430)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(f'adex_output{i + 1}.pdf', transparent=True, dpi=500)
  # plt.show()


def AdEx_patterns():
  group = AdEx(
    size=6,
    a=bm.asarray([0., 0., 0.5, -0.5, 1., -1.]),
    b=bm.asarray([60., 5., 7., 7., 10., 5.]),
    tau=bm.asarray([20., 20., 5., 5., 10., 5.]),
    tau_w=bm.asarray([30., 100., 100., 100., 100., 100.]),
    V_reset=bm.asarray([-55., -55., -51., -47., -60., -60.]),
    R=.5, delta_T=2., V_rest=-70, V_th=-30, V_T=-50
  )
  group.V.value = group.V_reset

  par_I = bm.asarray([65., 65., 65., 65., 55., 25.])
  runner = bp.dyn.DSRunner(group, monitors=['V', 'w', 'spike'], inputs=('input', par_I))
  runner.run(500.)

  runner.mon.V = np.where(runner.mon.spike, 20., runner.mon.V)
  names = ['Tonic', 'Adapting', 'Init Bursting', 'Bursting', 'Transient', 'Delayed']
  for i_col in range(2):
    for i_row in range(3):
      i = i_col * 3 + i_row
      fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
      ax = fig.add_subplot(gs[0, 0])
      plt.plot(runner.mon.ts, runner.mon.V[:, i], label='V')
      plt.plot(runner.mon.ts, runner.mon.w[:, i], label='w')
      # plt.title(names[i])
      plt.xlabel(r'$t$ (ms)')
      plt.ylabel(r'$V$ (mV)')
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      plt.savefig(f'adex_pattern_{names[i]}.pdf', transparent=True, dpi=500)
  # plt.show()


def _ppa2d(group, title, v_range=None, w_range=None, Iext=65.,
           duration=400, num_text_sp=0, sp_text=None, extra_fun=None):
  v_range = [-70., -40.] if not v_range else v_range
  w_range = [-10., 50.] if not w_range else w_range
  if sp_text is None:
    sp_text = dict()

  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
  ax = fig.add_subplot(gs[0, 0])
  # 使用BrainPy中的相平面分析工具
  phase_plane_analyzer = bp.analysis.PhasePlane2D(
    model=group,
    target_vars={'V': v_range, 'w': w_range, },  # 待分析变量
    pars_update={'Iext': Iext},  # 需要更新的变量
    resolutions=0.05
  )

  # 画出V, w的零增长曲线
  phase_plane_analyzer.plot_nullcline()
  # 画出奇点
  phase_plane_analyzer.plot_fixed_point()
  # 画出向量场
  phase_plane_analyzer.plot_vector_field(with_return=True)

  # 分段画出V, w的变化轨迹
  group.V[:] = group.V_reset
  runner = bp.DSRunner(group, monitors=['V', 'w', 'spike'], inputs=('input', Iext))
  runner(duration)
  spike = runner.mon.spike.squeeze()
  s_idx = np.where(spike)[0]  # 找到所有发放动作电位对应的index
  s_idx = np.concatenate(([0], s_idx, [len(spike) - 1]))  # 加上起始点和终止点的index
  for i in range(len(s_idx) - 1):
    vs = runner.mon.V[s_idx[i]: s_idx[i + 1]]
    ws = runner.mon.w[s_idx[i]: s_idx[i + 1]]
    plt.plot(vs, ws, color='darkslateblue')
    if i < num_text_sp:
      plt.text(group.V_reset - 1, ws[0] - 0.5, sp_text.get(i, str(i)))
  # 画出虚线 x = V_reset
  plt.plot([group.V_reset, group.V_reset], w_range, '--', color='grey', zorder=-1)

  if extra_fun:
    extra_fun()
  ax.get_legend().remove()
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.savefig(f'adex_phase_plane_{title.replace(" ", "-")}.pdf', transparent=True, dpi=500)
  # plt.show()


def _vt_plot(neu, title, input=('input', 65.), duration=400):
  runner = bp.DSRunner(neu, monitors=['V', 'w', 'spike'], inputs=input)
  runner(duration)

  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
  ax = fig.add_subplot(gs[0, 0])
  runner.mon.V = np.where(runner.mon.spike, 0., runner.mon.V)
  ax.plot(runner.mon.ts, runner.mon.V, label='V')
  ax.plot(runner.mon.ts, runner.mon.w, label='w', color=u'#ff7f0e')
  ax.set_xlabel(r'$t$ (ms)')
  plt.text(-16, runner.mon.V[0, 0], r'$V$')
  plt.text(-16, runner.mon.w[0, 0], r'$w$')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.savefig(f'adex_pp_pattern_{title.replace(" ", "-")}.pdf', transparent=True, dpi=500)
  # plt.show()


def AdEx_phase_plane_analysis():
  bp.math.enable_x64()

  # Tonic Spiking
  model = AdEx(1, tau=20., a=0., tau_w=30., b=60., V_reset=-55.,
               V_rest=-70., V_th=0., V_T=-50., R=0.5, delta_T=2.)
  def f():
    plt.plot(np.linspace(-70, -40, 500), np.zeros(500), '.', color='lightcoral', alpha=.7)
    plt.text(-54.5, 0, '0')
    plt.text(-54.5, 60, '1')
    plt.text(-54.5, 69.5, '2, 3, ...')
    plt.text(-68., 62.8, 'V nullcline')
    plt.text(-68., 2, 'w nullcline')
  _ppa2d(model, title='Tonic Spiking', w_range=[-5, 75.], extra_fun=f)
  model.reset_state()
  _vt_plot(model, title='Tonic Spiking')

  # Adaptation
  model = AdEx(1, tau=20., a=0., tau_w=100., b=5., V_reset=-55.,
               V_rest=-70., V_th=0., V_T=-50., R=0.5, delta_T=2.)

  def f():
    plt.plot(np.linspace(-70, -40, 500), np.zeros(500), '.', color='lightcoral', alpha=.7)
    plt.text(-47.7, 30.5, 'V nullcline')
    plt.text(-68., 2., 'w nullcline')
    plt.text(-50.7, 19., 'Trajectory')

  _ppa2d(model, title='Adaptation', w_range=[-5, 45.], extra_fun=f, num_text_sp=5)
  model.reset_state()
  _vt_plot(model, title='Adaptation')

  # Initial Bursting
  model = AdEx(1, tau=5., a=0.5, tau_w=100., b=7., V_reset=-51.,
               V_rest=-70., V_th=0., V_T=-50., R=0.5, delta_T=2.)
  def f():
    plt.text(-59, 45.3, 'V nullcline')
    plt.text(-68., 7., 'w nullcline')
    plt.text(-48.1, 27.3, 'Trajectory')
  _ppa2d(model, title='Initial Bursting', w_range=[-5, 50.], extra_fun=f, num_text_sp=6)
  model.reset_state()
  _vt_plot(model, title='Initial Bursting')

  # Bursting
  model = AdEx(1, tau=5., a=-0.5, tau_w=100., b=7., V_reset=-47.,
               V_rest=-70., V_th=0., V_T=-50., R=0.5, delta_T=2.)
  def f():
    plt.text(-48, 29.5, '1')
    plt.text(-48, 35.5, '2')
    plt.text(-46.8, 38.8, '0')
    plt.text(-46.8, 42, '3')
    plt.text(-60.8, 48.83, 'V nullcline')
    plt.text(-68., 1.2, 'w nullcline')
    plt.text(-45.88, 36.8, 'Trajectory')
  _ppa2d(model, title='Bursting', w_range=[-5, 60.], extra_fun=f)
  model.reset_state()
  _vt_plot(model, title='Bursting')

  # Transient Spiking
  model = AdEx(1, tau=10., a=1., tau_w=100., b=10., V_reset=-60.,
               V_rest=-70., V_th=0., V_T=-50., R=0.5, delta_T=2.)
  def f():
    plt.text(-65., 47.5, 'V nullcline')
    plt.text(-69., 8., 'w nullcline')
    plt.text(-58.65, 2., 'Trajectory')
    plt.annotate('stable focus', xy=(-50.750613238954806, 19.25038786580243),
                 xytext=(-54., 28.44), arrowprops=dict(arrowstyle="->"))
    plt.annotate('saddle node', xy=(-47.9494123367877, 22.050461841990344),
                 xytext=(-46.4, 16.5), arrowprops=dict(arrowstyle="->"))
  _ppa2d(model, title='Transient Spiking', w_range=[-5, 60.], Iext=55., extra_fun=f, num_text_sp=3)
  model.reset_state()
  _vt_plot(model, title='Transient Spiking', input=('input', 55.))

  # Delayed Spiking
  model = AdEx(1, tau=5., a=-1., tau_w=100., b=5., V_reset=-60.,
               V_rest=-70., V_th=0., V_T=-50., R=0.5, delta_T=2.)
  def f():
    plt.text(-62., 11., 'V nullcline')
    plt.text(-67., -12.6, 'w nullcline')
    plt.text(-48.9, -16.3, 'Trajectory')
  _ppa2d(model, title='Delayed Spiking', w_range=[-30, 20.], Iext=25., extra_fun=f,
         num_text_sp=2, sp_text={1: '1,2,..'})
  model.reset_state()
  _vt_plot(model, title='Delayed Spiking', input=('input', 25.))

  plt.show()


if __name__ == '__main__':
  run_AdEx_model()
  AdEx_patterns()
  AdEx_phase_plane_analysis()
