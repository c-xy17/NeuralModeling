import brainpy as bp
from brainpy import synapses, synouts
import brainpy.math as bm
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15})
plt.rcParams['font.sans-serif'] = ['Times New Roman']

# 定义各个阶段的时长
pre_stimulus_period = 100.
stimulus_period = 1000.
delay_period = 500.
total_period = pre_stimulus_period + stimulus_period + delay_period


# 产生随机泊松刺激的神经元群（用于生成I_A和I_B）
class PoissonStim(bp.dyn.NeuGroup):
  def __init__(self, size, freq_mean, freq_var, t_interval, **kwargs):
    super(PoissonStim, self).__init__(size=size, **kwargs)

    # 初始化参数
    self.freq_mean = freq_mean
    self.freq_var = freq_var
    self.t_interval = t_interval
    self.dt = bm.get_dt() / 1000.

    # 初始化变量
    self.freq = bm.Variable(bm.zeros(1))
    self.freq_t_last_change = bm.Variable(bm.ones(1) * -1e7)
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.rng = bm.random.RandomState()  # 随机数生成器

  def update(self, tdi):
    _t, _dt = tdi.t, tdi.dt
    # 下两行代码相当于：
    # if pre_stimulus_period < _t < pre_stimulus_period + stimulus_period:
    #   freq = self.freq[0]
    # else:
    #   freq = 0.
    in_interval = bm.logical_and(pre_stimulus_period < _t, _t < pre_stimulus_period + stimulus_period)
    freq = bm.where(in_interval, self.freq[0], 0.)

    # 判断是否需要改变freq的值
    change = bm.logical_and(in_interval, (_t - self.freq_t_last_change[0]) >= self.t_interval)
    # 更新freq, freq_t_last_change
    self.freq[:] = bm.where(change, self.rng.normal(self.freq_mean, self.freq_var), freq)
    self.freq_t_last_change[:] = bm.where(change, _t, self.freq_t_last_change[0])
    # 按照p=freq*dt的概率生成脉冲
    self.spike.value = self.rng.random(self.num) < self.freq[0] * self.dt


class DecisionMaking(bp.dyn.Network):
  def __init__(self, scale=1., mu0=40., coherence=25.6, f=0.15, mode=bp.modes.NormalMode()):
    super(DecisionMaking, self).__init__()

    num_exc = int(1600 * scale)
    num_inh = int(400 * scale)
    num_A = int(f * num_exc)
    num_B = int(f * num_exc)
    num_N = num_exc - num_A - num_B
    print(f'Total network size: {num_exc + num_inh}')

    poisson_freq = 2400.  # Hz
    w_pos = 1.7
    w_neg = 1. - f * (w_pos - 1.) / (1. - f)
    g_ext2E_AMPA = 2.1  # nS
    g_ext2I_AMPA = 1.62  # nS
    g_E2E_AMPA = 0.05 / scale  # nS
    g_E2I_AMPA = 0.04 / scale  # nS
    g_E2E_NMDA = 0.165 / scale  # nS
    g_E2I_NMDA = 0.13 / scale  # nS
    g_I2E_GABAa = 1.3 / scale  # nS
    g_I2I_GABAa = 1.0 / scale  # nS

    ampa_par = dict(delay_step=int(0.5 / bm.get_dt()), tau=2.0)
    gaba_par = dict(delay_step=int(0.5 / bm.get_dt()), tau=5.0)
    nmda_par = dict(delay_step=int(0.5 / bm.get_dt()), tau_decay=100, tau_rise=2., a=0.5)

    # E neurons/pyramid neurons
    A = bp.neurons.LIF(num_A, V_rest=-70., V_reset=-55., V_th=-50., tau=20., R=0.04,
                       tau_ref=2., V_initializer=bp.init.OneInit(-70.), mode=mode)
    B = bp.neurons.LIF(num_B, V_rest=-70., V_reset=-55., V_th=-50., tau=20., R=0.04,
                       tau_ref=2., V_initializer=bp.init.OneInit(-70.), mode=mode)
    N = bp.neurons.LIF(num_N, V_rest=-70., V_reset=-55., V_th=-50., tau=20., R=0.04,
                       tau_ref=2., V_initializer=bp.init.OneInit(-70.), mode=mode)
    # I neurons/interneurons
    I = bp.neurons.LIF(num_inh, V_rest=-70., V_reset=-55., V_th=-50., tau=10., R=0.05,
                       tau_ref=1., V_initializer=bp.init.OneInit(-70.), mode=mode)

    # poisson stimulus
    IA = PoissonStim(num_A, freq_var=10., t_interval=50., freq_mean=mu0 + mu0 / 100. * coherence, mode=mode)
    IB = PoissonStim(num_B, freq_var=10., t_interval=50., freq_mean=mu0 - mu0 / 100. * coherence, mode=mode)

    # noise neurons
    self.noise_B = bp.neurons.PoissonGroup(num_B, freqs=poisson_freq, mode=mode)
    self.noise_A = bp.neurons.PoissonGroup(num_A, freqs=poisson_freq, mode=mode)
    self.noise_N = bp.neurons.PoissonGroup(num_N, freqs=poisson_freq, mode=mode)
    self.noise_I = bp.neurons.PoissonGroup(num_inh, freqs=poisson_freq, mode=mode)

    # define external inputs
    self.IA2A = synapses.Exponential(IA, A, bp.conn.One2One(), g_max=g_ext2E_AMPA,
                                     mode=mode, output=synouts.COBA(E=0.), **ampa_par)
    self.IB2B = synapses.Exponential(IB, B, bp.conn.One2One(), g_max=g_ext2E_AMPA,
                                     mode=mode, output=synouts.COBA(E=0.), **ampa_par)

    # define E->E/I conn

    self.N2B_AMPA = synapses.Exponential(N, B, bp.conn.All2All(), g_max=g_E2E_AMPA * w_neg,
                                         output=synouts.COBA(E=0.), mode=mode, **ampa_par)
    self.N2A_AMPA = synapses.Exponential(N, A, bp.conn.All2All(), g_max=g_E2E_AMPA * w_neg,
                                         output=synouts.COBA(E=0.), mode=mode, **ampa_par)
    self.N2N_AMPA = synapses.Exponential(N, N, bp.conn.All2All(), g_max=g_E2E_AMPA,
                                         output=synouts.COBA(E=0.), mode=mode, **ampa_par)
    self.N2I_AMPA = synapses.Exponential(N, I, bp.conn.All2All(), g_max=g_E2I_AMPA,
                                         output=synouts.COBA(E=0.), mode=mode, **ampa_par)
    self.N2B_NMDA = synapses.NMDA(N, B, bp.conn.All2All(), g_max=g_E2E_NMDA * w_neg,
                                  output=synouts.MgBlock(E=0., cc_Mg=1.), mode=mode, **nmda_par)
    self.N2A_NMDA = synapses.NMDA(N, A, bp.conn.All2All(), g_max=g_E2E_NMDA * w_neg,
                                  output=synouts.MgBlock(E=0., cc_Mg=1.), mode=mode, **nmda_par)
    self.N2N_NMDA = synapses.NMDA(N, N, bp.conn.All2All(), g_max=g_E2E_NMDA,
                                  output=synouts.MgBlock(E=0., cc_Mg=1.), mode=mode, **nmda_par)
    self.N2I_NMDA = synapses.NMDA(N, I, bp.conn.All2All(), g_max=g_E2I_NMDA,
                                  output=synouts.MgBlock(E=0., cc_Mg=1.), mode=mode, **nmda_par)

    self.B2B_AMPA = synapses.Exponential(B, B, bp.conn.All2All(), g_max=g_E2E_AMPA * w_pos,
                                         output=synouts.COBA(E=0.), mode=mode, **ampa_par)
    self.B2A_AMPA = synapses.Exponential(B, A, bp.conn.All2All(), g_max=g_E2E_AMPA * w_neg,
                                         output=synouts.COBA(E=0.), mode=mode, **ampa_par)
    self.B2N_AMPA = synapses.Exponential(B, N, bp.conn.All2All(), g_max=g_E2E_AMPA,
                                         output=synouts.COBA(E=0.), mode=mode, **ampa_par)
    self.B2I_AMPA = synapses.Exponential(B, I, bp.conn.All2All(), g_max=g_E2I_AMPA,
                                         output=synouts.COBA(E=0.), mode=mode, **ampa_par)
    self.B2B_NMDA = synapses.NMDA(B, B, bp.conn.All2All(), g_max=g_E2E_NMDA * w_pos,
                                  output=synouts.MgBlock(E=0., cc_Mg=1.), mode=mode, **nmda_par)
    self.B2A_NMDA = synapses.NMDA(B, A, bp.conn.All2All(), g_max=g_E2E_NMDA * w_neg,
                                  output=synouts.MgBlock(E=0., cc_Mg=1.), mode=mode, **nmda_par)
    self.B2N_NMDA = synapses.NMDA(B, N, bp.conn.All2All(), g_max=g_E2E_NMDA,
                                  output=synouts.MgBlock(E=0., cc_Mg=1.), mode=mode, **nmda_par)
    self.B2I_NMDA = synapses.NMDA(B, I, bp.conn.All2All(), g_max=g_E2I_NMDA,
                                  output=synouts.MgBlock(E=0., cc_Mg=1.), mode=mode, **nmda_par)

    self.A2B_AMPA = synapses.Exponential(A, B, bp.conn.All2All(), g_max=g_E2E_AMPA * w_neg,
                                         output=synouts.COBA(E=0.), mode=mode, **ampa_par)
    self.A2A_AMPA = synapses.Exponential(A, A, bp.conn.All2All(), g_max=g_E2E_AMPA * w_pos,
                                         output=synouts.COBA(E=0.), mode=mode, **ampa_par)
    self.A2N_AMPA = synapses.Exponential(A, N, bp.conn.All2All(), g_max=g_E2E_AMPA,
                                         output=synouts.COBA(E=0.), mode=mode, **ampa_par)
    self.A2I_AMPA = synapses.Exponential(A, I, bp.conn.All2All(), g_max=g_E2I_AMPA,
                                         output=synouts.COBA(E=0.), mode=mode, **ampa_par)
    self.A2B_NMDA = synapses.NMDA(A, B, bp.conn.All2All(), g_max=g_E2E_NMDA * w_neg,
                                  output=synouts.MgBlock(E=0., cc_Mg=1.), mode=mode, **nmda_par)
    self.A2A_NMDA = synapses.NMDA(A, A, bp.conn.All2All(), g_max=g_E2E_NMDA * w_pos,
                                  output=synouts.MgBlock(E=0., cc_Mg=1.), mode=mode, **nmda_par)
    self.A2N_NMDA = synapses.NMDA(A, N, bp.conn.All2All(), g_max=g_E2E_NMDA,
                                  output=synouts.MgBlock(E=0., cc_Mg=1.), mode=mode, **nmda_par)
    self.A2I_NMDA = synapses.NMDA(A, I, bp.conn.All2All(), g_max=g_E2I_NMDA,
                                  output=synouts.MgBlock(E=0., cc_Mg=1.), mode=mode, **nmda_par)

    # define I->E/I conn
    self.I2B = synapses.Exponential(I, B, bp.conn.All2All(), g_max=g_I2E_GABAa,
                                    output=synouts.COBA(E=-70.), mode=mode, **gaba_par)
    self.I2A = synapses.Exponential(I, A, bp.conn.All2All(), g_max=g_I2E_GABAa,
                                    output=synouts.COBA(E=-70.), mode=mode, **gaba_par)
    self.I2N = synapses.Exponential(I, N, bp.conn.All2All(), g_max=g_I2E_GABAa,
                                    output=synouts.COBA(E=-70.), mode=mode, **gaba_par)
    self.I2I = synapses.Exponential(I, I, bp.conn.All2All(), g_max=g_I2I_GABAa,
                                    output=synouts.COBA(E=-70.), mode=mode, **gaba_par)

    # define external projections
    self.noise2B = synapses.Exponential(self.noise_B, B, bp.conn.One2One(), g_max=g_ext2E_AMPA,
                                        output=synouts.COBA(E=0.), mode=mode, **ampa_par)
    self.noise2A = synapses.Exponential(self.noise_A, A, bp.conn.One2One(), g_max=g_ext2E_AMPA,
                                        output=synouts.COBA(E=0.), mode=mode, **ampa_par)
    self.noise2N = synapses.Exponential(self.noise_N, N, bp.conn.One2One(), g_max=g_ext2E_AMPA,
                                        output=synouts.COBA(E=0.), mode=mode, **ampa_par)
    self.noise2I = synapses.Exponential(self.noise_I, I, bp.conn.One2One(), g_max=g_ext2I_AMPA,
                                        output=synouts.COBA(E=0.), mode=mode, **ampa_par)

    # nodes
    self.B = B
    self.A = A
    self.N = N
    self.I = I
    self.IA = IA
    self.IB = IB


def run_model_coherence1():
  # 数值模拟
  bm.random.seed(1234)
  coherence = 25.6
  net = DecisionMaking(scale=1., coherence=coherence, mu0=40.)
  runner = bp.dyn.DSRunner(net, monitors=['A.spike', 'B.spike', 'IA.freq', 'IB.freq'])
  runner.run(total_period)

  # 可视化
  fig, gs = bp.visualize.get_figure(4, 1, 3, 10)

  # 神经元群A的脉冲发放图
  ax1 = fig.add_subplot(gs[0])
  bp.visualize.raster_plot(runner.mon.ts, runner.mon['A.spike'], markersize=1)
  plt.title("Spiking activity of group A")
  plt.ylabel("Neuron Index")

  # 神经元群B的脉冲发放图
  ax2 = fig.add_subplot(gs[1])
  bp.visualize.raster_plot(runner.mon.ts, runner.mon['B.spike'], markersize=1)
  plt.title("Spiking activity of group B")
  plt.ylabel("Neuron Index")

  # 神经元群A、B的发放率图
  ax3 = fig.add_subplot(gs[2])
  rateA = bp.measure.firing_rate(runner.mon['A.spike'], width=10.)
  rateB = bp.measure.firing_rate(runner.mon['B.spike'], width=10.)
  plt.plot(runner.mon.ts, rateA, label="Group A")
  plt.plot(runner.mon.ts, rateB, label="Group B")
  plt.ylabel('Firing rate [Hz]')
  plt.title("Population activity")
  plt.text(350, 40, 'Group A')
  plt.text(600, 8, 'Group B')

  # 神经元群A、B接收到的刺激频率图
  ax4 = fig.add_subplot(gs[3])
  plt.plot(runner.mon.ts, runner.mon['IA.freq'], label="group A")
  plt.plot(runner.mon.ts, runner.mon['IB.freq'], label="group B")
  plt.title("Input activity")
  plt.ylabel("Firing rate [Hz]")
  plt.xlabel("Time [ms]")
  plt.text(700, 60, 'Group A')
  plt.text(400, 10, 'Group B')


  for ax in (ax1, ax2, ax3, ax4):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(pre_stimulus_period, linestyle='dashed', color=u'#444444')
    ax.axvline(pre_stimulus_period + stimulus_period, linestyle='dashed', color=u'#444444')
    ax.set_xlim(-1, total_period + 1)

  plt.savefig('decision_making_output_c={}.png'.format(coherence), transparent=True, dpi=500)
  plt.show()


def run_model_coherence2():
  # 数值模拟
  bm.random.seed(1234)
  coherence = -6.4
  net = DecisionMaking(scale=1., coherence=coherence, mu0=40.)
  runner = bp.dyn.DSRunner(net, monitors=['A.spike', 'B.spike', 'IA.freq', 'IB.freq'])
  runner.run(total_period)

  # 可视化
  fig, gs = bp.visualize.get_figure(4, 1, 3, 10)

  # 神经元群A的脉冲发放图
  ax1 = fig.add_subplot(gs[0])
  bp.visualize.raster_plot(runner.mon.ts, runner.mon['A.spike'], markersize=1)
  plt.title("Spiking activity of group A")
  plt.ylabel("Neuron Index")

  # 神经元群B的脉冲发放图
  ax2 = fig.add_subplot(gs[1])
  bp.visualize.raster_plot(runner.mon.ts, runner.mon['B.spike'], markersize=1)
  plt.title("Spiking activity of group B")
  plt.ylabel("Neuron Index")

  # 神经元群A、B的发放率图
  ax3 = fig.add_subplot(gs[2])
  rateA = bp.measure.firing_rate(runner.mon['A.spike'], width=10.)
  rateB = bp.measure.firing_rate(runner.mon['B.spike'], width=10.)
  plt.plot(runner.mon.ts, rateA, label="Group A")
  plt.plot(runner.mon.ts, rateB, label="Group B")
  plt.ylabel('Firing rate [Hz]')
  plt.title("Population activity")
  plt.text(350, 40, 'Group B')
  plt.text(600, 8, 'Group A')

  # 神经元群A、B接收到的刺激频率图
  ax4 = fig.add_subplot(gs[3])
  plt.plot(runner.mon.ts, runner.mon['IA.freq'], label="group A")
  plt.plot(runner.mon.ts, runner.mon['IB.freq'], label="group B")
  plt.title("Input activity")
  plt.ylabel("Firing rate [Hz]")
  plt.xlabel("Time [ms]")
  plt.text(700, 55, 'Group B')
  plt.text(800, 15, 'Group A')

  for ax in (ax1, ax2, ax3, ax4):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(pre_stimulus_period, linestyle='dashed', color=u'#444444')
    ax.axvline(pre_stimulus_period + stimulus_period, linestyle='dashed', color=u'#444444')
    ax.set_xlim(-1, total_period + 1)

  plt.savefig('decision_making_output_c={}.png'.format(coherence), transparent=True, dpi=500)
  plt.show()


if __name__ == '__main__':
  run_model_coherence1()
  run_model_coherence2()
