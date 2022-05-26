import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

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

  def update(self, _t, _dt):
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
  def __init__(self, scale=1., mu0=40., coherence=25.6, dt=0.1):
    super(DecisionMaking, self).__init__()

    # 初始神经元化参数
    f = 0.15
    num_exc = int(1600 * scale)
    num_inh = int(400 * scale)
    num_A = int(f * num_exc)
    num_B = int(f * num_exc)
    num_N = num_exc - num_A - num_B
    poisson_freq = 2400.  # Hz

    # 初始化突触参数
    w_pos = 1.7
    w_neg = 1. - f * (w_pos - 1.) / (1. - f)
    g_max_ext2E_AMPA = 2.1 * 1e-3  # uS
    g_max_ext2I_AMPA = 1.62 * 1e-3  # uS
    g_max_E2E_AMPA = 0.05 * 1e-3 / scale  # uS
    g_max_E2E_NMDA = 0.165 * 1e-3 / scale  # uS
    g_max_E2I_AMPA = 0.04 * 1e-3 / scale  # uS
    g_max_E2I_NMDA = 0.13 * 1e-3 / scale  # uS
    g_max_I2E_GABAa = 1.3 * 1e-3 / scale  # uS
    g_max_I2I_GABAa = 1.0 * 1e-3 / scale  # uS

    ampa_par = dict(delay_step=int(0.5 / dt), E=0., tau=2.0)  # AMP受体的参数
    gaba_par = dict(delay_step=int(0.5 / dt), E=-70., tau=5.0)  # GABA受体的参数
    nmda_par = dict(delay_step=int(0.5 / dt), tau_decay=100, tau_rise=2., E=0., cc_Mg=1., a=0.5)  # NMDA受体的参数

    # 兴奋性神经元群（锥体神经元）
    A = bp.dyn.LIF(num_A, V_rest=-70., V_reset=-55., V_th=-50., tau=20., R=40.,
                   tau_ref=2., V_initializer=bp.init.OneInit(-55.))
    B = bp.dyn.LIF(num_B, V_rest=-70., V_reset=-55., V_th=-50., tau=20., R=40.,
                   tau_ref=2., V_initializer=bp.init.OneInit(-55.))
    N = bp.dyn.LIF(num_N, V_rest=-70., V_reset=-55., V_th=-50., tau=20., R=40.,
                   tau_ref=2., V_initializer=bp.init.OneInit(-55.))

    # 抑制性神经元细胞（中间神经元）
    I = bp.dyn.LIF(num_inh, V_rest=-70., V_reset=-55., V_th=-50., tau=10., R=50.,
                   tau_ref=1., V_initializer=bp.init.OneInit(-55.))

    # 产生输入信号的神经元群（给予神经元群A和B泊松刺激）
    IA = PoissonStim(num_A, freq_var=10., t_interval=50., freq_mean=mu0 + mu0 / 100. * coherence)
    IB = PoissonStim(num_B, freq_var=10., t_interval=50., freq_mean=mu0 - mu0 / 100. * coherence)

    # 产生噪声的神经元群（模拟其他脑区传来的噪音）
    self.noise_A = bp.dyn.PoissonGroup(num_A, freqs=poisson_freq)
    self.noise_B = bp.dyn.PoissonGroup(num_B, freqs=poisson_freq)
    self.noise_N = bp.dyn.PoissonGroup(num_N, freqs=poisson_freq)
    self.noise_I = bp.dyn.PoissonGroup(num_inh, freqs=poisson_freq)

    # IA和A的连接、IB和B的连接
    self.IA2A = bp.dyn.ExpCOBA(IA, A, bp.conn.One2One(), g_max=g_max_ext2E_AMPA, **ampa_par)
    self.IB2B = bp.dyn.ExpCOBA(IB, B, bp.conn.One2One(), g_max=g_max_ext2E_AMPA, **ampa_par)

    # 兴奋性神经元群A、B、N内部和之间的连接（每个突触都同时有AMPA和NMDA受体）
    self.A2A_AMPA = bp.dyn.ExpCOBA(A, A, bp.conn.All2All(), g_max=g_max_E2E_AMPA * w_pos, **ampa_par)
    self.A2A_NMDA = bp.dyn.NMDA(A, A, bp.conn.All2All(), g_max=g_max_E2E_NMDA * w_pos, **nmda_par)

    self.A2B_AMPA = bp.dyn.ExpCOBA(A, B, bp.conn.All2All(), g_max=g_max_E2E_AMPA * w_neg, **ampa_par)
    self.A2B_NMDA = bp.dyn.NMDA(A, B, bp.conn.All2All(), g_max=g_max_E2E_NMDA * w_neg, **nmda_par)

    self.A2N_AMPA = bp.dyn.ExpCOBA(A, N, bp.conn.All2All(), g_max=g_max_E2E_AMPA, **ampa_par)
    self.A2N_NMDA = bp.dyn.NMDA(A, N, bp.conn.All2All(), g_max=g_max_E2E_NMDA, **nmda_par)

    self.B2A_AMPA = bp.dyn.ExpCOBA(B, A, bp.conn.All2All(), g_max=g_max_E2E_AMPA * w_neg)
    self.B2A_NMDA = bp.dyn.NMDA(B, A, bp.conn.All2All(), g_max=g_max_E2E_NMDA * w_neg, **nmda_par)

    self.B2B_AMPA = bp.dyn.ExpCOBA(B, B, bp.conn.All2All(), g_max=g_max_E2E_AMPA * w_pos, **ampa_par)
    self.B2B_NMDA = bp.dyn.NMDA(B, B, bp.conn.All2All(), g_max=g_max_E2E_NMDA * w_pos, **nmda_par)

    self.B2N_AMPA = bp.dyn.ExpCOBA(B, N, bp.conn.All2All(), g_max=g_max_E2E_AMPA, **ampa_par)
    self.B2N_NMDA = bp.dyn.NMDA(B, N, bp.conn.All2All(), g_max=g_max_E2E_NMDA, **nmda_par)

    self.N2A_AMPA = bp.dyn.ExpCOBA(N, A, bp.conn.All2All(), g_max=g_max_E2E_AMPA * w_neg, **ampa_par)
    self.N2A_NMDA = bp.dyn.NMDA(N, A, bp.conn.All2All(), g_max=g_max_E2E_NMDA * w_neg, **nmda_par)

    self.N2B_AMPA = bp.dyn.ExpCOBA(N, B, bp.conn.All2All(), g_max=g_max_E2E_AMPA * w_neg, **ampa_par)
    self.N2B_NMDA = bp.dyn.NMDA(N, B, bp.conn.All2All(), g_max=g_max_E2E_NMDA * w_neg, **nmda_par)

    self.N2N_AMPA = bp.dyn.ExpCOBA(N, N, bp.conn.All2All(), g_max=g_max_E2E_AMPA, **ampa_par)
    self.N2N_NMDA = bp.dyn.NMDA(N, N, bp.conn.All2All(), g_max=g_max_E2E_NMDA, **nmda_par)

    # 兴奋性神经元群A、B、N到抑制性神经元群I的连接（每个突触都同时有AMPA和NMDA受体）
    self.A2I_AMPA = bp.dyn.ExpCOBA(A, I, bp.conn.All2All(), g_max=g_max_E2I_AMPA, **ampa_par)
    self.A2I_NMDA = bp.dyn.NMDA(A, I, bp.conn.All2All(), g_max=g_max_E2I_NMDA, **nmda_par)

    self.B2I_AMPA = bp.dyn.ExpCOBA(B, I, bp.conn.All2All(), g_max=g_max_E2I_AMPA, **ampa_par)
    self.B2I_NMDA = bp.dyn.NMDA(B, I, bp.conn.All2All(), g_max=g_max_E2I_NMDA, **nmda_par)

    self.N2I_AMPA = bp.dyn.ExpCOBA(N, I, bp.conn.All2All(), g_max=g_max_E2I_AMPA, **ampa_par)
    self.N2I_NMDA = bp.dyn.NMDA(N, I, bp.conn.All2All(), g_max=g_max_E2I_NMDA, **nmda_par)

    # 抑制性神经元群I到兴奋性神经元群A、B、N的连接
    self.I2A = bp.dyn.ExpCOBA(I, A, bp.conn.All2All(), g_max=g_max_I2E_GABAa, **gaba_par)
    self.I2B = bp.dyn.ExpCOBA(I, B, bp.conn.All2All(), g_max=g_max_I2E_GABAa, **gaba_par)
    self.I2N = bp.dyn.ExpCOBA(I, N, bp.conn.All2All(), g_max=g_max_I2E_GABAa, **gaba_par)

    # 抑制性神经元群I内部的连接
    self.I2I = bp.dyn.ExpCOBA(I, I, bp.conn.All2All(), g_max=g_max_I2I_GABAa, **gaba_par)

    # 产生噪声的神经元群到神经元群A、B、N、I的连接
    self.noise2A = bp.dyn.ExpCOBA(self.noise_A, A, bp.conn.One2One(), g_max=g_max_ext2E_AMPA, **ampa_par)
    self.noise2B = bp.dyn.ExpCOBA(self.noise_B, B, bp.conn.One2One(), g_max=g_max_ext2E_AMPA, **ampa_par)
    self.noise2N = bp.dyn.ExpCOBA(self.noise_N, N, bp.conn.One2One(), g_max=g_max_ext2E_AMPA, **ampa_par)
    self.noise2I = bp.dyn.ExpCOBA(self.noise_I, I, bp.conn.One2One(), g_max=g_max_ext2I_AMPA, **ampa_par)

    # 将各个神经元群的变量保存到类中
    self.A = A
    self.B = B
    self.N = N
    self.I = I
    self.IA = IA
    self.IB = IB


# 数值模拟
net = DecisionMaking(coherence=25.6)
runner = bp.dyn.DSRunner(net, monitors=['A.spike', 'B.spike', 'IA.freq', 'IB.freq'])
t = runner(total_period)

# 可视化
fig, gs = plt.subplots(4, 1, figsize=(10, 12), sharex='all')
t_start = 0.

# 神经元群A的脉冲发放图
fig.add_subplot(gs[0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['A.spike'], markersize=1)
plt.title("Spiking activity of group A")
plt.ylabel("Neuron Index")

# 神经元群B的脉冲发放图
fig.add_subplot(gs[1])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['B.spike'], markersize=1)
plt.title("Spiking activity of group B")
plt.ylabel("Neuron Index")

# 神经元群A、B的发放率图
fig.add_subplot(gs[2])
rateA = bp.measure.firing_rate(runner.mon['A.spike'], width=10.)
rateB = bp.measure.firing_rate(runner.mon['B.spike'], width=10.)
plt.plot(runner.mon.ts, rateA, label="Group A")
plt.plot(runner.mon.ts, rateB, label="Group B")
plt.ylabel('Firing rate [Hz]')
plt.title("Population activity")
plt.legend()

# 神经元群A、B接收到的刺激频率图
fig.add_subplot(gs[3])
plt.plot(runner.mon.ts, runner.mon['IA.freq'], label="group A")
plt.plot(runner.mon.ts, runner.mon['IB.freq'], label="group B")
plt.title("Input activity")
plt.ylabel("Firing rate [Hz]")
plt.legend()

for i in range(4):
  gs[i].axvline(pre_stimulus_period, linestyle='dashed', color=u'#444444')
  gs[i].axvline(pre_stimulus_period + stimulus_period, linestyle='dashed', color=u'#444444')

plt.xlim(t_start, total_period + 1)
plt.xlabel("Time [ms]")
plt.tight_layout()
plt.show()

# plt.savefig('E:\\2021-2022RA\\神经计算建模实战\\NeuralModeling\\'
#             'images_network_models\\decision_making_output_c=-25.6-3.png')