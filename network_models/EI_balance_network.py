import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

plt.rcParams.update({"font.size": 15})
plt.rcParams['font.sans-serif'] = ['Times New Roman']


class EINet(bp.Network):
  def __init__(self, num_exc, num_inh, method='exp_auto', **kwargs):
    super(EINet, self).__init__(**kwargs)

    # 搭建神经元
    pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.)  # 神经元模型需要的参数
    E = bp.neurons.LIF(num_exc, **pars, method=method, V_initializer=bp.init.Normal(-60., 4.))
    I = bp.neurons.LIF(num_inh, **pars, method=method, V_initializer=bp.init.Normal(-60., 4.))
    self.E = E
    self.I = I

    # 搭建神经元连接
    E_pars = dict(g_max=0.3, tau=5.)  # 兴奋性突触需要的参数
    I_pars = dict(g_max=3.2, tau=10.)  # 抑制性突触需要的参数
    self.E2E = bp.synapses.Exponential(E, E, bp.conn.FixedProb(prob=0.02), method=method,
                                       output=bp.synouts.COBA(E=0.), **E_pars)
    self.E2I = bp.synapses.Exponential(E, I, bp.conn.FixedProb(prob=0.02), method=method,
                                       output=bp.synouts.COBA(E=0.), **E_pars)
    self.I2E = bp.synapses.Exponential(I, E, bp.conn.FixedProb(prob=0.02), method=method,
                                       output=bp.synouts.COBA(E=-80.), **I_pars)
    self.I2I = bp.synapses.Exponential(I, I, bp.conn.FixedProb(prob=0.02), method=method,
                                       output=bp.synouts.COBA(E=-80.), **I_pars)

  def update(self, tdi):
    e2e_current = self.E2E(tdi)
    self.E2I(tdi)
    self.I2E(tdi)
    self.I2I(tdi)
    self.E(tdi)
    self.I(tdi)


def define_EI_v2():
  # 调用BrainPy中的预置模型

  # 搭建神经元
  pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.)  # 神经元模型需要的参数
  E = bp.neurons.LIF(3200, **pars)
  I = bp.neurons.LIF(800, **pars)
  E.V.value = bm.random.randn(3200) * 4. - 60.  # 随机初始化膜电位
  I.V.value = bm.random.randn(800) * 4. - 60.  # 随机初始化膜电位

  # 搭建神经元连接
  E_pars = dict(output=bp.synouts.COBA(E=0.), g_max=0.3, tau=5.)  # 兴奋性突触需要的参数
  I_pars = dict(output=bp.synouts.COBA(E=-80.), g_max=3.2, tau=10.)  # 抑制性突触需要的参数
  E2E = bp.synapses.Exponential(E, E, bp.conn.FixedProb(prob=0.02), **E_pars)
  E2I = bp.synapses.Exponential(E, I, bp.conn.FixedProb(prob=0.02), **E_pars)
  I2E = bp.synapses.Exponential(I, E, bp.conn.FixedProb(prob=0.02), **I_pars)
  I2I = bp.synapses.Exponential(I, I, bp.conn.FixedProb(prob=0.02), **I_pars)

  einet = bp.Network(E2E, E2I, I2E, I2I, E=E, I=I)

  runner = bp.DSRunner(einet,
                       monitors=['E.spike', 'I.spike', 'E.input', 'E.V'],
                       inputs=[('E.input', 12.), ('I.input', 12.)])
  runner(200.)


def run_EI_net():
  # 数值模拟
  duration = 200.
  net = EINet(3200, 800)
  runner = bp.DSRunner(
    net,
    monitors=['E.spike', 'I.spike', 'E.input', 'E.V'],
    inputs=[('E.input', 12.), ('I.input', 12.)]
  )
  runner.run(duration)

  # 可视化
  # 定义可视化脉冲发放的函数
  def raster_plot(spikes, title, name=None):
    fig, gs = bp.visualize.get_figure(3, 1, 1.5, 6)
    ax = fig.add_subplot(gs[0:2, 0])
    plt.title(title)
    t, neu_index = np.where(spikes)
    t = t * bp.math.get_dt()
    plt.scatter(t, neu_index, s=0.5, c='k')
    plt.ylabel('Neuron Index')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_xlim(-1, duration + 1)

    ax = fig.add_subplot(gs[2, 0])
    rate = bp.measure.firing_rate(spikes, 5.)
    plt.plot(runner.mon.ts, rate)
    plt.ylabel('Firing Rate')
    plt.xlabel(r'$t$ (ms)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-1, duration + 1)

    if name:
      plt.savefig(f'{name}.pdf', transparent=True, dpi=500)

  # 可视化脉冲发放
  raster_plot(runner.mon['E.spike'], 'Spikes of Excitatory Neurons', name='EI_exc_pop')
  raster_plot(runner.mon['I.spike'], 'Spikes of Inhibitory Neurons', name='EI_inh_pop')
  plt.show()


def rate_current_relation():
  # 构建分段电流
  dur_per_I = 500.
  Is = np.array([10., 15., 20., 30., 40., 50., 60., 70.])
  inputs, total_dur = bp.inputs.constant_input([(Is[0], dur_per_I), (Is[1], dur_per_I),
                                                (Is[2], dur_per_I), (Is[3], dur_per_I),
                                                (Is[4], dur_per_I), (Is[5], dur_per_I),
                                                (Is[6], dur_per_I), (Is[7], dur_per_I), ])

  # 运行数值模拟
  net = EINet(3200, 800)
  runner = bp.DSRunner(net,
                       monitors=['E.spike', 'I.spike'],
                       inputs=[('E.input', inputs, 'iter'), ('I.input', inputs, 'iter')])
  runner(total_dur)

  # # 可视化
  # # 定义可视化脉冲发放的函数
  # def raster_plot(spikes, title):
  #   t, neu_index = np.where(spikes)
  #   t = t * bp.math.get_dt()
  #   plt.scatter(t, neu_index, s=0.5, c='k')
  #   plt.title(title)
  #   plt.ylabel('neuron index')
  #
  # # 定义可视化平均发放速率的函数
  # def fr_plot(t, spikes):
  #   rate = bp.measure.firing_rate(spikes, 5.)
  #   plt.plot(t, rate)
  #   plt.ylabel('firing rate')
  #   plt.xlabel('t (ms)')
  #
  # # 可视化脉冲发放
  # fig, gs = plt.subplots(2, 2, gridspec_kw={'height_ratios': [3, 1]}, figsize=(12, 8), sharex='all')
  # plt.sca(gs[0, 0])
  # raster_plot(runner.mon['E.spike'], 'Spikes of Excitatory Neurons')
  # plt.sca(gs[0, 1])
  # raster_plot(runner.mon['I.spike'], 'Spikes of Inhibitory Neurons')
  #
  # # 可视化平均发放速率
  # plt.sca(gs[1, 0])
  # fr_plot(runner.mon.ts, runner.mon['E.spike'])
  # plt.sca(gs[1, 1])
  # fr_plot(runner.mon.ts, runner.mon['I.spike'])
  #
  # plt.subplots_adjust(hspace=0.1)
  # plt.show()

  def fit_fr(neuron_type, color, linestyle):
    # 计算各个电流下网络稳定后的发放率
    firing_rates = []
    for i in range(8):
      start = int((i * dur_per_I + 100) / bm.get_dt())  # 从每一阶段的第100ms开始计算
      end = start + int(400 / bm.get_dt())  # 从开始到结束选取共400ms
      # firing_rates.append(np.mean(runner.mon[neuron_type + '.spike'][start: end]))  # 计算整个时间段的平均发放率
      sps = runner.mon[neuron_type + '.spike'][start: end]
      firing_rates.append(sps.sum() / 0.4 / sps.shape[1])  # 计算整个时间段的平均发放率
    firing_rates = np.asarray(firing_rates)

    plt.scatter(Is, firing_rates, color=color, alpha=0.7)

    # 使用sklearn中的线性回归工具进行线性拟合
    model = linear_model.LinearRegression()
    model.fit(Is.reshape(-1, 1), firing_rates.reshape(-1, 1))
    # 画出拟合直线
    x = np.array([5., 75.])
    y = model.coef_[0] * x + model.intercept_[0]
    plt.plot(x, y, color=color, label=neuron_type + ' neurons', linestyle=linestyle)

  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
  ax = fig.add_subplot(gs[0, 0])
  # 可视化
  fit_fr('E', u'#d62728', linestyle='dotted')
  fit_fr('I', u'#1f77b4', linestyle='dashed')
  plt.xlabel('External input (mA)')
  plt.ylabel('Firing rate (Hz)')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.legend()
  # plt.savefig('EI_balance_fr_I.pdf', transparent=True, dpi=500)
  plt.show()


def I_tracking():
  # 构建一个随时间不断增大的电流
  duration = 200.
  t_start, t_end = 150., 170.
  V_low, V_high = 12., 72.
  current = bp.inputs.ramp_input(V_low, V_high, duration, t_start, t_end)
  current += bp.inputs.section_input(values=[V_low, 0., V_high],
                                     durations=[t_start, t_end - t_start, duration - t_end])

  # 构建EINet运行数值模拟
  net = EINet(4000, 1000)
  net.E.tau = bm.random.normal(20., 3., size=net.E.size)
  runner_einet = bp.DSRunner(net, monitors=['E.spike'],
                             inputs=[('E.input', current, 'iter'), ('I.input', current, 'iter')])
  runner_einet(duration)

  # 构建无连接的LIF神经元群并运行数值模拟
  lif = bp.neurons.LIF(4000, V_rest=-60., V_th=-50., V_reset=-60., tau=20.,
                       tau_ref=5., V_initializer=bp.init.Uniform(-70., -50.))
  lif.tau = bm.random.normal(30., 3., size=lif.size)
  runner_lif = bp.DSRunner(lif, monitors=['spike'], inputs=('input', current, 'iter'))
  runner_lif(duration)

  # net2 = EINet(3200, 100)
  # net2.E.t_last_spike.value = bm.random.uniform(-5., 0., size=net2.E.size)  # 随机初始化不应期状态
  # runner2 = bp.DSRunner(net2, monitors=['E.spike'], inputs=('E.input', current, 'iter'))
  # runner2(duration)

  # 可视化
  # 可视化电流
  ts = runner_einet.mon.ts[1000:]  # 只要100ms之后的数据
  plt.plot(ts, current[1000:], label='input current')
  plt.xlabel('t (ms)')
  plt.ylabel('Input current')
  plt.legend(loc='lower right')

  twin_ax = plt.twinx()
  # 可视化EINet的发放率
  fr = bp.measure.firing_rate(runner_einet.mon['E.spike'], 10.)[1000:]
  fr = (fr - np.min(fr)) / (np.max(fr) - np.min(fr))  # 标准化到[0, 1]之间
  twin_ax.plot(ts, fr, label='EINet firing rate', color=u'#d62728')

  # 可视化无连接的LIF神经元群的发放率
  # fr = bp.measure.firing_rate(runner_lif.mon['spike'], 10.)[1000:]
  # fr = (fr - np.min(fr)) / (np.max(fr) - np.min(fr))  # 标准化[0, 1]之间
  # twin_ax.plot(ts, fr, label='LIF firing rate', color=u'#ff7f0e')
  plt.legend(loc='right')

  twin_ax.set_ylabel('firing rate (normalized)')
  plt.show()


if __name__ == '__main__':
  run_EI_net()
  rate_current_relation()
  I_tracking()
