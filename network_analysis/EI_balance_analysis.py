import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt

from network_models.EI_balance import EINet


# 数值模拟
net = EINet(3200, 800)
runner = bp.DSRunner(net,
                     monitors=['E.spike', 'I.spike', 'E.input', 'E.V'],
                     inputs=[('E.input', 12.), ('I.input', 12.)], jit=False)
runner(200.)

# 可视化
# 定义可视化脉冲发放的函数
def raster_plot(spikes, title):
  t, neu_index = np.where(spikes)
  t = t * bp.math.get_dt()
  plt.scatter(t, neu_index, s=0.5, c='k')
  plt.title(title)
  plt.ylabel('neuron index')

# 定义可视化平均发放速率的函数
def fr_plot(t, spikes):
  rate = bp.measure.firing_rate(spikes, 5.)
  plt.plot(t, rate)
  plt.ylabel('firing rate')
  plt.xlabel('t (ms)')

# 可视化脉冲发放
fig, gs = plt.subplots(2, 2, gridspec_kw={'height_ratios': [3, 1]}, figsize=(12, 8), sharex='all')
plt.sca(gs[0, 0])
raster_plot(runner.mon['E.spike'], 'Spikes of Excitatory Neurons')
plt.sca(gs[0, 1])
raster_plot(runner.mon['I.spike'], 'Spikes of Inhibitory Neurons')

# 可视化平均发放速率
plt.sca(gs[1, 0])
fr_plot(runner.mon.ts, runner.mon['E.spike'])
plt.sca(gs[1, 1])
fr_plot(runner.mon.ts, runner.mon['I.spike'])

plt.subplots_adjust(hspace=0.1)
plt.show()


fig, gs = plt.subplots(2, 1, figsize=(6, 4), sharex='all')

i = 299  # 随机指定一个神经元序号
print(runner.mon['E.spike'][:, i].sum())
gs[0].plot(runner.mon.ts, runner.mon['E.input'][:, i] + 12.)  # input中不包括外部输入，在此需加上12
gs[0].set_ylabel('input current')

gs[1].plot(runner.mon.ts, runner.mon['E.V'][:, i])
gs[1].plot([0., 200.], [net.E.V_th, net.E.V_th], '--')
gs[1].set_ylabel('V')

plt.xlabel('t (ms)')
plt.show()

