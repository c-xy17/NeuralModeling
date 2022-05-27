import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

from network_models.EI_balance import EINet


dur_per_I = 500.
Is = np.array([10., 15., 20., 30., 40., 50., 60., 70.])
inputs, total_dur = bp.inputs.constant_input([(Is[0], dur_per_I), (Is[1], dur_per_I),
                                              (Is[2], dur_per_I), (Is[3], dur_per_I),
                                              (Is[4], dur_per_I), (Is[5], dur_per_I),
                                              (Is[6], dur_per_I), (Is[7], dur_per_I),])
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

def fit_fr(neuron_type, color):
  firing_rates = []
  for i in range(8):
    start = int((i * dur_per_I + 100) / bm.get_dt())  # 从每一阶段的第100ms开始计算
    end = start + int(400 / bm.get_dt())   # 从开始到结束选取共400ms
    firing_rates.append(np.mean(runner.mon[neuron_type + '.spike'][start: end])) # 计算整个时间段的平均发放率
  firing_rates = np.asarray(firing_rates)

  plt.scatter(Is, firing_rates, color=color, alpha=0.7)

  # 线性拟合
  model = linear_model.LinearRegression()
  model.fit(Is.reshape(-1, 1), firing_rates.reshape(-1, 1))
  # 画出拟合直线
  x = np.array([5., 75.])
  y = model.coef_[0] * x + model.intercept_[0]
  plt.plot(x, y, color=color, label=neuron_type + ' neurons')

fit_fr('E', u'#d62728')
fit_fr('I', u'#1f77b4')

plt.xlabel('external input')
plt.ylabel('Mean firing rate')
plt.legend()
plt.show()
