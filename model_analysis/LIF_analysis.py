import subprocess

from neuron_models.LIF_model import *

import numpy as np
import matplotlib.pyplot as plt

# duration = 1000  # 设定仿真时长
# I_cur = np.arange(0, 600, 2)  # 定义大小为10, 20, ..., 1000的100束电流
#
# neu = LIF(len(I_cur), tau=5., t_ref=5.)  # 定义神经元群
# runner = bp.StructRunner(neu, monitors=['spike'], inputs=('input', I_cur), dt=0.01)  # 设置运行器，其中每一个神经元接收I_cur的一个恒定电流
# runner(duration=duration)  # 运行神经元模型
# f_list = runner.mon.spike.sum(axis=0) / (duration / 1000)  # 计算每个神经元的脉冲发放次数
# # bp.visualize.line_plot(I_cur, f_list, xlabel='Input current', ylabel='spiking frequency', show=True)  # 画出曲线
#
# plt.plot(I_cur, f_list)
# plt.xlabel('Input current')
# plt.ylabel('spiking frequency')
# plt.show()


duration = 200

neu1 = LIF(1, t_ref=5.)
neu1.V[:] = bm.array([-5.])
runner = bp.StructRunner(neu1, monitors=['V'], inputs=('input', 20))  # 给neu1一个大小为20的恒定电流
runner(duration)
# bp.visualize.line_plot(runner.mon.ts, runner.mon.V, ylabel='V', legend='input=20', show=False)
plt.plot(runner.mon.ts, runner.mon.V, label='$I = 20$')

neu2 = LIF(1, t_ref=5.)
neu2.V[:] = bm.array([-5.])
runner = bp.StructRunner(neu2, monitors=['V'], inputs=('input', 21))  # 给neu2一个大小为21的恒定电流
runner(duration)
# bp.visualize.line_plot(runner.mon.ts, runner.mon.V, ylabel='V', legend='input=21', show=True)
plt.plot(runner.mon.ts, runner.mon.V, label='$I = 21$')

plt.xlabel('t (ms)')
plt.ylabel('V')
plt.legend()
plt.show()
