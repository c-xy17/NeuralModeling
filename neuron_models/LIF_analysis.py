import subprocess

from LIF_model import *

import numpy as np

bm.set_platform('cpu')

duration = 2000  # 设定仿真时长

# inputs, _ = bp.inputs.constant_input([(0, duration)])
# # inputs = inputs.astype(float)
# I_list = [0]
# for I_ext in range(10, 200, 10):  # 为每个神经元设定不同大小的恒定外部输入
#   input, _ = bp.inputs.constant_input([(I_ext, duration)])
#   # input = input.astype(float)
#   inputs = np.vstack((inputs, input))
#   I_list.append(I_ext)
# I_list = bm.array(I_list)
# inputs = bm.array(inputs)

Icur = np.arange(10, 201, 2)
Iext= bp.inputs.section_input(values=[Icur], durations=[duration])

neu = LIF(len(Icur), V_rest=-5., V_th=20., t_ref=5.)  # 定义神经元群
runner = bp.StructRunner(neu, monitors=['V', 'refractory', 'spike'], inputs=('input', Iext, 'iter'))  # 设置运行器
runner(duration=duration)
f_list = runner.mon.spike.sum(axis=0) / (duration / 1000)  # 计算每个神经元发放脉冲的次数
bp.visualize.line_plot(Icur, f_list, xlabel='Input current', ylabel='spiking frequency', show=True)  # 画出曲线


duration = 200

neu = LIF(2, V_rest=-5., V_th=20., t_ref=5.)
runner = bp.StructRunner(neu, monitors=['V', 'refractory', 'spike'], inputs=('input', bm.asarray([20, 26.])))
runner(duration)
# bp.visualize.line_plot(runner.mon.ts, runner.mon.V, xlabel='t', ylabel='V')
bp.visualize.line_plot(runner.mon.ts, runner.mon.V, plot_ids=[0, 1], xlabel='t', ylabel='V', show=True)

# runner = bp.StructRunner(neu, monitors=['V', 'refractory', 'spike'], inputs=('input', 26))
# runner(duration)
# bp.visualize.line_plot(runner.mon.ts, runner.mon.V[:, 0], xlabel='t', ylabel='V', show=True)
