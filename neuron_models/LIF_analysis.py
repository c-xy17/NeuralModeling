import subprocess

from LIF_models import *

import numpy as np

bm.set_platform('cpu')

duration = 200  # 设定仿真时长

inputs, _ = bp.inputs.constant_input([(0, duration)])
# inputs = inputs.astype(float)
I_list = [0]
for I_ext in range(10, 200, 10):  # 为每个神经元设定不同大小的恒定外部输入
  input, _ = bp.inputs.constant_input([(I_ext, duration)])
  # input = input.astype(float)
  inputs = np.vstack((inputs, input))
  I_list.append(I_ext)
I_list = np.array(I_list)
inputs = np.array(inputs)

neu = LIF(20, V_rest=-5., V_th=20., tau_ref=5.)  # 定义神经元群
runner= bp.StructRunner(neu, monitors=['V', 'refractory', 'spike'], inputs=('input', inputs.T))  # 设置运行器
runner.run(duration)
f_list = runner.mon.spike.sum(axis=0)  # 计算每个神经元发放脉冲的次数
bp.visualize.line_plot(I_list, f_list, show=True)  # 画出曲线
