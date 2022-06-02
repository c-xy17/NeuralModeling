import brainpy as bp
import numpy as np
import matplotlib.pyplot as plt

from network_models.decision_making_rate_model import DecisionMakingRateModel


# 定义各个阶段的时长
pre_stimulus_period = 100.
stimulus_period = 2000.
delay_period = 500.

# 生成模型
dmnet = DecisionMakingRateModel(1, coherence=25.6, noise_freq=2400.)

# 定义电流随时间的变化
inputs, total_period = bp.inputs.constant_input([(0., pre_stimulus_period),
                                                 (20., stimulus_period),
                                                 (0., delay_period)])
# 运行数值模拟
runner = bp.DSRunner(dmnet, monitors=['s1', 's2', 'r1', 'r2'], inputs=('mu0', inputs, 'iter'))
runner.run(total_period)

# 可视化
fig, gs = plt.subplots(2, 1, figsize=(6, 6), sharex='all')

gs[0].plot(runner.mon.ts, runner.mon.s1, label='s1')
gs[0].plot(runner.mon.ts, runner.mon.s2, label='s2')
gs[0].axvline(pre_stimulus_period, 0., 1., linestyle='dashed', color=u'#444444')
gs[0].axvline(pre_stimulus_period + stimulus_period, 0., 1., linestyle='dashed', color=u'#444444')
gs[0].set_ylabel('gating variable $s$')
gs[0].legend()

gs[1].plot(runner.mon.ts, runner.mon.r1, label='r1')
gs[1].plot(runner.mon.ts, runner.mon.r2, label='r2')
gs[1].axvline(pre_stimulus_period, 0., 1., linestyle='dashed', color=u'#444444')
gs[1].axvline(pre_stimulus_period + stimulus_period, 0., 1., linestyle='dashed', color=u'#444444')
gs[1].set_xlabel('t (ms)')
gs[1].set_ylabel('firing rate $r$')
gs[1].legend()

plt.subplots_adjust(hspace=0.1)
plt.show()

