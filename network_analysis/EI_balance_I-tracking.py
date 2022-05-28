import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt

from network_models.EI_balance import EINet

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
lif = bp.dyn.LIF(4000, V_rest=-60., V_th=-50., V_reset=-60., tau=20.,
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