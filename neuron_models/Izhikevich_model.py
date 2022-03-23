import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np


class Izhikevich(bp.dyn.NeuGroup):
	def __init__(self, size, a=0.02, b=0.20, c=-65., d=2., tau_ref=0.,
	             V_th=30., **kwargs):
		# 初始化父类
		super(Izhikevich, self).__init__(size=size, **kwargs)

		# 初始化参数
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.V_th = V_th
		self.tau_ref = tau_ref

		# 初始化变量
		self.V = bm.Variable(bm.random.randn(self.num) - 65.)
		self.u = bm.Variable(self.V * b)
		self.input = bm.Variable(bm.zeros(self.num))
		self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)  # 上一次脉冲发放时间
		self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))  # 是否处于不应期
		self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))  # 脉冲发放状态

		# 定义积分器
		self.integral = bp.odeint(f=self.derivative, method='exp_auto')

	def dV(self, V, t, u, Iext):
		return 0.04 * V * V + 5 * V + 140 - u + Iext

	def du(self, u, t, V):
		return self.a * (self.b * V - u)

	# 将两个微分方程联合为一个，以便同时积分
	@property
	def derivative(self):
		return bp.JointEq([self.dV, self.du])

	def update(self, _t, _dt):
		V, u = self.integral(self.V, self.u, _t, self.input, dt=_dt)  # 更新变量V, u
		refractory = (_t - self.t_last_spike) <= self.tau_ref  # 判断神经元是否处于不应期
		V = bm.where(refractory, self.V, V)  # 若处于不应期，则返回原始膜电位self.V，否则返回更新后的膜电位V
		u = bm.where(refractory, self.u, u)  # u同理
		spike = V > self.V_th  # 将大于阈值的神经元标记为发放了脉冲
		self.spike.value = spike  # 更新神经元脉冲发放状态
		self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)  # 更新最后一次脉冲发放时间
		self.V.value = bm.where(spike, self.c, V)  # 将发放了脉冲的神经元的V置为c，其余不变
		self.u.value = bm.where(spike, u + self.d, u)  # 将发放了脉冲的神经元的u增加d，其余不变
		self.refractory.value = bm.logical_or(refractory, spike)  # 更新神经元是否处于不应期
		self.input[:] = 0.  # 重置外界输入


# # 运行Izhikevich模型
# group = Izhikevich(10)
# runner = bp.DSRunner(group, monitors=['V', 'u'], inputs=('input', 10.))
# runner(300)
# bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V', show=False)
# bp.visualize.line_plot(runner.mon.ts, runner.mon.u, legend='u', show=True)


def subplot(i, izhi, title=None, input=('input', 10.), duration=200):
	plt.subplot(2, 3, i)
	runner = bp.DSRunner(izhi, monitors=['V', 'u'], inputs=input)
	runner(duration)
	bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V', show=False)
	bp.visualize.line_plot(runner.mon.ts, runner.mon.u, legend='u', show=False)
	plt.title(title)


# plt.figure(figsize=(12, 6))
# # input5, duration = bp.inputs.section_input(values=[-5, 0., 10., 0.],
# #                                            durations=[40, 100, 10, 100],
# #                                            return_length=True)
#
# subplot(1, Izhikevich(1, d=8.), title='Regular Spiking')
# subplot(2, Izhikevich(1, c=-55., d=4.), title='Intrinsic Bursting')
# subplot(3, Izhikevich(1, a=0.1), title='Fast Spiking')
# subplot(4, Izhikevich(1, c=-50.), title='Chattering (Bursting)')
#
# # subplot(5, Izhikevich(2, a=0.1, b=0.26), title='Resonator', input=('input', input5, 'iter'))
# # subplot(5, Izhikevich(2, a=0.1, b=0.26), title='Resonator', input=('input', 0.))
# input5 = bp.inputs.section_input(values=[-30, 3.5], durations=[50, 150])
# subplot(5, Izhikevich(1, b=0.2), title='Rebound Bursting', input=('input', input5, 'iter'))
#
# subplot(6, Izhikevich(1, b=0.25), title='Low Threshold Spiking')
#
# plt.tight_layout()
# plt.show()


# import matplotlib.pyplot as plt
#
# # 分段电流：按照时间顺序，电流值为[-30, 3.5]，持续时长为[50, 150]
# input = bp.inputs.section_input(values=[-30, 3.5], durations=[50, 150])
# # 'iter'表示电流是可迭代的而非一个静态值
# runner = bp.DSRunner(Izhikevich(1), monitors=['V', 'u'], inputs=('input', input, 'iter'))
# runner(200)
#
# plt.plot(runner.mon.ts, runner.mon.V, label='V')
# plt.plot(runner.mon.ts, runner.mon.u, label='u')
# plt.xlabel('Time (ms)')
# plt.legend()
# plt.show()
