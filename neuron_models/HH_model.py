import brainpy as bp
import brainpy.math as bm


class HH(bp.dyn.NeuGroup):
	def __init__(self, size, ENa=50., gNa=120., EK=-77., gK=36., 
                 EL=-54.387, gL=0.03, V_th=20., C=1.0, T=6.3):
		# 初始化
		super(HH, self).__init__(size=size)

		# 定义神经元参数
		self.ENa = ENa
		self.EK = EK
		self.EL = EL
		self.gNa = gNa
		self.gK = gK
		self.gL = gL
		self.C = C
		self.V_th = V_th
		self.Q10 = 3.
		self.T_base = 6.3
		self.phi = self.Q10 ** ((T - self.T_base) / 10)

		# 定义神经元变量
		self.V = bm.Variable(-70.68 * bm.ones(self.num))  # 膜电位
		self.m = bm.Variable(0.0266 * bm.ones(self.num))  # 离子通道m
		self.h = bm.Variable(0.772 * bm.ones(self.num))  # 离子通道h
		self.n = bm.Variable(0.235 * bm.ones(self.num))  # 离子通道n
		# 神经元接收到的输入电流
		self.input = bm.Variable(bm.zeros(self.num))
		# 神经元发放状态
		self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
		# 神经元上次发放的时刻
		self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

		# 定义积分函数
		self.integral = bp.odeint(f=self.derivative, method='exp_auto')

	@property
	def derivative(self):
		return bp.JointEq([self.dV, self.dm, self.dh, self.dn])

	def dm(self, m, t, V):
		alpha = 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
		beta = 4.0 * bm.exp(-(V + 65) / 18)
		dmdt = alpha * (1 - m) - beta * m
		return self.phi * dmdt

	def dh(self, h, t, V):
		alpha = 0.07 * bm.exp(-(V + 65) / 20.)
		beta = 1 / (1 + bm.exp(-(V + 35) / 10))
		dhdt = alpha * (1 - h) - beta * h
		return self.phi * dhdt

	def dn(self, n, t, V):
		alpha = 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
		beta = 0.125 * bm.exp(-(V + 65) / 80)
		dndt = alpha * (1 - n) - beta * n
		return self.phi * dndt

	def dV(self, V, t, m, h, n):
		I_Na = (self.gNa * m ** 3.0 * h) * (V - self.ENa)
		I_K = (self.gK * n ** 4.0) * (V - self.EK)
		I_leak = self.gL * (V - self.EL)
		dVdt = (- I_Na - I_K - I_leak + self.input) / self.C
		return dVdt

	# 更新函数：每个时间步都会运行此函数完成变量更新
	def update(self, tdi):
		t, dt = tdi.t, tdi.dt
		# 更新下一时刻变量的值
		V, m, h, n = self.integral(self.V, self.m, self.h, self.n, t, dt=dt)
		# 判断神经元是否产生膜电位
		self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
		# 更新神经元发放的时间
		self.t_last_spike.value = bm.where(self.spike, t, self.t_last_spike)
		self.V.value = V
		self.m.value = m
		self.h.value = h
		self.n.value = n
		self.input[:] = 0.  # 重置神经元接收到的输入


import matplotlib.pyplot as plt
import numpy as np

currents, length = bp.inputs.section_input(values=[0., bm.asarray([1., 2., 4., 8., 10., 15.]), 0.],
																					 durations=[10, 2, 25],
																					 return_length=True)

hh = HH(currents.shape[1])
runner = bp.DSRunner(hh, monitors=['V', 'm', 'h', 'n'], inputs=['input', currents, 'iter'])
runner.run(length)

# 可视化
bp.visualize.line_plot(runner.mon.ts, runner.mon.V, ylabel='V (mV)',
											 plot_ids=np.arange(currents.shape[1]))
# 将电流变化画在膜电位变化的下方
plt.plot(runner.mon.ts, bm.where(currents[:, -1] > 0, 10., 0.).numpy() - 90)
plt.tight_layout()
plt.show()

