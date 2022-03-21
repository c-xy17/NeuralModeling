import brainpy as bp
import brainpy.math as bm


class GIF(bp.dyn.NeuGroup):
	def __init__(self, size, V_rest=-70., V_reset=-70., theta_inf=-50., theta_reset=-60.,
				 R=20., tau=20., a=0., b=0.01, k1=0.2, k2=0.02, R1=0., R2=1., A1=0.,
				 A2=0., **kwargs):
		# 初始化父类时计算了self.num供下文使用
		super(GIF, self).__init__(size=size, **kwargs)

		# 初始化参数
		self.V_rest = V_rest
		self.V_reset = V_reset
		self.theta_inf = theta_inf
		self.theta_reset = theta_reset
		self.R = R
		self.tau = tau
		self.a = a
		self.b = b
		self.k1 = k1
		self.k2 = k2
		self.R1 = R1
		self.R2 = R2
		self.A1 = A1
		self.A2 = A2

		# 初始化变量
		self.V = bm.Variable(bm.zeros(self.num) + V_reset)
		self.theta = bm.Variable(bm.ones(self.num) * theta_inf)
		self.input = bm.Variable(bm.zeros(self.num))
		self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
		self.I1 = bm.Variable(bm.zeros(self.num))
		self.I2 = bm.Variable(bm.zeros(self.num))

		# 定义积分器
		self.integral = bp.odeint(f=self.derivative, method='exp_auto')

	def dI1(self, I1, t):
		return - self.k1 * I1

	def dI2(self, I2, t):
		return - self.k2 * I2

	def dVth(self, V_th, t, V):
		return self.a * (V - self.V_rest) - self.b * (V_th - self.theta_inf)

	def dV(self, V, t, I1, I2, Iext):
		return (- (V - self.V_rest) + self.R * Iext + self.R * I1 + self.R * I2) / self.tau

	# 将所有微分方程联合为一个，以便同时积分
	@property
	def derivative(self):
		return bp.JointEq([self.dI1, self.dI2, self.dVth, self.dV])

	def update(self, _t, _dt):
		I1, I2, V_th, V = self.integral(self.I1, self.I2, self.theta, self.V, _t, self.input, dt=_dt)  # 更新变量I1, I2, V
		spike = self.theta <= V  # 将大于阈值的神经元标记为发放了脉冲
		V = bm.where(spike, self.V_reset, V)  # 将发放了脉冲的神经元V置为V_reset，其余赋值为更新后的V
		I1 = bm.where(spike, self.R1 * I1 + self.A1, I1)  # 按照公式更新发放了脉冲的神经元的I1
		I2 = bm.where(spike, self.R2 * I2 + self.A2, I2)  # 按照公式更新发放了脉冲的神经元的I2
		reset_th = bm.logical_and(V_th < self.theta_reset, spike)  # 判断哪些神经元的V_th需要重置
		V_th = bm.where(reset_th, self.theta_reset, V_th)  # 将需要重置的神经元V_th重置为V_th_reset

		# 将更新后的结果赋值给self.*
		self.spike.value = spike
		self.I1.value = I1
		self.I2.value = I2
		self.theta.value = V_th
		self.V.value = V
		self.input[:] = 0.  # 重置外界输入


# fig, gs = bp.visualize.get_figure(1, 2, 4, 6)
#
# # 模拟相位脉冲（phasic spiking）
# group = GIF(10, a=0.005, A1=0., A2=0.)
# runner = bp.DSRunner(group, monitors=['V', 'V_th'], inputs=('input', 1.5), dt=0.01)
# runner(500)
#
# fig.add_subplot(gs[0, 0])
# bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V', zorder=10, show=False)
# bp.visualize.line_plot(runner.mon.ts, runner.mon.V_th, legend='V_th',
#                        title='phasic spiking', show=False)
#
# # 模拟超极化爆发（hyperpolarization-induced bursting）
# group = GIF(10, a=0.03, A1=10., A2=-0.6)
# runner = bp.DSRunner(group, monitors=['V', 'V_th'], inputs=('input', -1), dt=0.01)
# runner(500)
#
# fig.add_subplot(gs[0, 1])
# bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V', zorder=10, show=False)
# bp.visualize.line_plot(runner.mon.ts, runner.mon.V_th, legend='V_th',
#                        title='hyperpolarization-induced bursting', show=True)
