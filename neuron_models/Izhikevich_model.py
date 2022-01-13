import brainpy as bp
import brainpy.math as bm


class Izhikevich(bp.NeuGroup):
	def __init__(self, size, a=0.02, b=0.20, c=-65., d=8., tau_ref=0.,
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
		self.V = bm.Variable(bm.random.randn(self.num) - 70.)
		self.u = bm.Variable(bm.ones(self.num))
		self.w = bm.Variable(bm.zeros(self.num))
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
		spike = self.V_th <= V  # 将大于阈值的神经元标记为发放了脉冲
		self.spike.value = spike  # 更新神经元脉冲发放状态
		self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)  # 更新最后一次脉冲发放时间
		self.V.value = bm.where(spike, self.c, V)  # 将发放了脉冲的神经元的V置为c，其余不变
		self.u.value = bm.where(spike, u + self.d, u)   # 将发放了脉冲的神经元的u增加d，其余不变
		self.refractory.value = bm.logical_or(refractory, spike)  # 更新神经元是否处于不应期
		self.input[:] = 0.  # 重置外界输入


# 运行Izhikevich模型
group = Izhikevich(10)
runner = bp.StructRunner(group, monitors=['V', 'u'], inputs=('input', 20.))
runner(200)  # 运行时长为200ms
bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V', show=False)
bp.visualize.line_plot(runner.mon.ts, runner.mon.u, legend='u', show=True)
