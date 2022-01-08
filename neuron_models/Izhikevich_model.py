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

	@property
	def derivative(self):
		return bp.JointEq([self.dV, self.du])

	def update(self, _t, _dt):
		V, u = self.integral(self.V, self.u, _t, self.input, dt=_dt)
		refractory = (_t - self.t_last_spike) <= self.tau_ref
		V = bm.where(refractory, self.V, V)
		spike = self.V_th <= V
		self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)
		self.V.value = bm.where(spike, self.c, V)
		self.u.value = bm.where(spike, u + self.d, u)
		self.refractory.value = bm.logical_or(refractory, spike)
		self.spike.value = spike
		self.input[:] = 0.


# 运行Izhikevich模型
group = Izhikevich(10)
runner = bp.StructRunner(group, monitors=['V', 'u'], inputs=('input', 20.))
runner(200)  # 运行时长为200ms
bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V', show=False)
bp.visualize.line_plot(runner.mon.ts, runner.mon.u, legend='u', show=True)
