import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


class HindmarshRose(bp.dyn.NeuGroup):
	def __init__(self, size, a=1., b=3., c=1., d=5., r=0.001, s=4., x_r=-1.6,
	             theta=1.0, **kwargs):
		# 初始化父类
		super(HindmarshRose, self).__init__(size=size, **kwargs)

		# 初始化参数
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.r = r
		self.s = s
		self.theta = theta
		self.x_r = x_r

		# 初始化变量
		self.x = bm.Variable(bm.random.randn(self.num) + x_r)
		self.y = bm.Variable(bm.ones(self.num) * -10.)
		self.z = bm.Variable(bm.ones(self.num) * 1.7)
		self.input = bm.Variable(bm.zeros(self.num))
		self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))  # 脉冲发放状态

		# 定义积分器
		self.integral = bp.odeint(f=self.derivative, method='exp_auto')

	def dV(self, x, t, y, z, Iext):
		return y - self.a * x * x * x + self.b * x * x - z + Iext

	def dy(self, y, t, x):
		return self.c - self.d * x * x - y

	def dz(self, z, t, x):
		return self.r * (self.s * (x - self.x_r) - z)

	# 将两个微分方程联合为一个，以便同时积分
	@property
	def derivative(self):
		return bp.JointEq([self.dV, self.dy, self.dz])

	def update(self, _t, _dt):
		x, y, z = self.integral(self.x, self.y, self.z, _t, self.input, dt=_dt)  # 更新变量V, y, z
		self.spike.value = bm.logical_and(x >= self.theta, self.x < self.theta)  # 判断神经元是否发放脉冲
		self.x.value = x
		self.y.value = y
		self.z.value = z
		self.input[:] = 0.  # 重置外界输入


# group = HindmarshRose(10)
# runner = bp.StructRunner(group, monitors=['V', 'y', 'z'], inputs=('input', 2.), dt=0.01)
# runner(20)
# runner(100)  # 再运行100ms
# plt.figure(figsize=(6, 4))
# bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V', show=False)
# bp.visualize.line_plot(runner.mon.ts, runner.mon.y, legend='y', show=False)
# bp.visualize.line_plot(runner.mon.ts, runner.mon.z, legend='z', show=True)
