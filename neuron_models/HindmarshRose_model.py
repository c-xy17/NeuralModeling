import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


class HindmarshRose(bp.dyn.NeuGroup):
	def __init__(self, size, a=1., b=3., c=1., d=5., r=0.002, s=4., V_rest=-1.6,
	             V_th=1.0, **kwargs):
		# 初始化父类
		super(HindmarshRose, self).__init__(size=size, **kwargs)

		# 初始化参数
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.r = r
		self.s = s
		self.V_th = V_th
		self.V_rest = V_rest

		# 初始化变量
		self.V = bm.Variable(bm.random.randn(self.num) + V_rest)
		self.z = bm.Variable(bm.ones(self.num) * 1.4)
		self.y = bm.Variable(bm.ones(self.num) * -10.)
		self.input = bm.Variable(bm.zeros(self.num))
		self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)  # 上一次脉冲发放时间
		self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))  # 脉冲发放状态

		# 定义积分器
		self.integral = bp.odeint(f=self.derivative, method='exp_auto')

	def dV(self, V, t, y, z, Iext):
		return y - self.a * V * V * V + self.b * V * V - z + Iext

	def dy(self, y, t, V):
		return self.c - self.d * V * V - y

	def dz(self, z, t, V):
		return self.r * (self.s * (V - self.V_rest) - z)

	# 将两个微分方程联合为一个，以便同时积分
	@property
	def derivative(self):
		return bp.JointEq([self.dV, self.dy, self.dz])

	def update(self, _t, _dt):
		V, y, z = self.integral(self.V, self.y, self.z, _t, self.input, dt=_dt)  # 更新变量V, y, z
		self.spike.value = bm.logical_and(V >= self.V_th, self.V < self.V_th)  # 判断神经元是否发放脉冲
		self.t_last_spike.value = bm.where(self.spike, _t, self.t_last_spike)  # 更新最后一次脉冲发放时间
		self.V.value = V
		self.y.value = y
		self.z.value = z
		self.input[:] = 0.  # 重置外界输入


# 运行Hindmarsh-Rose模型
group = HindmarshRose(10)
runner = bp.StructRunner(group, monitors=['V', 'y', 'z'], inputs=('input', 2.), dt=0.01)
runner(1000)  # 运行时长为1000ms

plt.figure(figsize=(10, 4))
bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V', show=True)

group = HindmarshRose(10)
runner = bp.StructRunner(group, monitors=['V', 'y', 'z'], inputs=('input', 2.), dt=0.01)
runner(20)
runner(100)  # 再运行100ms
plt.figure(figsize=(6, 4))
bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V', show=False)
bp.visualize.line_plot(runner.mon.ts, runner.mon.y, legend='y', show=False)
bp.visualize.line_plot(runner.mon.ts, runner.mon.z, legend='z', show=True)
