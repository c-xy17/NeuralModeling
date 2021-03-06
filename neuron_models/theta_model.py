import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


class Theta(bp.dyn.NeuGroup):
	def __init__(self, size, b=0., c=0., t_ref=0., **kwargs):
		# 初始化父类
		super(Theta, self).__init__(size=size, **kwargs)

		# 初始化参数
		self.b = b
		self.c = c
		self.t_ref = t_ref

		# 初始化变量
		self.theta = bm.Variable(bm.random.randn(self.num) * bm.pi / 18)
		self.input = bm.Variable(bm.zeros(self.num))
		self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)  # 上一次脉冲发放时间
		self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))  # 是否处于不应期
		self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))  # 脉冲发放状态

		# 使用指数欧拉方法进行积分
		self.integral = bp.odeint(f=self.derivative, method='exponential_euler')

	# 定义膜电位关于时间变化的微分方程
	def derivative(self, theta, t, I_ext):
		dthetadt = -bm.cos(theta) + (1. + bm.cos(theta)) * (2 * self.c + 1 / 2 + 2 * self.b * I_ext)
		return dthetadt

	def update(self, _t, _dt):
		# 以数组的方式对神经元进行更新
		refractory = (_t - self.t_last_spike) <= self.t_ref  # 判断神经元是否处于不应期
		theta = self.integral(self.theta, _t, self.input, dt=_dt) % (2 * bm.pi)  # 根据时间步长更新theta
		theta = bm.where(refractory, self.theta, theta)  # 若处于不应期，则返回原始膜电位self.theta，否则返回更新后的膜电位V
		spike = (theta < bm.pi) & (self.theta > bm.pi)  # 将theta从2*pi跳跃到0的神经元标记为发放了脉冲
		self.spike[:] = spike  # 更新神经元脉冲发放状态
		self.t_last_spike[:] = bm.where(spike, _t, self.t_last_spike)  # 更新最后一次脉冲发放时间
		self.theta[:] = theta
		self.refractory[:] = bm.logical_or(refractory, spike)  # 更新神经元是否处于不应期
		self.input[:] = 0.  # 重置外界输入


# 根据QIF模型的参数计算theta神经元模型的参数
V_rest, R, tau, t_ref = -65., 1., 10., 5.
a_0, V_c = .07, -50.0
b = a_0 * R / tau ** 2
c = a_0 ** 2 / tau ** 2 * (V_rest * V_c - ((V_rest + V_c) / 2) ** 2)

# 运行theta神经元模型
neu = Theta(1, b=b, c=c, t_ref=t_ref)
runner = bp.DSRunner(neu, monitors=['theta'], inputs=('input', 6.))
runner(500)

# 可视化
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 4))

ax1.plot(runner.mon.ts, runner.mon.theta)
ax1.set_xlabel('t (ms)')
ax1.set_ylabel('$\Theta$')

ax2.plot(bm.cos(runner.mon.theta), bm.sin(runner.mon.theta))
ax2.set_xlabel('$\cos(\Theta)$')
ax2.set_ylabel('$\sin(\Theta)$')

plt.tight_layout()
plt.show()

# fig, gs = bp.visualize.get_figure(1, 2, 4.5, 6)
# fig.add_subplot(gs[0, 0])
# bp.visualize.line_plot(runner.mon.ts, runner.mon.theta, ylabel='theta', show=False)
# fig.add_subplot(gs[0, 1])
# bp.visualize.line_plot(bp.math.cos(runner.mon.theta),
#                        bp.math.sin(runner.mon.theta),
#                        xlabel='cos(theta)', ylabel='sin(theta)',
#                        show=True)
