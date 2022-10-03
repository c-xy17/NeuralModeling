import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15})
plt.rcParams['font.sans-serif'] = ['Times New Roman']


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
		self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))  # 脉冲发放状态

		# 使用指数欧拉方法进行积分
		self.integral = bp.odeint(f=self.derivative, method='exponential_euler')

	# 定义膜电位关于时间变化的微分方程
	def derivative(self, theta, t, I_ext):
		dthetadt = -bm.cos(theta) + (1. + bm.cos(theta)) * (2 * self.c + 1 / 2 + 2 * self.b * I_ext)
		return dthetadt

	def update(self, tdi):
		_t, _dt = tdi.t, tdi.dt
		# 以数组的方式对神经元进行更新
		theta = self.integral(self.theta, _t, self.input, dt=_dt) % (2 * bm.pi)  # 根据时间步长更新theta
		spike = (theta < bm.pi) & (self.theta > bm.pi)  # 将theta从2*pi跳跃到0的神经元标记为发放了脉冲
		self.spike[:] = spike  # 更新神经元脉冲发放状态
		self.theta[:] = theta
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

fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(runner.mon.ts, runner.mon.theta)
ax1.set_xlabel(r'$t$ (ms)')
ax1.set_ylabel(r'$\Theta$')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.savefig('theta_neuron_time_evolution.png', transparent=True, dpi=500)

fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
ax2 = fig.add_subplot(gs[0, 0])
ax2.plot(bm.cos(runner.mon.theta), bm.sin(runner.mon.theta))
ax2.set_xlabel(r'$\cos(\Theta)$')
ax2.set_ylabel(r'$\sin(\Theta)$')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
plt.savefig('theta_neuron_phase.png', transparent=True, dpi=500)

plt.show()

