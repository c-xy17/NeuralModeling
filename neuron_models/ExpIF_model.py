import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


class ExpIF(bp.dyn.NeuGroup):
	def __init__(self, size, V_rest=-65., V_reset=-68., V_th=20., V_T=-60., delta_T=1.,
	             R=1., tau=10., tau_ref=2., **kwargs):
		# 初始化父类
		super(ExpIF, self).__init__(size=size, **kwargs)

		# 初始化参数
		self.V_rest = V_rest
		self.V_reset = V_reset
		self.V_th = V_th
		self.V_T = V_T
		self.delta_T = delta_T
		self.R = R
		self.tau = tau
		self.tau_ref = tau_ref

		# 初始化变量
		self.V = bm.Variable(bm.random.randn(self.num) + V_reset)
		self.input = bm.Variable(bm.zeros(self.num))
		self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)  # 上一次脉冲发放时间
		self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))  # 是否处于不应期
		self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))  # 脉冲发放状态

		# 使用指数欧拉方法进行积分
		self.integral = bp.odeint(f=self.derivative, method='exp_auto')

	# 定义膜电位关于时间变化的微分方程
	def derivative(self, V, t, Iext):
		exp_v = self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
		dvdt = (- (V - self.V_rest) + exp_v + self.R * Iext) / self.tau
		return dvdt

	def update(self, _t, _dt):
		# 以数组的方式对神经元进行更新
		refractory = (_t - self.t_last_spike) <= self.tau_ref  # 判断神经元是否处于不应期
		V = self.integral(self.V, _t, self.input, dt=_dt)  # 根据时间步长更新膜电位
		V = bm.where(refractory, self.V, V)  # 若处于不应期，则返回原始膜电位self.V，否则返回更新后的膜电位V
		spike = V > self.V_th  # 将大于阈值的神经元标记为发放了脉冲
		self.spike.value = spike  # 更新神经元脉冲发放状态
		self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)  # 更新最后一次脉冲发放时间
		self.V.value = bm.where(spike, self.V_reset, V)  # 将发放了脉冲的神经元膜电位置为V_reset，其余不变
		self.refractory.value = bm.logical_or(refractory, spike)  # 更新神经元是否处于不应期
		self.input[:] = 0.  # 重置外界输入


# # 运行ExpIF模型
# group = ExpIF(1)
# runner = bp.DSRunner(group, monitors=['V'], inputs=('input', 5.), dt=0.01)
# runner(500)
#
# # 结果可视化
# plt.plot(runner.mon.ts, runner.mon.V)
# plt.xlabel('t (ms)')
# plt.ylabel('V (mV)')

# plt.show()
