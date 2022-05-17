import brainpy as bp
import brainpy.math as bm

from run_synapse import run_syn


class Exponential(bp.dyn.TwoEndConn):
	def __init__(self, pre, post, conn, g_max=0.02, tau=12., delay_step=2, E=0.,
	             syn_type='CUBA', method='exp_auto', **kwargs):
		super(Exponential, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
		self.check_pre_attrs('spike')
		self.check_post_attrs('input', 'V')

		# 初始化参数
		self.tau = tau
		self.g_max = g_max
		self.delay_step = delay_step
		self.E = E

		assert syn_type == 'CUBA' or syn_type == 'COBA'  # current-based 或 conductance-based
		self.type = syn_type

		# 获取关于连接的信息
		self.pre2post = self.conn.require('pre2post')  # 获取从pre到post的连接信息

		# 初始化变量
		self.g = bm.Variable(bm.zeros(self.post.num))
		self.delay = bm.LengthDelay(self.pre.spike, delay_step)  # 定义一个延迟处理器

		# 定义积分函数
		self.integral = bp.odeint(self.derivative, method=method)

	def derivative(self, g, t):
		dgdt = -g / self.tau
		return dgdt

	def update(self, _t, _dt):
		# 将突触前神经元传来的信号延迟delay_step的时间步长
		delayed_pre_spike = self.delay(self.delay_step)
		self.delay.update(self.pre.spike)

		# 根据连接模式计算各个突触后神经元收到的信号强度
		post_sp = bm.pre2post_event_sum(delayed_pre_spike, self.pre2post, self.post.num, self.g_max)
		# 突触的电导g的更新包括常规积分和突触前脉冲带来的跃变
		self.g.value = self.integral(self.g, _t, dt=_dt) + post_sp

		# 根据不同模式计算突触后电流
		if self.type == 'CUBA':
			self.post.input += self.g * (self.E - (-65.))  # E - V_rest
		else:
			self.post.input += self.g * (self.E - self.post.V)


run_syn(Exponential, syn_type='CUBA', title='Exponential Synapse Model (Current-Based)')
run_syn(Exponential, syn_type='COBA', title='Exponential Synapse Model (Conductance-Based)')
