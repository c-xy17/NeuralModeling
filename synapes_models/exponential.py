import brainpy as bp
import brainpy.math as bm

from run_synapse import run_syn


class Exponential(bp.dyn.TwoEndConn):
	def __init__(self, pre, post, conn, g_max=1., tau=8.0, E=None, syn_type='CUBA', method='exp_auto',
	             delay_step=None, **kwargs):
		super(Exponential, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
		self.check_pre_attrs('spike')
		self.check_post_attrs('input', 'V')

		# 初始化参数
		self.tau = tau
		self.g_max = g_max
		assert syn_type == 'CUBA' or syn_type == 'COBA'  # current-based or conductance-based
		self.type = syn_type
		if syn_type == 'COBA':
			self.E = E

		# 获取关于连接的信息
		assert self.conn is not None
		self.pre2post = self.conn.require('pre2post')

		# 初始化变量
		self.g = bm.Variable(bm.zeros(self.post.num))
		# 将突出前神经元传来的信号延迟delay_step的长度
		self.delay_type, self.delay_step, self.pre_spike = bp.dyn.utils.init_delay(delay_step, self.pre.spike)

		# 定义积分函数
		self.integral = bp.odeint(self.derivative, method=method)

	def derivative(self, g, t):
		dgdt = -g / self.tau
		return dgdt

	def update(self, _t, _dt):
		# 处理delay
		if self.delay_type == 'homo':
			delayed_pre_spike = self.pre_spike(self.delay_step)
			self.pre_spike.update(self.pre.spike)
		elif self.delay_type == 'heter':
			delayed_pre_spike = self.pre_spike(self.delay_step, bm.arange())
			self.pre_spike.update(self.pre.spike)
		else:
			delayed_pre_spike = self.pre.spike

		# 根据连接模式计算各个突触后神经元收到的信号强度
		post_sp = bm.pre2post_event_sum(delayed_pre_spike, self.pre2post, self.post.num, self.g_max)
		# 突触的电导g的更新包括常规积分和突触前脉冲带来的跃变
		self.g.value = self.integral(self.g.value, _t, dt=_dt) + post_sp

		if self.type == 'CUBA':
			self.post.input += self.g
		else:
			self.post.input += self.g * (self.E - self.post.V)


run_syn(Exponential)
