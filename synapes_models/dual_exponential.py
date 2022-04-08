import brainpy as bp
import brainpy.math as bm

from run_synapse import run_syn


class DualExponential(bp.dyn.TwoEndConn):
	def __init__(self, pre, post, conn, g_max=1., tau_decay=8., tau_rise=2., delay_step=2,
	             E=0., syn_type='CUBA', method='exp_auto', **kwargs):
		super(DualExponential, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
		self.check_pre_attrs('spike')
		self.check_post_attrs('input', 'V')

		# 初始化参数
		self.tau_decay = tau_decay
		self.tau_rise = tau_rise
		self.g_max = g_max
		self.h_max = (tau_decay - tau_rise) / (tau_decay * tau_rise)
		self.delay_step = delay_step
		self.E = E

		assert syn_type == 'CUBA' or syn_type == 'COBA'  # current-based 或 conductance-based
		self.type = syn_type

		# 获取关于连接的信息
		self.pre2post = self.conn.require('pre2post')  # 获取从pre到post的连接信息

		# 初始化变量
		self.g = bm.Variable(bm.zeros(self.post.num))
		self.h = bm.Variable(bm.zeros(self.post.num))
		self.delay = bm.LengthDelay(self.pre.spike, delay_step)  # 定义一个延迟处理器

		# 定义微分方程及其对应的积分函数
		self.int_h = bp.odeint(method=method, f=lambda h, t: -h / self.tau_rise)
		self.int_g = bp.odeint(method=method, f=lambda g, t, h: -g / self.tau_decay + h * self.g_max)

	def update(self, _t, _dt):
		# 将突触前神经元传来的信号延迟delay_step的时间步长
		delayed_pre_spike = self.delay(self.delay_step)
		self.delay.update(self.pre.spike)

		# 根据连接模式计算各个突触后神经元收到的信号强度
		post_sp = bm.pre2post_event_sum(delayed_pre_spike, self.pre2post, self.post.num, self.h_max)
		# 突触的电导g的更新包括常规积分和突触前脉冲带来的跃变
		self.h.value = self.int_h(self.h, _t) + post_sp
		self.g.value = self.int_g(self.g, _t, self.h)

		# 根据不同模式计算突触后电流
		if self.type == 'CUBA':
			self.post.input += self.g * (self.E - (-65.))  # E - V_rest
		else:
			self.post.input += self.g * (self.E - self.post.V)


run_syn(DualExponential, syn_type='CUBA', title='Dual Exponential Synapse Model (Current-Based)')
run_syn(DualExponential, syn_type='COBA', title='Dual Exponential Synapse Model (Conductance-Based)')
