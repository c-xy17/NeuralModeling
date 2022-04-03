import brainpy as bp
import brainpy.math as bm

from run_synapse import run_syn, run_syn2


class DualExponential(bp.dyn.TwoEndConn):
	def __init__(self, pre, post, conn, g_max=1., tau_decay=10.0, tau_rise=1., delay_step=2, E=None,
	             syn_type='CUBA', method='exp_auto', **kwargs):
		super(DualExponential, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
		self.check_pre_attrs('spike')
		self.check_post_attrs('input', 'V')

		# 初始化参数
		self.tau_decay = tau_decay
		self.tau_rise = tau_rise
		self.g_max = g_max
		self.delay_step = delay_step

		assert syn_type == 'CUBA' or syn_type == 'COBA'  # current-based 或 conductance-based
		self.type = syn_type
		if syn_type == 'COBA':
			self.E = E if E is not None else 0.

		# 获取关于连接的信息
		assert self.conn is not None
		self.pre2post = self.conn.require('pre2post')

		# 初始化变量
		self.g = bm.Variable(bm.zeros(self.post.num))
		self.h = bm.Variable(bm.zeros(self.post.num))
		self.delay = bm.LengthDelay(self.pre.spike, delay_step)  # 定义一个延迟处理器

		# 定义微分方程及其对应的积分函数
		self.int_h = bp.odeint(method=method, f=lambda h, t: -h / self.tau_rise)
		self.int_g = bp.odeint(method=method, f=lambda g, t, h: -g / self.tau_decay + h)

	def update(self, _t, _dt):
		# 将突触前神经元传来的信号延迟delay的时长
		delayed_pre_spike = self.delay(self.delay_step)
		self.delay.update(self.pre.spike)

		# 根据连接模式计算各个突触后神经元收到的信号强度
		post_sp = bm.pre2post_event_sum(delayed_pre_spike, self.pre2post, self.post.num, self.g_max)
		# 突触的电导g的更新包括常规积分和突触前脉冲带来的跃变
		h = self.int_h(self.h, _t) + post_sp
		g = self.int_g(self.g, _t, self.h)
		self.h, self.g = h, g

		# 根据不同模式计算突触后电流
		if self.type == 'CUBA':
			self.post.input += self.g
		else:
			self.post.input += self.g * (self.E - self.post.V)


run_syn2(DualExponential, syn_type='CUBA')
run_syn2(DualExponential, syn_type='COBA', E=0.)
