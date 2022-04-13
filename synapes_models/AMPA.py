import brainpy as bp
import brainpy.math as bm

from run_synapse import run_syn


class AMPA(bp.dyn.TwoEndConn):
	def __init__(self, pre, post, conn, g_max=0.6, E=0., alpha=0.98, beta=0.18,
	             T_0=0.5, T_dur=0.5, delay_step=2, method='exp_auto', **kwargs):
		super(AMPA, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
		self.check_pre_attrs('spike')
		self.check_post_attrs('input', 'V')

		# 初始化参数
		self.g_max = g_max
		self.E = E
		self.alpha = alpha
		self.beta = beta
		self.T_0 = T_0
		self.T_dur = T_dur
		self.delay_step = delay_step

		# 获取关于连接的信息
		self.pre2post = self.conn.require('pre2post')  # 获取从pre到post的连接信息

		# 初始化变量
		self.s = bm.Variable(bm.zeros(self.post.num))
		self.g = bm.Variable(bm.zeros(self.post.num))
		self.spike_arrival_time = bm.Variable(bm.ones(self.pre.num) * -1e7)  # 脉冲到达的时间
		self.delay = bm.LengthDelay(self.pre.spike, delay_step)  # 定义一个延迟处理器

		# 定义积分函数
		self.integral = bp.odeint(self.derivative, method=method)

	def derivative(self, g, t, T):
		dgdt = self.alpha * T * (1 - g) - self.beta * g
		return dgdt

	def update(self, _t, _dt):
		# 将突触前神经元传来的信号延迟delay_step的时间步长
		delayed_pre_spike = self.delay(self.delay_step)
		self.delay.update(self.pre.spike)

		# 更新脉冲到达的时间，并以此计算T
		self.spike_arrival_time.value = bm.where(delayed_pre_spike, _t, self.spike_arrival_time)
		T = ((_t - self.spike_arrival_time) < self.T_dur) * self.T_0

		# 更新s和g
		self.s.value = self.integral(self.s, _t, T, dt=_dt)
		self.g.value = self.g_max * self.s

		# 电导模式下计算突触后电流大小
		self.post.input += self.g * (self.E - self.post.V)

#
# run_syn(AMPA, title='AMPA Synapse Model')
