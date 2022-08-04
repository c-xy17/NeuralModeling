import brainpy as bp
import brainpy.math as bm

from run_synapse import run_syn_NMDA


# second-order kinetics
class NMDA(bp.dyn.TwoEndConn):
	def __init__(self, pre, post, conn, g_max=0.02, E=0., c_Mg=1.2, alpha1=2.,
	             beta1=0.01, alpha2=0.2, beta2=0.5, delay_step=2,
	             method='exp_auto', **kwargs):
		super(NMDA, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
		self.check_pre_attrs('spike')
		self.check_post_attrs('input', 'V')

		# 初始化参数
		self.g_max = g_max
		self.E = E
		self.c_Mg = c_Mg
		self.alpha1 = alpha1
		self.beta1 = beta1
		self.alpha2 = alpha2
		self.beta2 = beta2
		self.delay_step = delay_step

		# 获取关于连接的信息
		self.pre2post = self.conn.require('pre2post')  # 获取从pre到post的连接信息

		# 初始化变量
		self.x = bm.Variable(bm.zeros(self.post.num))
		self.s = bm.Variable(bm.zeros(self.post.num))
		self.g = bm.Variable(bm.zeros(self.post.num))
		self.b = bm.Variable(bm.zeros(self.post.num))
		self.delay = bm.LengthDelay(self.pre.spike, delay_step)  # 定义一个延迟处理器

		# 定义积分函数
		self.int_s = bp.odeint(method=method,
		                       f=lambda s, t, x: self.alpha1 * x * (1 - s) - self.beta1 * s)
		self.int_x = bp.odeint(method=method, f=lambda x, t,: - self.beta2 * x)

	def update(self, tdi):
		# 将突触前神经元传来的信号延迟delay_step的时间步长
		delayed_pre_spike = self.delay(self.delay_step)
		self.delay.update(self.pre.spike)

		# 根据连接模式计算各个突触后神经元收到的信号强度
		post_sp = bm.pre2post_event_sum(delayed_pre_spike, self.pre2post, self.post.num, 1.)

		# 更新x，s和g
		self.x.value = self.int_x(self.x, tdi.t) + self.alpha2 * post_sp
		self.s.value = self.int_s(self.s, tdi.t, self.x, tdi.t)
		self.g.value = self.g_max * self.s

		# 更新b
		self.b.value = 1 / (1 + bm.exp(-0.062 * self.post.V) * self.c_Mg / 3.57)

		# 电导模式下计算突触后电流大小
		self.post.input += self.g * self.b * (self.E - self.post.V)


run_syn_NMDA(NMDA, title='NMDA Synapse Model (Kinetic)')
