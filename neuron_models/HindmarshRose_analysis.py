import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


class HindmarshRose(bp.NeuGroup):
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
		self.int_V = bp.odeint(f=self.dV, method='exp_auto')
		self.int_y = bp.odeint(f=self.dy, method='exp_auto')
		self.int_z = bp.odeint(f=self.dz, method='exp_auto')

	def dV(self, V, t, y, z, Iext):
		return y - self.a * V * V * V + self.b * V * V - z + Iext

	def dy(self, y, t, V):
		return self.c - self.d * V * V - y

	def dz(self, z, t, V):
		return self.r * (self.s * (V - self.V_rest) - z)

	def update(self, _t, _dt):
		V = self.int_V(self.V, _t, self.y, self.z, self.input, dt=_dt)
		y = self.int_y(self.y, _t, self.V, dt=_dt)
		z = self.int_z(self.z, _t, self.V, dt=_dt)
		self.spike.value = bm.logical_and(V >= self.V_th, self.V < self.V_th)  # 判断神经元是否发放脉冲
		self.t_last_spike.value = bm.where(self.spike, _t, self.t_last_spike)  # 更新最后一次脉冲发放时间
		self.V.value = V
		self.y.value = y
		self.z.value = z
		self.input[:] = 0.  # 重置外界输入


# 相平面分析
group = HindmarshRose(1)
phase_plane_analyzer = bp.analysis.PhasePlane2D(
	model=[group.int_V, group.int_y, group.int_z],
	target_vars={'V': [-3., 3.], 'y': [-20., 5.]},
	fixed_vars={'z': 1.4},
	pars_update={'Iext': 2.},
	resolutions=0.01
)
phase_plane_analyzer.plot_nullcline(x_style={'fmt': '-'}, y_style={'fmt': '-'})
phase_plane_analyzer.plot_fixed_point()
# phase_plane_analyzer.plot_vector_field()
phase_plane_analyzer.plot_trajectory(
	{'V': [1.], 'y': [0.], 'z': [1.4]},
	duration=100.,
	show=True
)
