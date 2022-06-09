import brainpy as bp
import brainpy.math as bm


class CANN1D(bp.dyn.NeuGroup):
  def __init__(self, num, tau=1., k=8.1, a=0.5, A=10., J0=4.,
               z_min=-bm.pi, z_max=bm.pi, **kwargs):
    super(CANN1D, self).__init__(size=num, **kwargs)

    # 初始化参数
    self.tau = tau
    self.k = k
    self.a = a
    self.A = A
    self.J0 = J0

    # 初始化特征空间相关参数
    self.z_min = z_min
    self.z_max = z_max
    self.z_range = z_max - z_min
    self.x = bm.linspace(z_min, z_max, num)
    self.rho = num / self.z_range
    self.dx = self.z_range / num

    # 初始化变量
    self.u = bm.Variable(bm.zeros(num))
    self.input = bm.Variable(bm.zeros(num))
    self.conn_mat = self.make_conn(self.x)  # 连接矩阵

    # 定义积分函数
    self.integral = bp.odeint(self.derivative)

  # 微分方程
  def derivative(self, u, t, Iext):
    u2 = bm.square(u)
    r = u2 / (1.0 + self.k * bm.sum(u2))
    Irec = bm.dot(self.conn_mat, r)
    du = (-u + Irec + Iext) / self.tau
    return du

  # 将距离转换到[-z_range/2, z_range/2)之间
  def dist(self, d):
    d = bm.remainder(d, self.z_range)
    d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
    return d

  # 计算连接矩阵
  def make_conn(self, x):
    assert bm.ndim(x) == 1
    x_left = bm.reshape(x, (-1, 1))
    x_right = bm.repeat(x.reshape((1, -1)), len(x), axis=0)
    d = self.dist(x_left - x_right)  # 距离矩阵
    Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)
    return Jxx

  # 获取各个神经元到pos处神经元的输入
  def get_stimulus_by_pos(self, pos):
    return self.A * bm.exp(-0.25 * bm.square(self.dist(self.x - pos) / self.a))

  def update(self, _t, _dt):
    self.u[:] = self.integral(self.u, _t, self.input)
    self.input[:] = 0.  # 重置外部电流


# class CANN1D(bp.dyn.NeuGroup):
#   def __init__(self, num, tau=1., k=8.1, a=0.5, A=10., J0=4.,
#                z_min=-bm.pi, z_max=bm.pi, **kwargs):
#     super(CANN1D, self).__init__(size=num, **kwargs)
#
#     # parameters
#     self.tau = tau  # The synaptic time constant
#     self.k = k  # Degree of the rescaled inhibition
#     self.a = a  # Half-width of the range of excitatory connections
#     self.A = A  # Magnitude of the external input
#     self.J0 = J0  # maximum connection value
#
#     # feature space
#     self.z_min = z_min
#     self.z_max = z_max
#     self.z_range = z_max - z_min
#     self.x = bm.linspace(z_min, z_max, num)  # The encoded feature values
#     self.rho = num / self.z_range  # The neural density
#     self.dx = self.z_range / num  # The stimulus density
#
#     # variables
#     self.u = bm.Variable(bm.zeros(num))
#     self.input = bm.Variable(bm.zeros(num))
#
#     # The connection matrix
#     self.conn_mat = self.make_conn(self.x)
#
#     # function
#     self.integral = bp.odeint(self.derivative)
#
#   def derivative(self, u, t, Iext):
#     r1 = bm.square(u)
#     r2 = 1.0 + self.k * bm.sum(r1)
#     r = r1 / r2
#     Irec = bm.dot(self.conn_mat, r)
#     du = (-u + Irec + Iext) / self.tau
#     return du
#
#   def dist(self, d):
#     d = bm.remainder(d, self.z_range)
#     d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
#     return d
#
#   def make_conn(self, x):
#     assert bm.ndim(x) == 1
#     x_left = bm.reshape(x, (-1, 1))
#     x_right = bm.repeat(x.reshape((1, -1)), len(x), axis=0)
#     d = self.dist(x_left - x_right)
#     Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / \
#           (bm.sqrt(2 * bm.pi) * self.a)
#     return Jxx
#
#   def get_stimulus_by_pos(self, pos):
#     return self.A * bm.exp(-0.25 * bm.square(self.dist(self.x - pos) / self.a))
#
#   def update(self, _t, _dt):
#     self.u[:] = self.integral(self.u, _t, self.input)
#     self.input[:] = 0.
