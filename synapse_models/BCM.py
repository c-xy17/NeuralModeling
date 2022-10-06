import brainpy as bp
import brainpy.math as bm

from run_synapse import run_BCM


class BCM(bp.dyn.TwoEndConn):
  def __init__(self, pre, post, conn, eta=0.01, eps=0., p=1, y_o=1., w_max=2., w_min=-2.,
               E_step=1000, delay_step=0, method='exp_auto', **kwargs):
    super(BCM, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
    self.check_pre_attrs('r', 'input')
    self.check_post_attrs('r', 'input')

    # 初始化参数
    self.eta = eta
    self.eps = eps
    self.p = p
    self.y_o = y_o
    self.w_max = w_max
    self.w_min = w_min
    self.E_step = E_step
    self.delay_step = delay_step

    # 获取每个连接的突触前神经元pre_ids和突触后神经元post_ids
    self.pre_ids, self.post_ids = self.conn.require('pre_ids', 'post_ids')

    # 初始化变量
    self.w = bm.Variable(bm.zeros(len(self.pre_ids)) + 1.)
    self.theta_M = bm.Variable(bm.zeros(self.post.num))
    self.y_history = bm.LengthDelay(self.post.r, E_step)  # 记录过去E_step内y的历史值
    self.delay = bm.LengthDelay(self.pre.r, delay_step)  # 定义一个延迟处理器

    # 定义积分函数
    self.integral = bp.odeint(self.derivative, method=method)

  def derivative(self, w, t, x, y, theta):
    dwdt = self.eta * y * (y - theta) * x - self.eps * w
    return dwdt

  def update(self, tdi):
    _t, _dt = tdi.t, tdi.dt
    # 将突触前的信号延迟delay_step的时间步长
    delayed_pre_r = self.delay(self.delay_step)
    self.delay.update(self.pre.r)

    # 更新突触后神经元的响应
    weight = delayed_pre_r[self.pre_ids.value] * self.w  # 计算每个突触i对应的突触后神经元反应y_i
    post_r = bm.syn2post_sum(weight, self.post_ids, self.post.num, )  # 每个突触后神经元k的所有y_k求和
    self.post.r.value += post_r

    current_step = bm.asarray(_t / _dt, dtype=int).value
    # 将最新的y (post.r)放进y_history
    self.y_history.update(self.post.r)
    # 如果current_step < E_step, 则只选取前current_step个数据求平均，否则选择全部
    t_step = bm.minimum(current_step + 1, self.E_step).value
    self.theta_M.value = bm.power(bm.sum(self.y_history.data, axis=0) / t_step, self.p)

    # 更新w
    w = self.integral(self.w, _t, self.pre.r[self.pre_ids], self.post.r[self.post_ids],
                      self.theta_M[self.post_ids])
    # 将w限制在[w_min, w_max]范围内
    w = bm.where(w > self.w_max, self.w_max, w)
    w = bm.where(w < self.w_min, self.w_min, w)
    self.w.value = w


dur = 200.
I1, _ = bp.inputs.constant_input([(1.5, 20.), (0., 20.)] * 5)
I2, _ = bp.inputs.constant_input([(0., 20.), (1., 20.)] * 5)
I_pre = bm.stack((I1, I2))

# dur = 200.
# I1, _ = bp.inputs.constant_input([(1., 100.), (0., dur - 100.)])
# I2, _ = bp.inputs.constant_input([(1., dur)])
# I_pre = bm.stack((I1, I2))

run_BCM(BCM, I_pre, dur, eps=0.002)
