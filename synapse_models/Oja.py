import brainpy as bp
import brainpy.math as bm

# from run_synapse import run_FR


class Oja(bp.dyn.TwoEndConn):
  def __init__(self, pre, post, conn, eta=0.05, delay_step=0, method='exp_auto', **kwargs):
    super(Oja, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
    self.check_pre_attrs('r', 'input')
    self.check_post_attrs('r', 'input')

    # 初始化参数
    self.eta = eta
    self.delay_step = delay_step

    # 获取每个连接的突触前神经元pre_ids和突触后神经元post_ids
    self.pre_ids, self.post_ids = self.conn.require('pre_ids', 'post_ids')

    # 初始化变量
    num = len(self.pre_ids)
    # self.w = bm.Variable(bm.zeros(num) + 1. / bm.sqrt(num))
    self.w = bm.Variable(bm.random.uniform(size=num) * 2./bm.sqrt(num))
    self.delay = bm.LengthDelay(self.pre.r, delay_step)  # 定义一个延迟处理器

    # 定义积分函数
    self.integral = bp.odeint(self.derivative, method=method)

  def derivative(self, w, t, x, y):
    dwdt = self.eta * y * (x - y * w)
    return dwdt

  def update(self, _t, _dt):
    # 将突触前的信号延迟delay_step的时间步长
    delayed_pre_r = self.delay(self.delay_step)
    self.delay.update(self.pre.r)

    # 更新突触后的firing rate
    weight = delayed_pre_r[self.pre_ids.value] * self.w  # 计算每个突触i对应的突触后神经元反应y_i
    post_r = bm.syn2post_sum(weight, self.post_ids, self.post.num, )  # 每个突触后神经元k的所有y_k求和
    self.post.r.value += post_r

    # 更新w
    self.w.value = self.integral(self.w, _t, self.pre.r[self.pre_ids], self.post.r[self.post_ids])


# # 自定义电流
# dur = 200.
# I1, _ = bp.inputs.constant_input([(1., 100.), (0., dur - 100.)])
# I2, _ = bp.inputs.constant_input([(1., dur)])
# I_pre = bm.stack((I1, I2))
#
# run_FR(Oja, I_pre, dur)