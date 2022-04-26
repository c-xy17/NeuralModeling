import brainpy as bp
import brainpy.math as bm

from run_synapse import run_FR


class BCM(bp.dyn.TwoEndConn):
  def __init__(self, pre, post, conn, eta=0.05, eps=0., p=1, y_o=1., E_step=1000,
               delay_step=0, method='exp_auto', **kwargs):
    super(BCM, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
    self.check_pre_attrs('r', 'input')
    self.check_post_attrs('r', 'input')

    # 初始化参数
    self.eta = eta
    self.eps = eps
    self.p = p
    self.y_o = y_o
    self.E_step = E_step
    self.delay_step = delay_step

    # 获取每个连接的突触前神经元pre_ids和突触后神经元post_ids
    self.pre_ids, self.post_ids = self.conn.require('pre_ids', 'post_ids')

    # 初始化变量
    self.w = bm.Variable(bm.zeros(len(self.pre_ids)) + 1.)
    self.y_history = bm.Variable(bm.zeros((E_step, self.post.num)))  # 记录过去E_step内y的历史值
    self.y_sum = bm.Variable(bm.zeros(self.post.num))
    self.current_step = 0  # 当前步长
    self.delay = bm.LengthDelay(self.pre.r, delay_step)  # 定义一个延迟处理器

    # 定义积分函数
    self.integral = bp.odeint(self.derivative, method=method)

  def derivative(self, w, t, x, y, theta):
    dwdt = self.eta * y * (y - theta) * x - self.eps * w
    return dwdt

  def update(self, _t, _dt):
    # 将突触前的信号延迟delay_step的时间步长
    delayed_pre_r = self.delay(self.delay_step)
    self.delay.update(self.pre.r)

    # 更新突触后神经元的响应
    weight = delayed_pre_r[self.pre_ids.value] * self.w  # 计算每个突触i对应的突触后神经元反应y_i
    post_r = bm.syn2post_sum(weight, self.post_ids, self.post.num, )  # 每个突触后神经元k的所有y_k求和
    self.post.r.value += post_r

    # 更新theta_M
    self.y_sum += self.post.r
    self.current_step += 1
    theta_M = bm.Variable(self.y_sum / self.current_step)

    # self.y_history[self.current_step % self.E_step] = self.post.r
    # # 如果current_step < E_step, 则只选取前current_step个数据求平均，否则选择全部
    # t_step = min(self.current_step+1, self.E_step)
    # theta_M = bm.power(bm.mean(self.y_history[:t_step] / self.y_o, axis=0), self.p)
    # self.current_step += 1

    # 更新w
    w = self.integral(self.w, _t, self.pre.r[self.pre_ids], self.post.r[self.post_ids],
                                 theta_M[self.post_ids])
    self.w.value = w


dur = 200.
I1, _ = bp.inputs.constant_input([(1.5, 10.), (0., 10.)] * 10)
I2, _ = bp.inputs.constant_input([(0., 10.), (1., 10.)] * 10)
I_pre = bm.stack((I1, I2))

run_FR(BCM, I_pre, dur)