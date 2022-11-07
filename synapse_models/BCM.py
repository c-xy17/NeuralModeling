import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from neuron_models.FRNeuron import FR

plt.rcParams.update({"font.size": 15})
plt.rcParams['font.sans-serif'] = ['Times New Roman']


class BCM(bp.dyn.DynamicalSystem):
  def __init__(self, num_pre, num_post, eta=0.01, eps=0., p=1, y_o=1.,
               w_max=2., w_min=-2., method='exp_auto'):
    super(BCM, self).__init__()

    # 初始化参数
    self.eta = eta
    self.eps = eps
    self.p = p
    self.y_o = y_o
    self.w_max = w_max
    self.w_min = w_min

    # 初始化变量
    self.pre = bm.Variable(num_pre)  # 突触前发放率
    self.post = bm.Variable(num_post)  # 突触后发放率
    self.post_mean = bm.Variable(self.post.value)  # 突触后平均发放率
    self.w = bm.Variable(bm.ones((num_pre, num_post)))  # 突触权重
    self.theta_M = bm.Variable(num_post)

    # 定义积分函数
    self.integral = bp.odeint(self.derivative, method=method)

  def derivative(self, w, t, x, y, theta):
    dwdt = self.eta * y * (y - theta) * bm.reshape(x, (-1, 1)) - self.eps * w
    return dwdt

  def update(self, tdi):
    # 更新w
    w = self.integral(self.w, tdi.t, self.pre, self.post, self.theta_M, tdi.dt)
    # 将w限制在[w_min, w_max]范围内
    w = bm.where(w > self.w_max, self.w_max, w)
    w = bm.where(w < self.w_min, self.w_min, w)
    self.w.value = w

    # 突触后发放率
    self.post.value = self.pre @ self.w

    # 更新突触后神经元平均发放率
    self.post_mean.value = (self.post_mean * (tdi.i + 1) + self.post) / (tdi.i + 2)
    self.theta_M.value = bm.power(self.post_mean, self.p)


def bcm_dw():
  eps = 0.01
  eta = 0.1
  theta = 5.
  x = 1
  w = 1
  dwdt = lambda y: eta * y * (y - theta) * x - eps * w
  ys = bm.arange(0, 8, 0.1)
  fig, gs = bp.visualize.get_figure(1, 1, 3, 6)
  ax = fig.add_subplot(gs[0, 0])
  plt.plot(bm.as_numpy(ys), bm.as_numpy(dwdt(ys)))
  plt.axhline(0., color='k', lw=1)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.set_xticks([])
  plt.text(theta, -0.3, r'$\theta_M$')
  plt.text(2.3, -0.4, 'LTD')
  plt.text(6.5, 0.4, 'LTP')
  plt.ylabel(r'$\mathrm{d}w/\mathrm{d}t$')
  ax.set_xlim(0, 8)
  plt.savefig('BCM_diagram_dwdt.pdf', transparent=True, dpi=500)
  plt.show()


def try_bcm_rule():
  dur = 200.
  I1, _ = bp.inputs.constant_input([(1.5, 20.), (0., 20.)] * 5)
  I2, _ = bp.inputs.constant_input([(0., 20.), (1., 20.)] * 5)
  I_pre = bm.stack((I1, I2)).T

  # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
  model = BCM(num_pre=2, num_post=1, eps=0.)

  # 运行模拟
  def f_input(tdi):  model.pre.value = I_pre[tdi.i]

  runner = bp.dyn.DSRunner(model, fun_inputs=f_input,
                           monitors=['pre', 'post', 'w', 'theta_M'],
                           fun_monitors={'w': lambda tdi: model.w.flatten()})
  runner.run(dur)

  # 可视化
  fig, gs = bp.visualize.get_figure(3, 1, 1.5, 8.)

  ax = fig.add_subplot(gs[0, 0])
  plt.plot(runner.mon.ts, runner.mon['pre'][:, 0], label='$\mathrm{pre}_0$', )
  plt.plot(runner.mon.ts, runner.mon['pre'][:, 1], label='$\mathrm{pre}_1$', linestyle='--')
  plt.legend(loc='center right')
  plt.xticks([])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  ax = fig.add_subplot(gs[1, 0])
  plt.plot(runner.mon.ts, runner.mon['post'][:, 0], label=r'$\mathrm{post}_0$', )
  plt.plot(runner.mon.ts, runner.mon['theta_M'], label=r'$\theta_\mathrm{M}$', linestyle='dashed')
  plt.legend(loc='center right')
  plt.xticks([])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  ax = fig.add_subplot(gs[2, 0])
  plt.plot(runner.mon.ts, runner.mon['w'][:, 0], label='$w_0$')
  plt.plot(runner.mon.ts, runner.mon['w'][:, 1], label='$w_1$', linestyle='--')
  plt.legend(loc='center right')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  plt.xlabel(r'$t$ (ms)')
  plt.savefig('BCM_output1.pdf', transparent=True, dpi=500)
  plt.show()


if __name__ == '__main__':
  # bcm_dw()
  try_bcm_rule()
