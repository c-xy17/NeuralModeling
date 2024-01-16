import brainpy as bp
import brainpy.math as bm
import brainpy_datasets as bd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 15})
plt.rcParams['font.sans-serif'] = ['Times New Roman']


class ESN(bp.DynamicalSystem):
  def __init__(self, num_in, num_rec, num_out, lambda_max=0.9,
               W_in_initializer=bp.init.Uniform(-0.1, 0.1, seed=345),
               W_rec_initializer=bp.init.Normal(scale=0.1, seed=456),
               in_connectivity=0.05, rec_connectivity=0.05):
    super(ESN, self).__init__(mode=bm.batching_mode)

    self.num_in = num_in
    self.num_rec = num_rec
    self.num_out = num_out
    self.rng = bm.random.RandomState(1)  # 随机数生成器

    # 初始化连接矩阵
    self.W_in = W_in_initializer((num_in, num_rec))
    conn_mat = self.rng.random((num_in, num_rec)) > in_connectivity
    self.W_in = bm.where(conn_mat, 0., self.W_in)  # 按连接概率削减连接度

    self.W = W_rec_initializer((num_rec, num_rec))
    conn_mat = self.rng.random(self.W.shape) > rec_connectivity
    self.W = bm.where(conn_mat, 0., self.W)  # 按连接概率削减连接度

    # 用BrainPy库里的Dense作为库到输出的全连接层
    self.readout = bp.layers.Dense(num_rec, num_out,
                                   W_initializer=bp.init.Normal(),
                                   mode=bm.training_mode)

    # 缩放W，使ESN具有回声性质
    spectral_radius = max(abs(bm.linalg.eig(self.W)[0]))
    self.W *= lambda_max / spectral_radius

    # 初始化变量
    self.state = bm.Variable(bm.zeros((1, num_rec)), batch_axis=0)
    self.y = bm.Variable(bm.zeros((1, num_out)), batch_axis=0)

  # 重置函数：重置模型中各变量的值
  def reset(self, batch_size=None):
    if batch_size is None:
      self.state.value = bm.zeros(self.state.shape)
      self.y.value = bm.zeros(self.y.shape)
    else:
      self.state.value = bm.zeros((int(batch_size),) + self.state.shape[1:])
      self.y.value = bm.zeros((int(batch_size),) + self.y.shape[1:])

  def update(self, u):
    self.state.value = bm.tanh(bm.dot(u, self.W_in) + bm.dot(self.state, self.W))
    out = self.readout(self.state.value)
    self.y.value = out
    return out


def show_ESN_property():
  num_in = 10
  num_res = 500
  num_out = 30
  num_step = 500  # 模拟总步长
  num_batch = 1

  # 生成网络，运行两次模拟，两次模拟的输入相同，但网络的初始化状态不同
  def get_esn_states(lambda_max):
    model = ESN(num_in, num_res, num_out, lambda_max=lambda_max)
    model.reset(batch_size=num_batch)

    inputs = bm.random.randn(num_batch, int(num_step / num_batch), num_in)  # 第0个维度为batch的大小

    # 第一次运行
    model.state.value = bp.init.Uniform(-1., 1., seed=123)((num_batch, num_res))  # 随机初始化网络状态
    runner = bp.DSTrainer(model, monitors=['state'])
    runner.predict(inputs)
    state1 = np.concatenate(runner.mon['state'], axis=0)

    # 第二次运行
    model.state.value = bp.init.Uniform(-1., 1., seed=234)((num_batch, num_res))  # 再次随机初始化网络状态
    runner = bp.DSTrainer(model, monitors=['state'])
    runner.predict(inputs)
    state2 = np.concatenate(runner.mon['state'], axis=0)

    return state1, state2

  # 画出两次模拟中某一时刻网络的状态
  def plot_states(state1, state2, title):
    assert len(state1) == len(state2)
    x = np.arange(len(state1))
    plt.plot(x, state1, marker='.', markersize=4, linestyle='', label='first state')
    plt.plot(x, state2, marker='+', markersize=4, linestyle='', label='last state')
    plt.legend(loc='upper right')
    plt.xlabel('Neuron index')
    plt.ylabel('State')
    plt.title(title)

  bm.random.seed(23545)

  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 4)
  ax = fig.add_subplot(gs[0, 0])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  lambda1, lambda2, lambda3 = 0.9, 1.0, 1.1
  # 画出每个lambda_max下两次模拟的网络状态的距离随时间的变化
  state1, state2 = get_esn_states(lambda_max=lambda1)
  distance = np.sqrt(np.sum(np.square(state1 - state2), axis=1))
  plt.plot(np.arange(num_step), distance, label='$|\lambda_{}|={}$'.format('{max}', lambda1))
  plt.annotate('$|\lambda_{}|={}$'.format('{max}', lambda1),
               xy=(22, 0.4), xytext=(60, 4.), arrowprops=dict(arrowstyle="->"))

  state3, state4 = get_esn_states(lambda_max=lambda2)
  distance = np.sqrt(np.sum(np.square(state3 - state4), axis=1))
  plt.plot(np.arange(num_step), distance, label='$|\lambda_{}|={}$'.format('{max}', lambda2))
  plt.annotate('$|\lambda_{}|={}$'.format('{max}', lambda2),
               xy=(84.5, 0.4), xytext=(150, 1.7), arrowprops=dict(arrowstyle="->"))

  state5, state6 = get_esn_states(lambda_max=lambda3)
  distance = np.sqrt(np.sum(np.square(state5 - state6), axis=1))
  plt.plot(np.arange(num_step), distance, label='$|\lambda_{}|={}$'.format('{max}', lambda3))
  plt.text(337, 8.7, '$|\lambda_{}|={}$'.format('{max}', lambda3))

  plt.xlabel('Running step')
  plt.ylabel('Distance')
  # plt.savefig('ESN_state_property1.pdf', transparent=True, dpi=500)

  # 画出两次模拟时网络的初始状态和最终状态
  fig, gs = bp.visualize.get_figure(2, 1, 2.25, 4)
  ax = fig.add_subplot(gs[0, 0])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plot_states(state1[0], state2[0], title='$|\lambda_{}|={}, n=0$'.format('{max}', lambda1))
  ax.set_xticks([])
  ax.set_xlabel('')
  ax = fig.add_subplot(gs[1, 0])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plot_states(state1[-1], state2[-1], title='$|\lambda_{}|={}, n={}$'.format('{max}', lambda1, num_step))
  # plt.savefig('ESN_state_property2.pdf', transparent=True, dpi=500)

  fig, gs = bp.visualize.get_figure(2, 1, 2.25, 4)
  ax = fig.add_subplot(gs[0, 0])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plot_states(state5[0], state6[0], title='$|\lambda_{}|={}, n=0$'.format('{max}', lambda3))
  ax.set_xticks([])
  ax.set_xlabel('')
  ax = fig.add_subplot(gs[1, 0])
  plot_states(state5[-1], state6[-1], title='$|\lambda_{}|={}, n={}$'.format('{max}', lambda3, num_step))
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  # plt.savefig('ESN_state_property3.pdf', transparent=True, dpi=500)

  plt.show()


def fit_sine_wave():
  bm.enable_x64()  # 使用更高精度的float以提高训练精度

  num_in, num_res, num_out = 1, 600, 1
  num_step = 1000  # 模拟总步长
  num_discard = 200  # 训练时，丢弃掉前200个数据

  def plot_result(output, Y, title):
    assert output.shape == Y.shape
    x = np.arange(output.shape[0])
    plt.plot(x, Y, linestyle='--', label='$y$')
    plt.plot(x, output, label='$\hat{y}$')
    plt.legend()
    plt.xlabel('Running step')
    plt.ylabel('State')
    plt.title(title)

  # 生成训练数据
  n = bm.linspace(0., bm.pi, num_step)
  U = bm.sin(10 * n) + bm.random.normal(scale=0.1, size=num_step)  # 输入
  U = U.reshape((1, -1, num_in))  # 维度：(num_batch, num_step, num_dim)
  Y = bm.power(bm.sin(10 * n), 7)  # 输出
  Y = Y.reshape((1, -1, num_out))  # 维度：(num_batch, num_step, num_dim)

  model = ESN(num_in, num_res, num_out, lambda_max=0.95)

  # 训练前，运行模型得到结果
  runner = bp.DSTrainer(model, monitors=['state'])
  untrained_out = runner.predict(U)
  print(bp.losses.mean_absolute_error(untrained_out[:, num_discard:], Y[:, num_discard:]))

  # 用岭回归法进行训练
  trainer = bp.RidgeTrainer(model, alpha=1e-12)
  trainer.fit([U[:, num_discard:], Y[:, num_discard:]])

  # 训练后，运行模型得到结果
  runner = bp.DSTrainer(model, monitors=['state'])
  out = runner.predict(U)
  print(bp.losses.mean_absolute_error(out[:, num_discard:], Y[:, num_discard:]))

  # 可视化
  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
  ax = fig.add_subplot(gs[0, 0])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plot_result(untrained_out.flatten()[num_discard:], Y.flatten()[num_discard:], 'Before training')
  # plt.savefig('ESN_fit_sine_wave1.pdf', transparent=True, dpi=500)

  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
  ax = fig.add_subplot(gs[0, 0])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plot_result(out.flatten()[num_discard:], Y.flatten()[num_discard:], 'After training')
  # plt.savefig('ESN_fit_sine_wave2.pdf', transparent=True, dpi=500)
  plt.show()

  max_ = 0
  rng = np.random.RandomState(12354)
  i1, i2, i3, i4 = tuple(rng.choice(np.arange(num_res), 4, replace=False))
  fig, gs = bp.visualize.get_figure(1, 3, 3, 4)
  state = runner.mon['state'].squeeze()
  ax1 = fig.add_subplot(gs[0, 0])
  plt.plot(np.arange(num_step - num_discard), state[num_discard:, i1])
  plt.title('Neuron {}'.format(i1))
  plt.xlabel('Running step')
  ax1.spines['top'].set_visible(False)
  ax1.spines['right'].set_visible(False)
  if max_ < state[num_discard:, i1].max():
    max_ = state[num_discard:, i1].max()

  ax2 = fig.add_subplot(gs[0, 1])
  plt.plot(np.arange(num_step - num_discard), state[num_discard:, i2])
  plt.title('Neuron {}'.format(i2))
  plt.xlabel('Running step')
  ax2.spines['top'].set_visible(False)
  ax2.spines['right'].set_visible(False)
  if max_ < state[num_discard:, i2].max():
    max_ = state[num_discard:, i2].max()

  ax3 = fig.add_subplot(gs[0, 2])
  plt.plot(np.arange(num_step - num_discard), state[num_discard:, i3])
  plt.title('Neuron {}'.format(i3))
  plt.xlabel('Running step')
  ax3.spines['top'].set_visible(False)
  ax3.spines['right'].set_visible(False)
  if max_ < state[num_discard:, i3].max():
    max_ = state[num_discard:, i3].max()

  max_ *= 1.1
  ax1.set_ylim(-max_, max_)
  ax2.set_ylim(-max_, max_)
  ax3.set_ylim(-max_, max_)

  # plt.savefig('ESN_fit_sine_wave_example_neurons.pdf')
  plt.show()


def fit_Lorenz_system():
  bm.enable_x64()

  # 从brainpy中获取劳伦兹系统的数据
  lorenz = bd.chaos.LorenzEq(100)
  data = bm.hstack([lorenz.xs, lorenz.ys, lorenz.zs])

  X, Y = data[:-200], data[200:]  # Y比X提前200个步长，即需要预测系统未来的Y
  # 将第0维扩展为batch的维度
  X = bm.expand_dims(X, axis=0)
  Y = bm.expand_dims(Y, axis=0)

  num_in, num_res, num_out = 3, 200, 3
  num_discard = 50

  model = ESN(num_in, num_res, num_out, lambda_max=0.9)

  def training_lorenz(trainer, title, name=None):
    trainer.fit([X[:, :30000, :], Y[:, :30000, :]])  # 用前30000个时间的数据来训练

    predict = trainer.predict(X, reset_state=True)
    predict = bm.as_numpy(predict)

    fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
    ax = fig.add_subplot(gs[:, 0], projection='3d')
    # 画图时舍去最初20个步长的数据，下同
    plt.plot(Y[0, num_discard:, 0],
             Y[0, num_discard:, 1],
             Y[0, num_discard:, 2],
             alpha=0.8, label='standard output', linestyle='--')
    plt.plot(predict[0, num_discard:, 0],
             predict[0, num_discard:, 1],
             predict[0, num_discard:, 2],
             alpha=0.8, label='prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(title)
    plt.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # if name: plt.savefig(f'{name}_3d.pdf', transparent=True, dpi=500)

    fig, gs = bp.visualize.get_figure(2, 1, 2.25, 6)
    ax = fig.add_subplot(gs[0, 0])
    t = np.arange(Y.shape[1])[num_discard:]
    plt.plot(t, Y[0, num_discard:, 0], linewidth=1, label='standard $x$', linestyle='--')  # 劳伦兹系统中的x变量
    plt.plot(t, predict[0, num_discard:, 0], linewidth=1, label='predicted $x$')
    plt.ylabel(r'$x$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    plt.legend()

    ax = fig.add_subplot(gs[1, 0])
    plt.plot(t, Y[0, num_discard:, 2], linewidth=1, label='standard $z$', linestyle='--')  # 劳伦兹系统中的z变量
    plt.plot(t, predict[0, num_discard:, 2], linewidth=1, label='predicted $z$')
    plt.ylabel(r'$z$')
    plt.xlabel('Time step')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()

    # if name: plt.savefig(f'{name}_xz.pdf', transparent=True, dpi=500)
    plt.show()

  # 用岭回归法训练
  ridge_trainer = bp.OfflineTrainer(model, fit_method=bp.algorithms.RidgeRegression(alpha=1e-6))
  training_lorenz(ridge_trainer, 'Training with Ridge Regression', name='ESN_lorenz_ridge')

  # 用FORCE学习法训练
  force_trainer = bp.OnlineTrainer(model, fit_method=bp.algorithms.RLS(alpha=0.1))
  training_lorenz(force_trainer, 'Training with Force Learning', name='ESN_lorenz_force')


class ESNv2(bp.DynamicalSystem):
  def __init__(self, num_in, num_hidden, num_out, lambda_max=None):
    super(ESNv2, self).__init__(mode=bm.batching_mode)
    self.r = bp.dyn.Reservoir(num_in, num_hidden,
                              Win_initializer=bp.init.Uniform(-0.1, 0.1),
                              Wrec_initializer=bp.init.Normal(scale=0.1),
                              in_connectivity=0.02,
                              rec_connectivity=0.05,
                              spectral_radius=lambda_max,
                              comp_type='dense',
                              mode=bm.batching_mode)
    self.o = bp.dnn.Dense(num_hidden, num_out, W_initializer=bp.init.Normal(),
                          mode=bm.training_mode)

  def update(self, x):
    return self.o(self.r(x))


def train_esn_with_ridge(num_in=100, num_out=30):
  model = ESNv2(num_in, 2000, num_out)

  # input-output
  print(model(dict(), bm.ones((1, num_in))))

  X = bm.random.random((1, 200, num_in))
  Y = bm.random.random((1, 200, num_out))

  # prediction
  runner = bp.DSTrainer(model, monitors=['r.state'])
  outputs = runner.predict(X)
  print(runner.mon['r.state'].shape)
  print(bp.losses.mean_absolute_error(outputs, Y))
  print()

  # training
  trainer = bp.RidgeTrainer(model)
  trainer.fit([X, Y])

  # prediction
  runner = bp.DSTrainer(model, monitors=['r.state'])
  outputs = runner.predict(X)
  print(runner.mon['r.state'].shape)
  print(bp.losses.mean_absolute_error(outputs, Y))
  print()

  outputs = trainer.predict(X)
  print(bp.losses.mean_absolute_error(outputs, Y))


if __name__ == '__main__':
  # show_ESN_property()
  # fit_sine_wave()
  fit_Lorenz_system()
  # train_esn_with_ridge(10, 30)
