import brainpy as bp
import brainpy.math as bm


class ESN(bp.dyn.TrainingSystem):
  def __init__(self, num_in, num_rec, num_out, lambda_max=0.9,
               W_in_initializer=bp.init.Uniform(-0.1, 0.1),
               W_rec_initializer=bp.init.Normal(scale=0.1),
               in_connectivity=0.02, rec_connectivity=0.02):
    super(ESN, self).__init__()

    self.num_in = num_in
    self.num_rec = num_rec
    self.num_out = num_out
    self.rng = bm.random.RandomState(seed=1)  # 随机数生成器

    # 初始化连接矩阵
    self.W_in = W_in_initializer((num_in, num_rec))
    conn_mat = self.rng.random(self.W_in.shape) > in_connectivity
    self.W_in[conn_mat] = 0.  # 按连接概率削减连接度

    self.W = W_rec_initializer((num_rec, num_rec))
    conn_mat = self.rng.random(self.W.shape) > rec_connectivity
    self.W[conn_mat] = 0.  # 按连接概率削减连接度

    # 用BrainPy库里的Dense作为库到输出的全连接层
    self.readout = bp.layers.Dense(num_rec, num_out, W_initializer=bp.init.Normal())

    # 缩放W，使ESN具有回声性质
    spectral_radius = max(abs(bm.linalg.eig(self.W)[0]))
    self.W *= lambda_max / spectral_radius

    # 初始化变量
    self.state = bm.Variable(bm.zeros((1, num_rec)), batch_axis=0)
    self.y = bm.Variable(bm.zeros((1, num_out)), batch_axis=0)

  def reset(self, batch_size=None):
    if batch_size is None:
      self.state.value = bm.zeros(self.state.shape)
      self.y.value = bm.zeros(self.y.shape)
    else:
      self.state.value = bm.zeros((int(batch_size),) + self.state.shape)

  def update(self, sha, u):
    self.state.value = bm.tanh(bm.dot(u, self.W_in) + bm.dot(self.state, self.W))
    out = self.readout(sha, self.state.value)
    self.y.value = out
    return out
