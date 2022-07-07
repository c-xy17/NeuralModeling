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
    self.rng = bm.random.RandomState(seed=1)

    # 初始化连接矩阵
    self.W_in = W_in_initializer(num_in, num_rec)
    conn_mat = self.rng.random(self.W_in.shape) > in_connectivity
    self.W_in[conn_mat] = 0.  # 按连接概率削减连接度

    self.W = W_rec_initializer(num_rec, num_rec)
    conn_mat = self.rng.random(self.W.shape) > rec_connectivity
    self.W[conn_mat] = 0.  # 按连接概率削减连接度

    self.W_out = bp.init.Normal()(num_rec, num_out)

    # 缩放W，使ESN具有回声性质
    spectral_radius = max(abs(bm.linalg.eig(self.W)[0]))
    self.W *= lambda_max / spectral_radius

    self.state = bm.Variable(bm.zeros(num_rec))

  def reset(self, batch_size=None):
    if batch_size is None:
      self.state.value = bm.zeros(self.state.shape)
    else:
      # todo: to be checked
      self.state.value = bm.Variable(bm.zeros((int(batch_size),) + self.state.shape), batch_axis=0)

  def update(self, tdi, u):
    self.state.value = bm.dot(u.flatten(), self.W_in) + bm.dot(self.state, self.W)
    out = bm.dot(self.state, self.W_out)
    return out

