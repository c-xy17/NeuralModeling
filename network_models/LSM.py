import brainpy as bp
import brainpy.math as bm
import numpy as np


l1, l2, l3 = 15, 3, 3
total_num = l1 * l2 * l3
f_inh = 0.2
f_input = 0.3
lamda = 2.

# 通过神经元编号获取它在三维网格中的位置
def get_position(index):
  x1 = index // (l2 * l3)
  remain = index - x1 * l2 * l3
  x2 = remain // l3
  x3 = remain % l3
  return x1, x2, x3

# 计算神经元之间的距离
def euclidean_dist(pos1, pos2):
  pos1 = np.asarray(pos1)
  pos2 = np.asarray(pos2)
  return np.sqrt(np.sum(np.square(pos1 - pos2), axis=-1))

# 获取两群神经元的距离矩阵，矩阵形状为(len(positions1), len(positions2))
def get_dist_matrix(positions1, positions2):
  dist_matrix = []
  for pos in positions1:
    dist_matrix.append(euclidean_dist(pos, positions2))
  return np.asarray(dist_matrix)

# 根据距离矩阵计算连接概率，从而获取连接矩阵
def get_conn_matrix(dist_matrix, C, include_self=True):
  prob_matrix = np.exp(- np.square(dist_matrix / lamda)) * C
  conn_mat = np.random.random(dist_matrix.shape) < prob_matrix
  if not include_self:
    conn_mat = np.logical_and(conn_mat, np.logical_not(np.eye(dist_matrix.shape[0])))
  return conn_mat


class InputSpike(bp.dyn.NeuGroup):
  def __init__(self, size, spike_train=None):
    super(InputSpike, self).__init__(size=size)
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.spike_train = spike_train

  def get_spike_train(self, spike_train):
    self.spike_train = spike_train

  def update(self, tdi, x=None):
    t, dt = tdi['t'], tdi['dt']
    index = int(t / dt)
    if index < len(self.spike_train):
      self.spike[:] = self.spike_train[index]
    else:
      self.spike[:] = False


class LSM(bp.dyn.TrainingSystem):
  def __init__(self, num_input=1, num_readout=1, input_freq=1600.):
    super(LSM, self).__init__()

    # reservoir层的兴奋性神经元群E和抑制性神经元群I
    E = bp.dyn.LIF(size=int(total_num * (1 - f_inh)), tau=30., V_th=15., V_reset=13.5, V_rest=13.5, R=1.,
                   tau_ref=3., V_initializer=bp.init.Uniform(13.5, 15.))
    I = bp.dyn.LIF(size=int(total_num * f_inh), tau=30., V_th=15., V_reset=13.5, V_rest=13.5, R=1.,
                   tau_ref=2., V_initializer=bp.init.Uniform(13.5, 15.))

    # 将E和I排列在15x3x3的网格中，每个神经元对应一个序号
    indices = np.random.permutation(total_num)
    E_indices = indices[:int(total_num * (1 - f_inh))]
    I_indices = indices[int(total_num * (1 - f_inh)):]
    # 计算E和I在网格中的位置
    pos_E = get_position(E_indices)
    pos_E = np.asarray(pos_E).T
    pos_I = get_position(I_indices)
    pos_I = np.asarray(pos_I).T

    # 生成E2E, E2I, I2E, I2I的连接矩阵
    dist_E2E = get_dist_matrix(pos_E, pos_E)
    conn_E2E = get_conn_matrix(dist_E2E, C=0.3, include_self=False)

    dist_E2I = get_dist_matrix(pos_E, pos_I)
    conn_E2I = get_conn_matrix(dist_E2I, C=0.2)

    dist_I2E = get_dist_matrix(pos_I, pos_E)
    conn_I2E = get_conn_matrix(dist_I2E, C=0.4)

    dist_I2I = get_dist_matrix(pos_I, pos_I)
    conn_I2I = get_conn_matrix(dist_I2I, C=0.1, include_self=False)

    # 生成reservoir层的突触连接
    # todo: what is s in STP?
    E2E = bp.dyn.synapses.STP(E, E, conn=bp.conn.MatConn(conn_E2E), tau_d=.5, tau_f=1.1, U=.05,
                              A=30., tau=3., delay_step=int(1.5 / bm.get_dt()))
    E2I = bp.dyn.synapses.STP(E, I, conn=bp.conn.MatConn(conn_E2I), tau_d=.05, tau_f=.125, U=1.2,
                              A=60., tau=3., delay_step=int(0.8 / bm.get_dt()))
    I2E = bp.dyn.synapses.STP(I, E, conn=bp.conn.MatConn(conn_I2E), tau_d=.25, tau_f=.7, U=.02,
                              A=-19., tau=6., delay_step=int(0.8 / bm.get_dt()))
    I2I = bp.dyn.synapses.STP(I, I, conn=bp.conn.MatConn(conn_I2I), tau_d=.32, tau_f=.144, U=.06,
                              A=-19., tau=6., delay_step=int(0.8 / bm.get_dt()))

    # 输入神经元
    # input_neuron = InputSpike(num_input)
    input_neuron = bp.dyn.PoissonGroup(num_input, freqs=input_freq)

    # input到reservoir的连接
    input2E = bp.dyn.synapses.Delta(input_neuron, E, bp.conn.FixedProb(f_input))
    input2I = bp.dyn.synapses.Delta(input_neuron, I, bp.conn.FixedProb(f_input))

    # readout神经元
    readout = bp.dyn.LIF(num_readout, tau=30., V_th=15., V_reset=13.5, R=1., tau_ref=3.,
                         V_initializer=bp.init.Uniform(13.5, 15.))

    # reservoir到readout的连接
    E2readout = bp.dyn.ExpCOBA(E, readout, bp.conn.All2All())
    I2readout = bp.dyn.ExpCOBA(I, readout, bp.conn.All2All())

    self.input_neuron = input_neuron
    self.E = E
    self.I = I
    self.readout = readout

    self.input2E = input2E
    self.input2I = input2I

    self.E2E = E2E
    self.E2I = E2I
    self.I2E = I2E
    self.I2I = I2I

    self.E2readout = E2readout
    self.I2readout = I2readout

  # def update(self, shared_args, x):
