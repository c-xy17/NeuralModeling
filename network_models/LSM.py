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
  def __init__(self, num_input=1, num_readout=1):
    super(LSM, self).__init__()

    self.num_input = num_input
    self.num_res = total_num
    self.num_output = num_readout

    # reservoir层的兴奋性神经元群E和抑制性神经元群I
    E = bp.dyn.LIF(size=int(total_num * (1 - f_inh)), tau=30., V_th=15., V_reset=13.5, V_rest=13.5, R=1.,
                   tau_ref=3., V_initializer=bp.init.Uniform(13.5, 15.), trainable=True)
    I = bp.dyn.LIF(size=int(total_num * f_inh), tau=30., V_th=15., V_reset=13.5, V_rest=13.5, R=1.,
                   tau_ref=2., V_initializer=bp.init.Uniform(13.5, 15.), trainable=True)

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
    E2E = bp.dyn.synapses.Exponential(E, E, conn=bp.conn.MatConn(conn_E2E), tau=3.,
                                      delay_step=int(1.5 / bm.get_dt()), g_max=30.,
                                      stp=bp.dyn.synplast.STP(tau_d=.5, tau_f=1.1, U=.05),
                                      trainable=True)
    E2I = bp.dyn.synapses.Exponential(E, I, conn=bp.conn.MatConn(conn_E2I), tau=3.,
                                      delay_step=int(0.8 / bm.get_dt()), g_max=60.,
                                      stp=bp.dyn.synplast.STP(tau_d=.05, tau_f=.125, U=0.12),
                                      trainable=True)
    I2E = bp.dyn.synapses.Exponential(I, E, conn=bp.conn.MatConn(conn_I2E), tau=6.,
                                      delay_step=int(0.8 / bm.get_dt()), g_max=-19.,
                                      stp=bp.dyn.synplast.STP(tau_d=.25, tau_f=.7, U=.02),
                                      trainable=True)
    I2I = bp.dyn.synapses.Exponential(I, I, conn=bp.conn.MatConn(conn_I2I), tau=6.,
                                      delay_step=int(0.8 / bm.get_dt()), g_max=-19.,
                                      stp=bp.dyn.synplast.STP(tau_d=.32, tau_f=.144, U=.06),
                                      trainable=True)

    # E2E = bp.dyn.synapses.STP(E, E, conn=bp.conn.MatConn(conn_E2E), tau_d=.5, tau_f=1.1, U=.05,
    #                           A=30., tau=3., delay_step=int(1.5 / bm.get_dt()), trainable=True)
    # E2I = bp.dyn.synapses.STP(E, I, conn=bp.conn.MatConn(conn_E2I), tau_d=.05, tau_f=.125, U=0.12,
    #                           A=60., tau=3., delay_step=int(0.8 / bm.get_dt()))
    # I2E = bp.dyn.synapses.STP(I, E, conn=bp.conn.MatConn(conn_I2E), tau_d=.25, tau_f=.7, U=.02,
    #                           A=-19., tau=6., delay_step=int(0.8 / bm.get_dt()))
    # I2I = bp.dyn.synapses.STP(I, I, conn=bp.conn.MatConn(conn_I2I), tau_d=.32, tau_f=.144, U=.06,
    #                           A=-19., tau=6., delay_step=int(0.8 / bm.get_dt()))

    # 输入神经元
    # i = InputSpike(num_input)
    # i = bp.dyn.PoissonGroup(num_input, freqs=input_freq)
    i = bp.neurons.InputGroup(num_input, trainable=True)

    # input到reservoir的连接
    input2E = bp.dyn.synapses.Exponential(i, E, bp.conn.FixedProb(f_input), trainable=True)
    input2I = bp.dyn.synapses.Exponential(i, I, bp.conn.FixedProb(f_input), trainable=True)

    # readout神经元
    # o = bp.dyn.LIF(num_readout, tau=30., V_th=15., V_reset=13.5, R=1., tau_ref=3.,
    #                      V_initializer=bp.init.Uniform(13.5, 15.))
    o = bp.layers.Dense(total_num, num_readout, trainable=True, fit_online=True)

    # # reservoir到readout的连接
    # E2readout = bp.dyn.ExpCOBA(E, o, bp.conn.All2All())
    # I2readout = bp.dyn.ExpCOBA(I, o, bp.conn.All2All())

    self.i = i
    self.E = E
    self.I = I
    self.o = o

    self.input2E = input2E
    self.input2I = input2I

    self.E2E = E2E
    self.E2I = E2I
    self.I2E = I2E
    self.I2I = I2I
    #
    # self.E2readout = E2readout
    # self.I2readout = I2readout

  def update(self, tdi, input_spike):
    # 更新突触连接
    self.input2E(tdi, input_spike)
    self.input2I(tdi, input_spike)
    self.E2E(tdi)
    self.E2I(tdi)
    self.I2E(tdi)
    self.I2I(tdi)

    # 更新神经元群
    self.E(tdi)
    self.I(tdi)

    res_spike = bm.concatenate([self.E.spike, self.I.spike])

    # 更新输出
    return self.o(tdi, res_spike)
