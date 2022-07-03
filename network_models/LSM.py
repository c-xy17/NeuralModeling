import brainpy as bp
import brainpy.math as bm
import numpy as np


l1, l2, l3 = 15, 3, 3
total_num = l1 * l2 * l3
f_inh = 0.2
lamda = 2.

def get_position(index):
  x1 = index // (l2 * l3)
  remain = index - x1 * l2 * l3
  x2 = remain // l3
  x3 = remain % l3
  return x1, x2, x3

def euclidean_dist(pos1, pos2):
  pos1 = np.asarray(pos1)
  pos2 = np.asarray(pos2)
  return np.sqrt(np.sum(np.square(pos1 - pos2)), axis=-1)

def get_dist_matrix(positions1, positions2):
  dist_matrix = []
  for pos in positions1:
    dist_matrix.append(euclidean_dist(pos, positions2))
  return np.asarray(dist_matrix)

def get_conn_matrix(dist_matrix, C):
  prob_matrix = np.exp(- np.square(dist_matrix / lamda)) * C
  return np.random.random(dist_matrix.shape) < prob_matrix


E = bp.dyn.LIF(size=int(total_num * (1 - f_inh)), tau=30., V_th=15., V_reset=13.5, R=1., tau_ref=3.)
I = bp.dyn.LIF(size=int(total_num * f_inh), tau=30., V_th=15., V_reset=13.5, R=1., tau_ref=2.)

# 将E和I排列在15x3x3的网格中
indices = np.random.permutation(total_num)
E_indices = indices[:int(total_num * (1 - f_inh))]
I_indices = indices[int(total_num * (1 - f_inh)):]

pos_E = get_position(E_indices)
pos_E = np.asarray(pos_E).T
pos_I = get_position(I_indices)
pos_I = np.asarray(pos_I).T

dist_E2E = get_dist_matrix(pos_E, pos_E)
conn_E2E = get_conn_matrix(dist_E2E, C=0.3)

dist_E2I = get_dist_matrix(pos_E, pos_I)
conn_E2I = get_conn_matrix(dist_E2I, C=0.2)

dist_I2E = get_dist_matrix(pos_I, pos_E)
conn_I2E = get_conn_matrix(dist_I2E, C=0.4)

dist_I2I = get_dist_matrix(pos_I, pos_I)
conn_I2I = get_conn_matrix(dist_I2I, C=0.1)

# todo: what is s in STP?
E2E = bp.dyn.synapses.STP(E, E, conn=bp.conn.MatConn(conn_E2E), tau_d=.5, tau_f=1.1, U=.05,
                          A=30., tau=3., delay_step=int(1.5/bm.get_dt()))
E2I = bp.dyn.synapses.STP(E, I, conn=bp.conn.MatConn(conn_E2I), tau_d=.05, tau_f=.125, U=1.2,
                          A=60., tau=3., delay_step=int(0.8/bm.get_dt()))
I2E = bp.dyn.synapses.STP(E, I, conn=bp.conn.MatConn(conn_I2E), tau_d=.25, tau_f=.7, U=.02,
                          A=-19., tau=6., delay_step=int(0.8/bm.get_dt()))
I2I = bp.dyn.synapses.STP(E, I, conn=bp.conn.MatConn(conn_I2I), tau_d=.32, tau_f=.144, U=.06,
                          A=-19., tau=6., delay_step=int(0.8/bm.get_dt()))


class Reservoir(bp.dyn.NeuGroup):
  def __init__(self, size, I_b=13.5):
    super(Reservoir, self).__init__(size=size)


class LSM(bp.dyn.TrainingSystem):
  def __init__(self, num_in, num_hidden, num_out):
    super(LSM, self).__init__()



    self.r = bp.layers.Reservoir(num_in, num_hidden,
                                 Win_initializer=bp.init.Uniform(-0.1, 0.1),
                                 Wrec_initializer=bp.init.Normal(scale=0.1),
                                 in_connectivity=0.02,
                                 rec_connectivity=0.02,
                                 conn_type='dense')
    self.o = bp.layers.Dense(num_hidden, num_out, W_initializer=bp.init.Normal())

  def update(self, shared_args, x):
    return self.o(shared_args, self.r(shared_args, x))
