import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt


LIF = bp.dyn.LIF
ExpCOBA = bp.dyn.ExpCOBA


class EINet(bp.dyn.Network):
  def __init__(self, num_exc, num_inh, method='exp_auto', **kwargs):
    super(EINet, self).__init__(**kwargs)

    # 搭建神经元
    pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.)
    E = LIF(num_exc, **pars, method=method)
    I = LIF(num_inh, **pars, method=method)
    E.V.value = bm.random.randn(num_exc) * 4. - 60.  # 随机初始化膜电位
    I.V.value = bm.random.randn(num_inh) * 4. - 60.  # 随机初始化膜电位
    self.E = E
    self.I = I

    # 搭建神经元连接
    E_pars = dict(E=0., g_max=0.6, tau=5.)
    I_pars = dict(E=-80., g_max=6.7, tau=10.)
    self.E2E = ExpCOBA(E, E, bp.conn.FixedProb(prob=0.02), **E_pars, method=method)
    self.E2I = ExpCOBA(E, I, bp.conn.FixedProb(prob=0.02), **E_pars, method=method)
    self.I2E = ExpCOBA(I, E, bp.conn.FixedProb(prob=0.02), **I_pars, method=method)
    self.I2I = ExpCOBA(I, I, bp.conn.FixedProb(prob=0.02), **I_pars, method=method)


