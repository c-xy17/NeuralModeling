import brainpy as bp
import brainpy.math as bm

from dual_exponential import DualExponential
from run_synapse import run_syn


class Alpha(DualExponential):
  def __init__(self, pre, post, conn, g_max=0.01, tau=6., delay_step=2, E=0.,
               syn_type='CUBA', method='exp_auto', **kwargs):
    super(Alpha, self).__init__(pre=pre, post=post, conn=conn,
                                g_max=g_max, tau_decay=tau, tau_rise=tau,
                                E=E, delay_step=delay_step, syn_type=syn_type,
                                method=method, **kwargs)


if __name__ == '__main__':
  run_syn(Alpha,
          syn_type='CUBA',
          title='Alpha Synapse Model (Current-Based)',
          sp_times=[25, 50, 75, 100, 150], g_max=5.)
  run_syn(Alpha,
          syn_type='COBA',
          title='Alpha Synapse Model (Conductance-Based)',
          sp_times=[25, 50, 75, 100, 150], g_max=5.)
