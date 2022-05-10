from AMPA import AMPA
from run_synapse import run_syn


# [T] modeling
class GABAa(AMPA):
	def __init__(self, pre, post, conn, g_max=0.2, E=-80., alpha=0.53, beta=0.18,
	             T_0=1., T_dur=1., delay_step=2, method='exp_auto', **kwargs):
		super(GABAa, self).__init__(pre, post, conn, g_max=g_max, E=E, alpha=alpha,
		                            beta=beta, T_0=T_0, T_dur=T_dur,
		                            delay_step=delay_step, method=method, **kwargs)


run_syn(GABAa, title='GABA$_\mathrm{A}$ Synapse Model')
