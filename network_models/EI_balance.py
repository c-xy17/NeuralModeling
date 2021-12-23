import brainpy as bp
import brainpy.math as bm
#
# from brainmodels.neurons import LIF
# from brainmodels.synapses import ExpCOBA

LIF = bp.models.LIF
ExpCOBA = bp.models.ExpCOBA


class EINet(bp.Network):
  def __init__(self, num_exc, num_inh, method='exp_auto', **kwargs):
    super(EINet, self).__init__(**kwargs)

    # neurons
    pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.)
    E = LIF(num_exc, **pars, method=method)
    I = LIF(num_inh, **pars, method=method)
    E.V[:] = bp.math.random.randn(num_exc) * 2 - 55.
    I.V[:] = bp.math.random.randn(num_inh) * 2 - 55.

    # synapses
    w_e = 0.6  # excitatory synaptic weight
    w_i = 6.7  # inhibitory synaptic weight
    E_pars = dict(E=0., g_max=w_e, tau=5.)
    I_pars = dict(E=-80., g_max=w_i, tau=10.)
    self.E2E = ExpCOBA(E, E, bp.conn.FixedProb(prob=0.02), **E_pars, method=method)
    self.E2I = ExpCOBA(E, I, bp.conn.FixedProb(prob=0.02), **E_pars, method=method)
    self.I2E = ExpCOBA(I, E, bp.conn.FixedProb(prob=0.02), **I_pars, method=method)
    self.I2I = ExpCOBA(I, I, bp.conn.FixedProb(prob=0.02), **I_pars, method=method)

    self.E = E
    self.I = I


net = EINet(3200, 800, method='exp_auto')
# simulation
runner = bp.StructRunner(net,
                         monitors=['E.spike', 'I.spike'],
                         inputs=[('E.input', 20.), ('I.input', 20.)])
t = runner.run(100.)
print(t)

# visualization
bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'],
                         title='Spikes of Excitatory Neurons', show=True)
bp.visualize.raster_plot(runner.mon.ts, runner.mon['I.spike'],
                         title='Spikes of Inhibitory Neurons', show=True)
