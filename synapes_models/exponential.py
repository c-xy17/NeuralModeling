import brainpy as bp
import brainpy.math as bm
# from neuron_models.LIF_model import LIF
from brainmodels.neurons import LIF

# LIF = bp.models.LIF


class Exponential(bp.TwoEndConn):
  def __init__(self, pre, post, conn, g_max=1., delay=0., E=0., tau=8.0, **kwargs):
    # connections are built in the initialization function of bp.TwoEndConn
    super(Exponential, self).__init__(pre=pre, post=post, conn=conn, **kwargs)

    # initialize parameters
    self.tau = tau
    self.g_max = g_max
    self.delay = delay
    self.E = E

    # acquire desired properties of the connection
    self.pre_ids, self.post_ids, self.pre2post = self.conn.requires('pre_ids', 'post_ids', 'pre2post')

    # variables
    self.s = bm.Variable(bm.zeros(self.post.num))
    self.pre_spike = self.register_constant_delay('pre_spike', self.pre.num, delay)

    self.integral = bp.odeint(f=self.derivative, method='exponential_euler')

  def derivative(self, s, t):
    dsdt = -s / self.tau
    return dsdt

  def update(self, _t, _dt):
    # push the pre-synaptic spikes into the delay
    self.pre_spike.push(self.pre.spike)

    # pull the delayed pre-synaptic spikes
    delayed_pre_spike = self.pre_spike.pull()

    # get the spikes of each presynaptic neuron
    spikes = bm.pre2syn(delayed_pre_spike, self.pre_ids)

    # transmit the spikes to postsynaptic neurons
    post_sp = bm.syn2post(spikes, self.post_ids, self.post.num)

    # update the state variable
    self.s[:] = self.integral(self.s, _t, dt=_dt) + post_sp

    # update the output of currents, i.e. the postsynaptic input
    self.post.input[:] += self.g_max * self.s * - (self.post.V - self.E)


pre_size = (8, 4)
post_size = 10
pre_neu = LIF(pre_size, tau=10, V_th=-30, V_rest=-60, V_reset=-60, tau_ref=5.)
post_neu = LIF(post_size, tau=20, V_th=-30, V_rest=-60, V_reset=-60, tau_ref=5.)

conn = bp.connect.All2All()  # all-to-all connections, a subclass of bp.connect.Connector
exp_syn = Exponential(pre_neu, post_neu, conn, E=0., g_max=0.6, tau=5)

net = bp.Network(exp_syn, pre=pre_neu, post=post_neu)
runner = bp.StructRunner(net, monitors=['pre.V', 'post.V'], inputs=[('pre.input', 35.)])
# runner = bp.StructRunner(net, monitors=['pre.V', 'post.V', 'post.spike'], inputs=[('pre.input', 35.)])
runner.run(200)
bp.visualize.line_plot(runner.mon.ts, runner.mon['pre.V'],
                       title='Presynaptic Spikes', show=True)
bp.visualize.line_plot(runner.mon.ts, runner.mon['post.V'],
                       title='Postsynaptic Spikes', show=True)
# bp.visualize.raster_plot(runner.mon.ts, runner.mon['post.spike'], show=True)
