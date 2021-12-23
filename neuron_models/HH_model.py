import brainpy as bp
import brainpy.math as bm

bm.set_platform('cpu')


class HH(bp.NeuGroup):
  def __init__(self, size, ENa=50., gNa=120., EK=-77., gK=36., EL=-54.387, gL=0.03,
                 V_th=20., C=1.0, **kwargs):
    super(HH, self).__init__(size=size, **kwargs)

    # initialize parameters
    self.ENa = ENa
    self.EK = EK
    self.EL = EL
    self.gNa = gNa
    self.gK = gK
    self.gL = gL
    self.C = C
    self.V_th = V_th

    # initialize variables
    self.V = bm.Variable(bm.random.randn(self.num) - 70.)
    self.m = bm.Variable(0.5 * bm.ones(self.num))
    self.h = bm.Variable(0.6 * bm.ones(self.num))
    self.n = bm.Variable(0.32 * bm.ones(self.num))
    self.input = bm.Variable(bm.zeros(self.num))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))

    # integral function
    self.integral = bp.odeint(f=self.derivative, method='exponential_euler')

  def derivative(self, V, m, h, n, t, Iext):
    alpha = 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
    beta = 4.0 * bm.exp(-(V + 65) / 18)
    dmdt = alpha * (1 - m) - beta * m

    alpha = 0.07 * bm.exp(-(V + 65) / 20.)
    beta = 1 / (1 + bm.exp(-(V + 35) / 10))
    dhdt = alpha * (1 - h) - beta * h

    alpha = 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
    beta = 0.125 * bm.exp(-(V + 65) / 80)
    dndt = alpha * (1 - n) - beta * n

    I_Na = (self.gNa * m ** 3.0 * h) * (V - self.ENa)
    I_K = (self.gK * n ** 4.0) * (V - self.EK)
    I_leak = self.gL * (V - self.EL)
    dVdt = (- I_Na - I_K - I_leak + Iext) / self.C

    return dVdt, dmdt, dhdt, dndt

  def update(self, _t, _dt):
    # compute V, m, h, n
    V, m, h, n = self.integral(self.V, self.m, self.h, self.n, _t, self.input, dt=_dt)

    # update the spiking state and the last spiking time
    self.spike[:] = bm.logical_and(self.V < self.V_th, V >= self.V_th)
    self.t_last_spike[:] = bm.where(self.spike, _t, self.t_last_spike)

    # update V, m, h, n
    self.V[:] = V
    self.m[:] = m
    self.h[:] = h
    self.n[:] = n

    # reset the external input
    self.input[:] = 0.


group = HH(10)
runner = bp.StructRunner(group, monitors=['V'], inputs=('input', 20.))
runner(200)  # 运行时长为200ms
bp.visualize.line_plot(runner.mon.ts, runner.mon.V, show=True)
