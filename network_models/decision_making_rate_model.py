import brainpy as bp
import brainpy.math as bm

bp.math.enable_x64()


class DecisionMakingRateModel(bp.dyn.NeuGroup):
  def __init__(self, size, coherence, JE=0.2609, JI=-0.0497, Jext=5.2e-4, I0=0.3255,
               gamma=0.641, tau=0.1, tau_n=2., sigma_n=0.02, a=270., b=108., d=0.154,
               method='exp_auto', **kwargs):
    super(DecisionMakingRateModel, self).__init__(size, **kwargs)

    # 初始化参数
    self.coherence = coherence
    self.JE = JE
    self.JI = JI
    self.Jext = Jext
    self.I0 = I0
    self.gamma = gamma
    self.tau = tau
    self.tau_n = tau_n
    self.sigma_n = sigma_n
    self.a = a
    self.b = b
    self.d = d

    # 初始化变量
    self.s1 = bm.Variable(bm.ones(self.num) * 0.06)
    self.s2 = bm.Variable(bm.ones(self.num) * 0.06)
    self.mu0 = bm.Variable(bm.zeros(self.num))
    self.I1_noise = bm.Variable(bm.zeros(self.num))
    self.I2_noise = bm.Variable(bm.zeros(self.num))

    self.noise1 = bp.dyn.PoissonGroup(self.num, freqs=2400)
    self.noise2 = bp.dyn.PoissonGroup(self.num, freqs=2400)

    # 定义积分函数
    self.integral = bp.odeint(self.derivative, method=method)

  @property
  def derivative(self):
    return bp.JointEq([self.ds1, self.ds2, self.dI1noise, self.dI2noise])

  def ds1(self, s1, t, s2, mu0):
    I1 = self.Jext * mu0 * (1. + self.coherence / 100.)
    x1 = self.JE * s1 + self.JI * s2 + self.I0 + I1 + self.I1_noise
    r1 = (self.a * x1 - self.b) / (1. - bm.exp(-self.d * (self.a * x1 - self.b)))
    return - s1 / self.tau + (1. - s1) * self.gamma * r1

  def ds2(self, s2, t, s1, mu0):
    I2 = self.Jext * mu0 * (1. - self.coherence / 100.)
    x2 = self.JE * s2 + self.JI * s1 + self.I0 + I2 + self.I2_noise
    r2 = (self.a * x2 - self.b) / (1. - bm.exp(-self.d * (self.a * x2 - self.b)))
    return - s2 / self.tau + (1. - s2) * self.gamma * r2

  def dI1noise(self, I1noise, t, noise1):
    return (- I1noise + noise1.spike * bm.sqrt(self.tau_n * self.sigma_n * self.sigma_n)) / self.tau_n

  def dI2noise(self, I2noise, t, noise2):
    return (- I2noise + noise2.spike * bm.sqrt(self.tau_n * self.sigma_n * self.sigma_n)) / self.tau_n

  def update(self, _t, _dt):
    self.noise1.update(_t, _dt)
    self.noise2.update(_t, _dt)
    integral = self.integral(self.s1, self.s2, self.I1_noise, self.I2_noise, _t, mu0=self.mu0,
                             noise1=self.noise1, noise2=self.noise2, dt=_dt)
    self.s1.value, self.s2.value, self.I1_noise.value, self.I2_noise.value = integral
    self.mu0[:] = 0.

