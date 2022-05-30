import brainpy as bp
import brainpy.math as bm

# 调用BrainPy中的预置模型
LIF = bp.dyn.LIF
ExpCOBA = bp.dyn.ExpCOBA

# 搭建神经元
pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.)  # 神经元模型需要的参数
E = LIF(3200, **pars)
I = LIF(800, **pars)
E.V.value = bm.random.randn(3200) * 4. - 60.  # 随机初始化膜电位
I.V.value = bm.random.randn(800) * 4. - 60.  # 随机初始化膜电位

# 搭建神经元连接
E_pars = dict(E=0., g_max=0.3, tau=5.)  # 兴奋性突触需要的参数
I_pars = dict(E=-80., g_max=3.2, tau=10.)  # 抑制性突触需要的参数
E2E = ExpCOBA(E, E, bp.conn.FixedProb(prob=0.02), **E_pars)
E2I = ExpCOBA(E, I, bp.conn.FixedProb(prob=0.02), **E_pars)
I2E = ExpCOBA(I, E, bp.conn.FixedProb(prob=0.02), **I_pars)
I2I = ExpCOBA(I, I, bp.conn.FixedProb(prob=0.02), **I_pars)

einet = bp.dyn.Network(E2E, E2I, I2E, I2I, E=E, I=I)

runner = bp.DSRunner(einet,
                     monitors=['E.spike', 'I.spike', 'E.input', 'E.V'],
                     inputs=[('E.input', 12.), ('I.input', 12.)])
runner(200.)


