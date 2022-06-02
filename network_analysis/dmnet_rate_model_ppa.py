import brainpy as bp
import matplotlib.pyplot as plt

from network_models.decision_making_rate_model import DecisionMakingRateModel

# 使用高精度float模式
bp.math.enable_x64()

def dmnet_ppa(coherence, mu0=20.):
  model = DecisionMakingRateModel(1, coherence=coherence)

  # 构相平面建分析器
  analyzer = bp.analysis.PhasePlane2D(
    model=model,
    target_vars={'s1': [0, 1], 's2': [0, 1]},
    fixed_vars={'I1_noise': 0., 'I2_noise': 0.},
    pars_update={'mu0': mu0},
    resolutions={'s1': 0.002, 's2': 0.002},
  )

  plt.figure(figsize=(4.5, 4.5))
  # 画出向量场
  analyzer.plot_vector_field(plot_style=dict(color='lightgrey'))
  # 画出零增长等值线
  analyzer.plot_nullcline(coords=dict(s2='s2-s1'), x_style={'fmt': '-'}, y_style={'fmt': '-'})
  # 画出奇点
  analyzer.plot_fixed_point(tol_aux=2e-10)
  # 画出s1, s2的运动轨迹
  analyzer.plot_trajectory(
    {'s1': [0.1], 's2': [0.1]},
    duration=2000., color='darkslateblue', linewidth=2, alpha=0.9,
  )

  plt.title('$c={}, \mu_0={}$'.format(coherence, mu0))
  plt.show()

dmnet_ppa(25.6, 20.)