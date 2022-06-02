import brainpy as bp
import matplotlib.pyplot as plt

from network_models.decision_making_rate_model import DecisionMakingRateModel

# 使用高精度float模式
bp.math.enable_x64()

coherence=6.4
model = DecisionMakingRateModel(1, coherence=coherence)

plt.figure(figsize=(4.5, 4.5))
analyzer = bp.analysis.PhasePlane2D(
  model=model,
  target_vars={'s1': [0, 1], 's2': [0, 1]},
  # fixed_vars={'I1_noise': 0., 'I2_noise': 0.},
  pars_update={'mu0': 20.},
  resolutions={'s1': 0.002, 's2': 0.002},
)
analyzer.plot_vector_field(plot_style=dict(color='lightgrey'))
analyzer.plot_nullcline(coords=dict(s2='s2-s1'),
                        x_style={'fmt': '-'},
                        y_style={'fmt': '-'})
analyzer.plot_fixed_point(tol_aux=2e-10)
analyzer.plot_trajectory(
	{'s1': [0.1], 's2': [0.1]},
	duration=2000., color='darkslateblue', linewidth=2, alpha=0.9,
)
# dmnet = DecisionMakingRateModel(1, coherence=coherence)
# runner = bp.DSRunner(dmnet, monitors=['s1', 's2'], inputs=('mu0', 20.))
# runner.run(2000)
# plt.plot(runner.mon.s1, runner.mon.s2, linewidth=2, color='darkslateblue')

plt.title('$c={}$'.format(coherence))
analyzer.show_figure()