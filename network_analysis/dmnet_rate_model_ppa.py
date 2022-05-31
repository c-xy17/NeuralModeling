import brainpy as bp
import matplotlib.pyplot as plt

from network_models.decision_making_rate_model import DecisionMakingRateModel

bp.math.enable_x64()


model = DecisionMakingRateModel(1, coherence=25.6)

plt.figure(figsize=(4.5, 4.5))
analyzer = bp.analysis.PhasePlane2D(
    model=model,
    target_vars={'s1': [0, 1], 's2': [0, 1]},
    pars_update={'mu0': 20.},
    resolutions={'s1': 0.002, 's2': 0.002},
)
analyzer.plot_vector_field(plot_style=dict(color='lightgrey'))
analyzer.plot_nullcline(coords=dict(s2='s2-s1'),
                        x_style={'fmt': '-'},
                        y_style={'fmt': '-'})
analyzer.plot_fixed_point()
analyzer.plot_trajectory(
	{'s1': [0.06], 's2': [0.06]},
	duration=200., color='darkslateblue', linewidth=2, alpha=0.9,
)
analyzer.show_figure()