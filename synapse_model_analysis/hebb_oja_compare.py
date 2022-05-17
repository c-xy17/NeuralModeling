import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

from synapse_models.Hebb import Hebb
from synapse_models.Oja import Oja
from run_FR import run_FR, visualize_cos


bm.random.seed(299)
n_pre = 32
num_sample = 20
dur = 100.
n_steps = int(dur / bm.get_dt())

I_pre = bm.random.normal(scale=0.1, size=(n_steps, n_pre)) + bm.random.uniform(size=n_pre)
step_m = np.linspace(0, n_steps - 1, num_sample).astype(int)
x = np.asarray(I_pre.value)[step_m]

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Hebb learning rule
runner = run_FR(Hebb, I_pre, dur, ax1, 'Hebb learning', eta=0.003)
w = runner.mon['syn.w'][step_m]
visualize_cos(ax2, x, w, step_m, 'cos($x, w$) - Hebb learning')

# Oja's rule
runner = run_FR(Oja, I_pre, dur, ax1, 'Oja\'rule', eta=0.003)
w = runner.mon['syn.w'][step_m]
visualize_cos(ax2, x, w, step_m, 'cos($x, w$) - Oja\'rule')

# eigenvectors
C = np.dot(x.T, x)
eigvals, eigvecs = np.linalg.eig(C)
eigvals, eigvecs  = eigvals.real, eigvecs.T.real
largest = eigvecs[np.argsort(eigvals)[-1]]
visualize_cos(ax2, x, np.ones((num_sample, n_pre)) * largest,
              step_m, 'cos($x, v_1$)', linestyle='--')

ax1.set_xlabel('t (ms)')
ax1.set_ylabel('$||w||$')
ax1.legend()

ax2.set_ylim(0.5-0.05, 1.05)
ax2.legend()
plt.tight_layout()
plt.show()