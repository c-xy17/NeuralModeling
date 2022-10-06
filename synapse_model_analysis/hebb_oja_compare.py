import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

from synapse_models.Hebb import Hebb
from synapse_models.Oja import Oja
from run_FR import run_FR, visualize_cos


bm.random.seed(299)
n_pre = 32  # 32个突触前神经元
num_sample = 20  # 挑选20个时间点可视化
dur = 100.  # 模拟总时长
n_steps = int(dur / bm.get_dt())  # 模拟总步长

I_pre = bm.random.normal(scale=0.1, size=(n_steps, n_pre)) + bm.random.uniform(size=n_pre)
step_m = np.linspace(0, n_steps - 1, num_sample).astype(int)
x = np.asarray(I_pre.value)[step_m]

# _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Hebb learning rule
runner = run_FR(Hebb, I_pre, dur, None, 'Hebb learning', eta=0.003)
w = runner.mon['syn.w'][step_m]
visualize_cos(None, x, w, step_m, 'cos($x, w$) - Hebb learning')

# Oja's rule
runner = run_FR(Oja, I_pre, dur, None, 'Oja\'rule', eta=0.003)
w = runner.mon['syn.w'][step_m]
visualize_cos(None, x, w, step_m, 'cos($x, w$) - Oja\'rule', linestyle='-')

# eigenvectors
C = np.dot(x.T, x)
eigvals, eigvecs = np.linalg.eig(C)
eigvals, eigvecs  = eigvals.real, eigvecs.T.real
largest = eigvecs[np.argsort(eigvals)[-1]]
visualize_cos(None, x, np.ones((num_sample, n_pre)) * largest,
              step_m, 'cos($x, v_1$)', linestyle='--')

plt.xlabel(r'$t$ (ms)')
plt.ylabel(r'$||w||$')
plt.legend()
# plt.savefig('../img/hebb_oja_compare2.pdf', transparent=True, dpi=500)
plt.show()
