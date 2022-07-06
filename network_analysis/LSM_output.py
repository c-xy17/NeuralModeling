import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt

from network_models.LSM import LSM, euclidean_dist


num_step = 5000
num_sample = 256

def run_lsm(runner, spike_train):
  runner.run(inputs=spike_train, duration=num_step * bm.get_dt(), inputs_are_batching=True, reset_state=True)
  res_state = np.concatenate([runner.mon['E.spike'], runner.mon['I.spike']], axis=1)
  return res_state

lsm = LSM()

runner = bp.dyn.DSRunner(lsm, monitors={'E.spike': lsm.E.spike, 'I.spike': lsm.I.spike}, jit=False)

mask = bm.random.rand(num_sample, num_step, lsm.num_input)
input_spike = bm.zeros((num_sample, num_step, lsm.num_input), dtype=bool)
input_spike[mask < 1600. * bm.get_dt() / 1000.] = True
print(input_spike.shape)
print(lsm.input2E.pre.size)
state1 = run_lsm(runner, input_spike)

mask = bm.random.rand(num_sample, num_step, lsm.num_input)
input_spike = bm.zeros((num_sample, num_step, lsm.num_input), dtype=bool)
input_spike[mask < 800. * bm.get_dt() / 1000.] = True
state2 = run_lsm(runner, input_spike)

dists = []
for i in range(len(state1)):
  dists.append(euclidean_dist(state1, state2))
dists = np.asarray(dists)

plt.plot(np.arange(len(dists)), dists)

# duration = 500.
#
# def compare_state(freq1, freq2, dur):
#   lsm1 = LSM(input_freq=freq1)
#
#   runner = bp.dyn.DSRunner(lsm1, monitors=['input_neuron.spike', 'E.spike', 'I.spike'])
#   runner.run(dur)
#
#   res_state1 = np.concatenate([runner.mon['E.spike'], runner.mon['I.spike']], axis=1)
#
#   lsm2 = LSM(input_freq=freq2)
#
#   runner = bp.dyn.DSRunner(lsm2, monitors=['input_neuron.spike', 'E.spike', 'I.spike'])
#   runner.run(dur)
#
#   res_state2 = np.concatenate([runner.mon['E.spike'], runner.mon['I.spike']], axis=1)
#   ts = runner.mon.ts
#
#   # 计算两个LSM中liquid state的差别
#   dists = []
#   for i in range(len(res_state1)):
#     dists.append(euclidean_dist(res_state1, res_state2))
#   dists = np.asarray(dists)
#
#   plt.plot(ts, dists)
#
# compare_state(1600, 800, duration)
#
# plt.show()



# pre = bp.dyn.PoissonGroup(1, freqs=1600)
# post = bp.dyn.LIF(1)
# syn = bp.dyn.ExpCOBA(pre, post, bp.conn.All2All(), tau=5., g_max=1.)
# net = bp.dyn.Network(pre=pre, post=post, syn=syn)
#
# runner = bp.dyn.DSRunner(net, monitors=['syn.g'])
# runner.run(500)
# g_u = runner.mon['syn.g'].flatten()
#
# net.pre.freqs = 800
# net.reset()
#
# runner = bp.dyn.DSRunner(net, monitors=['syn.g'])
# runner.run(500)
# g_v = runner.mon['syn.g'].flatten()
#
# print(g_v.shape)
#
# d_uv = euclidean_dist(g_u, g_v) / (500. / bm.get_dt())
#
# print(d_uv)

# runner = bp.train.DSTrainer(lsm, monitors=['E.spike', 'I.spike'])
# outputs = runner.predict()