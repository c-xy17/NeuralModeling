import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt

from network_models.ESN3 import ESN


num_in = 10
num_res = 500
num_out = 30
num_step = 500  # 模拟总步长
num_batch = 1

# 生成网络，运行两次模拟，两次模拟的输入相同，但网络的初始化状态不同
def get_esn_states(lambda_max):
  model = ESN(num_in, num_res, num_out, lambda_max=lambda_max)
  model.reset(batch_size=num_batch)

  inputs = bm.random.randn(num_batch, int(num_step/num_batch), num_in)  # 第0个维度为batch的大小

  # 第一次运行
  model.state.value = bp.init.Uniform(-1., 1.)((num_batch, num_res))  # 随机初始化网络状态
  runner = bp.train.DSTrainer(model, monitors=['state'])
  runner.predict(inputs)
  state1 = np.concatenate(runner.mon['state'], axis=0)

  # 第二次运行
  model.state.value = bp.init.Uniform(-1., 1.)((num_batch, num_res))  # 再次随机初始化网络状态
  runner = bp.train.DSTrainer(model, monitors=['state'])
  runner.predict(inputs)
  state2 = np.concatenate(runner.mon['state'], axis=0)

  return state1, state2

# 画出两次模拟中某一时刻网络的状态
def plot_states(state1, state2, title):
  assert len(state1) == len(state2)
  x = np.arange(len(state1))
  plt.scatter(x, state1, s=1., label='first state')
  plt.scatter(x, state2, s=1., label='last state')
  plt.legend(loc='upper right')
  plt.xlabel('neuron index')
  plt.ylabel('state')
  plt.title(title)


lambda1, lambda2, lambda3 = 0.9, 1.0, 1.1

plt.figure(figsize=(15., 4.5))
plt.subplot(131)

# 画出每个lambda_max下两次模拟的网络状态的距离随时间的变化
state1, state2 = get_esn_states(lambda_max=lambda1)
distance = np.sqrt(np.sum(np.square(state1 - state2), axis=1))
plt.plot(np.arange(num_step), distance, label='$|\lambda_{}|={}$'.format('{max}', lambda1))

state3, state4 = get_esn_states(lambda_max=lambda2)
distance = np.sqrt(np.sum(np.square(state3 - state4), axis=1))
plt.plot(np.arange(num_step), distance, label='$|\lambda_{}|={}$'.format('{max}', lambda2))

state5, state6 = get_esn_states(lambda_max=lambda3)
distance = np.sqrt(np.sum(np.square(state5 - state6), axis=1))
plt.plot(np.arange(num_step), distance, label='$|\lambda_{}|={}$'.format('{max}', lambda3))

plt.xlabel('running step')
plt.ylabel('distance')
plt.legend()

# 画出两次模拟时网络的初始状态和最终状态
plt.subplot(232)
plot_states(state1[0], state2[0], title='$|\lambda_{}|={}$, n=0'.format('{max}', lambda1))
plt.subplot(233)
plot_states(state1[-1], state2[-1], title='$|\lambda_{}|={}$, n={}'.format('{max}', lambda1, num_step))

plt.subplot(235)
plot_states(state5[0], state6[0], title='$|\lambda_{}|={}$, n=0'.format('{max}', lambda3))
plt.subplot(236)
plot_states(state5[-1], state6[-1], title='$|\lambda_{}|={}$, n={}'.format('{max}', lambda3, num_step))

plt.tight_layout()
plt.show()
# plt.savefig('E:\\2021-2022RA\\神经计算建模实战\\NeuralModeling\\images_network_models\\'
#             'ESN_state_distance.png')





# from network_models.ESN import ESN
#
# model1 = ESN(num_in, num_res, num_out, lambda_max=0.8)
# model2 = ESN(num_in, num_res, num_out, lambda_max=0.8)
#
# # inputs = bm.random.randn(num_step, num_in)
# inputs = bm.random.random((1, num_step, num_in))
#
# runner1 = bp.train.DSTrainer(model1, monitors=['r.state'])
# runner1.predict(inputs)
# runner2 = bp.train.DSTrainer(model2, monitors=['r.state'])
# runner2.predict(inputs)
# # print(runner1.mon['r.state'].shape)
#
# dist0 = np.sqrt(np.sum(np.square(model1.r.state - model2.r.state), axis=1))
# print(dist0)
#
# # runner1 = bp.dyn.DSRunner(model1, monitors=['state'])
# # runner1.run(duration, inputs)
#
# # runner2 = bp.dyn.DSRunner(model2, monitors=['state'])
# # runner2.run(duration, inputs)
#
# print(runner1.mon['r.state'].shape)
# state1 = runner1.mon['r.state'].squeeze()
# state2 = runner2.mon['r.state'].squeeze()
#
# distance = np.sqrt(np.sum(np.square(state1 - state2), axis=1))
#
# plt.plot(runner1.mon['ts'], distance)
#
# plt.show()