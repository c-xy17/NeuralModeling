import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt

from network_models.ESN3 import ESN


bm.enable_x64()  # 使用更高精度的float以提高训练精度

num_in, num_res, num_out = 1, 600, 1
num_step = 1000  # 模拟总步长
num_discard = 200  # 训练时，丢弃掉前200个数据

def plot_result(output, Y, title):
  assert output.shape == Y.shape
  x = np.arange(output.shape[0])
  plt.plot(x, Y, label='$y$')
  plt.plot(x, output, label='$\hat{y}$')
  plt.legend()
  plt.xlabel('running step')
  plt.ylabel('state')
  plt.title(title)

# 生成训练数据
n = bm.linspace(0., bm.pi, num_step)
U = bm.sin(10 * n) + bm.random.normal(scale=0.1, size=num_step)  # 输入
U = U.reshape((1, -1, num_in))  # 维度：(num_batch, num_step, num_dim)
Y = bm.power(bm.sin(10 * n), 7)  # 输出
Y = Y.reshape((1, -1, num_out))  # 维度：(num_batch, num_step, num_dim)

model = ESN(num_in, num_res, num_out, lambda_max=0.95)

# 训练前，运行模型得到结果
runner = bp.train.DSTrainer(model, monitors=['state'])
untrained_out = runner.predict(U)
print(bp.losses.mean_absolute_error(untrained_out[:, num_discard:], Y[:, num_discard:]))

# 用岭回归法进行训练
trainer = bp.train.RidgeTrainer(model, alpha=1e-12)
trainer.fit([U[:, num_discard:], Y[:, num_discard:]])

# 训练后，运行模型得到结果
runner = bp.train.DSTrainer(model, monitors=['state'])
out = runner.predict(U)
print(bp.losses.mean_absolute_error(out[:, num_discard:], Y[:, num_discard:]))

# 可视化
plt.figure(figsize=(12, 4.5))
ax1 = plt.subplot(121)
plot_result(untrained_out.flatten()[num_discard:], Y.flatten()[num_discard:], 'before training')
ax2 = plt.subplot(122, sharey=ax1)
plot_result(out.flatten()[num_discard:], Y.flatten()[num_discard:], 'after training')

# plt.show()
plt.savefig('E:\\2021-2022RA\\神经计算建模实战\\NeuralModeling\\images_network_models\\'
              'ESN_sin_training.png')


plt.figure(figsize=(12, 3))
state = runner.mon['state'].squeeze()
i1, i2, i3, i4 = tuple(np.random.choice(np.arange(num_res), 4, replace=False))
ax = plt.subplot(131)
plt.plot(np.arange(num_step - num_discard), state[num_discard:, i1])
plt.title('neuron {}'.format(i1))
plt.xlabel('running step')
plt.subplot(132, sharey=ax)
plt.plot(np.arange(num_step - num_discard), state[num_discard:, i2])
plt.title('neuron {}'.format(i2))
plt.xlabel('running step')
plt.subplot(133, sharey=ax)
plt.plot(np.arange(num_step - num_discard), state[num_discard:, i3])
plt.title('neuron {}'.format(i3))
plt.xlabel('running step')
plt.tight_layout()
# plt.show()
plt.savefig('E:\\2021-2022RA\\神经计算建模实战\\NeuralModeling\\images_network_models\\'
              'ESN_sin_example_neurons.png')
