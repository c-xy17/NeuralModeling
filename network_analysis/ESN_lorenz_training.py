import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

from network_models.ESN3 import ESN

bm.enable_x64()

# 从brainpy中获取劳伦兹系统的数据
lorenz = bp.datasets.lorenz_series(100)
data = bm.hstack([lorenz['x'], lorenz['y'], lorenz['z']])

X, Y = data[:-200], data[200:]  # Y比X提前200个步长，即需要预测系统未来的Y
# 将第0维扩展为batch的维度
X = bm.expand_dims(X, axis=0)
Y = bm.expand_dims(Y, axis=0)


num_in, num_res, num_out = 3, 200, 3
num_step = 500
num_batch = 1
num_discard = 50

model = ESN(num_in, num_res, num_out, lambda_max=0.9)

def training_lorenze(trainer, title):
  trainer.fit([X[:, :30000, :], Y[:, :30000, :]])  # 用前30000个时间的数据来训练

  predict = trainer.predict(X, reset_state=True)
  predict = bm.as_numpy(predict)

  fig = plt.figure(figsize=(10, 4))
  fig.add_subplot(121, projection='3d')
  # 画图时舍去最初20个步长的数据，下同
  plt.plot(Y[0, num_discard:, 0], Y[0, num_discard:, 1], Y[0, num_discard:, 2],
           alpha=0.8, label='standard output')
  plt.plot(predict[0, num_discard:, 0], predict[0, num_discard:, 1], predict[0, num_discard:, 2],
           alpha=0.8, label='prediction')
  plt.title(title)
  plt.legend()

  fig.add_subplot(222)
  t = np.arange(Y.shape[1])[num_discard:]
  plt.plot(t, Y[0, num_discard:, 0], linewidth=1, label='standard $x$')  # 劳伦兹系统中的x变量
  plt.plot(t, predict[0, num_discard:, 0], linewidth=1, label='predicted $x$')
  plt.ylabel('x')

  fig.add_subplot(224)
  plt.plot(t, Y[0, num_discard:, 2], linewidth=1,label='standard $z$')  # 劳伦兹系统中的z变量
  plt.plot(t, predict[0, num_discard:, 2], linewidth=1, label='predicted $z$')
  plt.ylabel('z')
  plt.xlabel('time step')

  plt.tight_layout()
  plt.show()
  # plt.savefig('E:\\2021-2022RA\\神经计算建模实战\\NeuralModeling\\images_network_models\\'
  #             'ESN_lorenz_{}.png'.format(title[14:]))

# 用岭回归法训练
ridge_trainer = bp.train.OfflineTrainer(model, fit_method=bp.algorithms.RidgeRegression(alpha=1e-6))
training_lorenze(ridge_trainer, 'Training with Ridge Regression')

# 用FORCE学习法训练
force_trainer = bp.train.OnlineTrainer(model, fit_method=bp.algorithms.ForceLearning(alpha=0.1))
training_lorenze(force_trainer, 'Training with Force Learning')
