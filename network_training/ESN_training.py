import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt

from network_models.ESN import ESN


num_in = 1
num_res = 200
num_out = 1
num_step = 1000  # 模拟总步长

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
X = bm.linspace(0., bm.pi, num_step)
U = bm.sin(30*X) + bm.random.normal(scale=0.05, size=num_step)  # 输入
# U = bm.ones(X.shape)
U = U.reshape((1, -1, num_in))  # 维度：(num_batch, num_step, num_dim)
Y = bm.sin(30*X) + bm.sin(50 * (X + bm.pi/2)) + bm.sin(70 * (X + bm.pi/4)) \
    + bm.random.normal(scale=0.05, size=num_step)   # 标准输出
Y = Y.reshape((1, -1, num_out))  # 维度：(num_batch, num_step, num_dim)

# U = bm.random.random((1, num_step, num_in))
# Y = bm.random.random((1, num_step, num_out))

model = ESN(num_in, num_res, num_out, lambda_max=0.9)
W = model.r.Wrec.value
bm.where(W > 0, 0.4, W)
bm.where(W < 0, -0.4, W)
model.r.Wrec.value = W

# 训练前，运行模型得到结果
runner = bp.train.DSTrainer(model, monitors=['r.state'])
untrained_out = runner.predict(U)
print(bp.losses.mean_absolute_error(untrained_out, Y))


from sklearn import linear_model

# trainer = linear_model.Ridge(fit_intercept=False, alpha=0.0)
trainer = linear_model.LinearRegression(fit_intercept=False)
trainer.fit(runner.mon['r.state'].squeeze()[200:], Y.squeeze()[200:])
model.o.W.value = bm.asarray(trainer.coef_.reshape(-1, 1))

# # # 用岭回归进行训练
# trainer = bp.train.RidgeTrainer(model, alpha=0.)
# trainer.fit([U, Y])

# 训练后，运行模型得到结果
runner = bp.train.DSTrainer(model, monitors=['r.state'])
out = runner.predict(U)
print(bp.losses.mean_absolute_error(out, Y))


# 可视化
plt.figure(figsize=(12, 4.5))
plt.subplot(121)
plot_result(untrained_out.flatten(), Y.flatten(), 'before training')
plt.subplot(122)
plot_result(out.flatten(), Y.flatten(), 'after training')

plt.show()

# # input-output
# print(model(dict(), bm.ones((1, num_in))))
#
# X = bm.random.random((1, 200, num_in))
# Y = bm.random.random((1, 200, num_out))
#
# # prediction
# runner = bp.train.DSTrainer(model, monitors=['state'])
# outputs = runner.predict(X)
# print(runner.mon['state'].shape)
# print(bp.losses.mean_absolute_error(outputs, Y))
# print()
#
# # training
# trainer = bp.train.RidgeTrainer(model)
# trainer.fit([X, Y])
#
# # prediction
# runner = bp.train.DSTrainer(model, monitors=['state'])
# outputs = runner.predict(X)
# print(runner.mon['state'].shape)
# print(bp.losses.mean_absolute_error(outputs, Y))
# print()
#
# outputs = trainer.predict(X)
# print(bp.losses.mean_absolute_error(outputs, Y))