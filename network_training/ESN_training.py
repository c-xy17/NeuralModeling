import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt

from network_models.ESN3 import ESN


num_in = 10
num_res = 500
num_out = 30
num_step = 500  # 模拟总步长

model = ESN(num_in, num_res, num_out)

# input-output
print(model(dict(), bm.ones((1, num_in))))

X = bm.random.random((1, 200, num_in))
Y = bm.random.random((1, 200, num_out))

# prediction
runner = bp.train.DSTrainer(model, monitors=['state'])
outputs = runner.predict(X)
print(runner.mon['state'].shape)
print(bp.losses.mean_absolute_error(outputs, Y))
print()

# training
trainer = bp.train.RidgeTrainer(model)
trainer.fit([X, Y])

# prediction
runner = bp.train.DSTrainer(model, monitors=['state'])
outputs = runner.predict(X)
print(runner.mon['state'].shape)
print(bp.losses.mean_absolute_error(outputs, Y))
print()

outputs = trainer.predict(X)
print(bp.losses.mean_absolute_error(outputs, Y))