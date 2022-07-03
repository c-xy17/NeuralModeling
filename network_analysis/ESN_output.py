import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

from network_models.ESN import ESN


num_in = 10
num_res = 2000
num_out = 30

model = ESN(num_in, num_res, num_out)

X = bm.random.random((1, 200, num_in))
