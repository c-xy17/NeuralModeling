import brainpy as bp
import brainpy.math as bm


class ESN(bp.dyn.TrainingSystem):
  def __init__(self, num_in, num_hidden, num_out):
    super(ESN, self).__init__()
    self.r = bp.layers.Reservoir(num_in, num_hidden,
                                 Win_initializer=bp.init.Uniform(-0.1, 0.1),
                                 Wrec_initializer=bp.init.Normal(scale=0.1),
                                 in_connectivity=0.02,
                                 rec_connectivity=0.02,
                                 conn_type='dense')
    self.o = bp.layers.Dense(num_hidden, num_out, W_initializer=bp.init.Normal())

  def forward(self, shared_args, x):
    rec = self.r(shared_args, x)
    output = self.o(shared_args, rec)
    return output

  # def update(self, sha: dict, x):
  #   self.forward(x=x, shared_args=sha)


def train_esn_with_ridge(num_in=100, num_out=30):
  model = ESN(num_in, 2000, num_out)

  # input-output
  print(model(bm.ones((1, num_in))))
  # print(model(x=bm.ones((1, num_in)), sha=None))

  X = bm.random.random((1, 200, num_in))
  Y = bm.random.random((1, 200, num_out))

  # prediction
  runner = bp.train.DSTrainer(model, monitors=['r.state'])
  outputs = runner.predict(X)
  print(runner.mon['r.state'].shape)
  print(bp.losses.mean_absolute_error(outputs, Y))
  print()

  # training
  trainer = bp.train.RidgeTrainer(model)
  trainer.fit([X, Y])

  # prediction
  runner = bp.train.DSTrainer(model, monitors=['r.state'])
  outputs = runner.predict(X)
  print(runner.mon['r.state'].shape)
  print(bp.losses.mean_absolute_error(outputs, Y))
  print()

  outputs = trainer.predict(X)
  print(bp.losses.mean_absolute_error(outputs, Y))

train_esn_with_ridge(10, 30)