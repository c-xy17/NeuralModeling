import brainpy as bp
import brainpy.math as bm

class FR(bp.NeuGroup):
  def __init__(self, size, **kwargs):
    super(FR, self).__init__(size=size, **kwargs)
    self.r = bm.Variable(bm.zeros(self.num))
    self.input = bm.Variable(bm.zeros(self.num))

  def update(self, tdi):
    self.r.value = self.input  # 将输入直接视为r
    self.input[:] = 0.
