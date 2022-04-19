import brainpy as bp
import brainpy.math as bm
from jax.ops import segment_sum

from run_synapse import run_syn_GJ


class GapJunction(bp.dyn.TwoEndConn):
	def __init__(self, pre, post, conn, g=0.2, **kwargs):
		super(GapJunction, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
		self.check_pre_attrs('V')
		self.check_post_attrs('input', 'V')

		# 初始化参数
		self.g = g

		# 获取每个连接的突触前神经元pre_ids和突触后神经元post_ids
		self.pre_ids, self.post_ids = self.conn.require('pre_ids', 'post_ids')

		# 初始化变量
		self.current = bm.Variable(bm.zeros(self.post.num))

	def update(self, _t, _dt):
		# 计算突触后电流（）
		inputs = self.g * (self.pre.V[self.pre_ids.value] - self.post.V[self.post_ids.value])

		# 从synapse到post的计算：post id相同电流加到一起
		self.current.value = bm.syn2post(inputs, self.post_ids, self.post.num)
		self.post.input += self.current


run_syn_GJ(GapJunction, title='Gap Junction Model')
run_syn_GJ(GapJunction, Iext=5., title='Gap Junction Model')


# 将相同id的值加和到一起
def id_sum(self, v, id):
	sorting = bm.argsort(id)
	sorted_id, sorted_v = id[sorting], v[sorting]
	unique_id, count = bm.unique(sorted_id, return_counts=True)
	id_count = bm.zeros(self.post.num, dtype=int)
	id_count[unique_id] = count

	count_cumsum = id_count.cumsum()
	v_cumsum = sorted_v.cumsum()
	cumsum = v_cumsum[count_cumsum - 1]
	return bm.insert(bm.diff(cumsum), 0, cumsum[0])
