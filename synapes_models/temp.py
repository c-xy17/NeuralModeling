# import numpy as bm
import brainpy.math as bm

import numpy as np
import jax
import jax.numpy as jnp


def id_sum(v, id, num):
	sorting = bm.argsort(id)
	sorted_id, sorted_v = bm.asarray(id[sorting]), bm.asarray(v[sorting])
	unique_id, count = bm.unique(sorted_id, return_counts=True)
	id_count = bm.zeros(num, dtype=int)
	id_count[unique_id] = count

	count_cumsum = id_count.cumsum()
	v_cumsum = sorted_v.cumsum()
	cumsum = v_cumsum[count_cumsum - 1]
	return bm.insert(bm.diff(cumsum), 0, cumsum[0])


def id_sum2(v, id, num):
	# v = jnp.asarray(v.value)
	# id = jnp.asarray(id.value)
	res = jax.ops.segment_sum(v.value, id.value)
	return bm.asarray(res)


a = bm.array([5, 4, 1, 3])
id = bm.array([0, 1, 0, 3])
num = 4

res = id_sum2(a, id, num)
print(res)
print(type(res))
