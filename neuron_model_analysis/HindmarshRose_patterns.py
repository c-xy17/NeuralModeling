import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

from neuron_models.HindmarshRose_model import HindmarshRose


def pattern(neu, Iext=('input', 2.), dur=1500.):
	# 运行Hindmarsh-Rose模型
	group = neu
	runner = bp.DSRunner(group, monitors=['x', 'y', 'z'], inputs=Iext, dt=0.01)
	runner(dur)

	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 9), sharex='all')

	ax1.plot(runner.mon.ts, runner.mon.z, color=u'#d62728', label='z')
	ax1.set_ylabel('z')
	# ax1.set_ylim(1.5, 2.5)
	ax1.legend(loc='upper right')

	ax2.plot(runner.mon.ts, runner.mon.y, color=u'#ff7f0e', label='y')
	ax2.set_ylabel('y')
	ax2.legend(loc='upper right')

	ax3.plot(runner.mon.ts, runner.mon.x, color=u'#1f77b4', label='x')
	ax3.set_ylabel('x')
	# ax3.set_ylim(-2., 2.)
	ax3.legend(loc='upper right')
	ax3.set_xlabel('t (ms)')

	plt.subplots_adjust(hspace=0.)
	plt.show()


# bursting
pattern(HindmarshRose(1, b=6.), dur=500)


def pattern_xy(neu, Iext=('input', 2.), dur=1500.):
	# 运行Hindmarsh-Rose模型
	group = neu
	runner = bp.DSRunner(group, monitors=['x', 'y', 'z'], inputs=Iext, dt=0.01)
	runner(dur)

	fig, (ax2, ax3) = plt.subplots(2, 1, sharex='all')

	ax2.plot(runner.mon.ts, runner.mon.y, color=u'#ff7f0e', label='y')
	ax2.set_ylabel('y')
	ax2.legend(loc='upper right')

	ax3.plot(runner.mon.ts, runner.mon.x, color=u'#1f77b4', label='x')
	ax3.set_ylabel('x')
	ax3.set_ylim(-2., 2.)
	ax3.legend(loc='upper right')
	ax3.set_xlabel('t (ms)')

	plt.subplots_adjust(hspace=0.)
	plt.show()


# # bursting
# pattern_xy(HindmarshRose(1, s=4.))


def pattern2(neu, Iext=('input', 2.), dur=500):
	# 运行Hindmarsh-Rose模型
	group = neu
	runner = bp.DSRunner(group, monitors=['x', 'y', 'z'], inputs=Iext, dt=0.01)
	runner(dur)

	fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, gridspec_kw={'height_ratios': [3, 3, 3, 1]},
	                                         figsize=(9, 10), sharex='all')

	ax1.plot(runner.mon.ts, runner.mon.z, color=u'#d62728', label='z')
	ax1.set_ylabel('z')
	# ax1.set_ylim(1.5, 2.5)
	ax1.legend(loc='upper right')

	ax2.plot(runner.mon.ts, runner.mon.y, color=u'#ff7f0e', label='y')
	ax2.set_ylabel('y')
	ax2.legend(loc='upper right')

	ax3.plot(runner.mon.ts, runner.mon.x, color=u'#1f77b4', label='x')
	ax3.set_ylabel('x')
	ax3.set_ylim(-2., 2.)
	ax3.legend(loc='upper right')

	ax4.plot(runner.mon.ts, Iext[1], color=u'#2ca02c', label='I')
	ax4.set_ylabel('I')
	ax4.set_xlabel('t (ms)')
	ax4.legend(loc='upper right')

	plt.subplots_adjust(hspace=0.)
	plt.show()


# # adaptation
# I, d = bp.inputs.section_input(values=[0., 2., 0.], durations=[40., 5., 200.], return_length=True, dt=0.01)
#
# pattern2(HindmarshRose(1, s=0.5), Iext=('input', I, 'iter'), dur=d)
