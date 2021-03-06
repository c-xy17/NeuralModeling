# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np


class SeparateNaK(bp.dyn.DynamicalSystem):
	def __init__(self, num=1, ENa=50., gNa=120., EK=-77., gK=36., EL=-54.387,
	             gL=0.03, V_th=20., C=1.0):
		# 初始化
		super(SeparateNaK, self).__init__()

		# 定义神经元参数
		self.ENa = ENa
		self.EK = EK
		self.EL = EL
		self.g_Na_max = gNa
		self.g_K_max = gK
		self.gL = gL
		self.C = C
		self.V_th = V_th

		# 定义神经元变量
		self.V = bm.Variable(bm.asarray([-70.67647] * num))  # 膜电位
		self.m = bm.Variable(bm.asarray([0.02657777] * num))  # 离子通道m
		self.h = bm.Variable(bm.asarray([0.77206403] * num))  # 离子通道h
		self.n = bm.Variable(bm.asarray([0.23536022] * num))  # 离子通道n
		self.gNa = bm.Variable(bm.zeros(num))
		self.gK = bm.Variable(bm.zeros(num))
		self.INa = bm.Variable(bm.zeros(num))
		self.IK = bm.Variable(bm.zeros(num))
		self.IL = bm.Variable(bm.zeros(num))
		self.Ifb = bm.Variable(bm.zeros(num))

		# 定义积分函数
		self.integral = bp.odeint(f=self.derivative, method='exponential_euler')

	def derivative(self, m, h, n, t, ):
		alpha = 0.1 * (self.V + 40) / (1 - bm.exp(-(self.V + 40) / 10))
		beta = 4.0 * bm.exp(-(self.V + 65) / 18)
		dmdt = alpha * (1 - m) - beta * m

		alpha = 0.07 * bm.exp(-(self.V + 65) / 20.)
		beta = 1 / (1 + bm.exp(-(self.V + 35) / 10))
		dhdt = alpha * (1 - h) - beta * h

		alpha = 0.01 * (self.V + 55) / (1 - bm.exp(-(self.V + 55) / 10))
		beta = 0.125 * bm.exp(-(self.V + 65) / 80)
		dndt = alpha * (1 - n) - beta * n

		return dmdt, dhdt, dndt

	def update(self, _t, _dt):
		m, h, n = self.integral(self.m, self.h, self.n, _t, dt=_dt)
		self.m.value = m
		self.h.value = h
		self.n.value = n
		self.gNa.value = self.g_Na_max * self.m ** 3 * self.h
		self.gK.value = self.g_K_max * self.n ** 4
		self.INa.value = self.gNa * (self.V - self.ENa)
		self.IK.value = self.gK * (self.V - self.EK)
		self.IL.value = self.gL * (self.V - self.EL)
		self.Ifb = self.INa + self.IK + self.IL


def try_steady_state():
	runner = bp.DSRunner(SeparateNaK(), monitors=['m', 'h', 'n', 'INa', 'IK', 'IL', 'Ifb'])
	runner.run(100.)

	bp.visualize.line_plot(runner.mon.ts, runner.mon.m)
	bp.visualize.line_plot(runner.mon.ts, runner.mon.h)
	bp.visualize.line_plot(runner.mon.ts, runner.mon.n, show=True)


def try_step_voltage():
	vs, duration = bp.inputs.section_input([-70.67647, -70.67647 + 56], durations=[1, 9],
	                                       return_length=True)
	runner = bp.DSRunner(SeparateNaK(), monitors=['m', 'h', 'n',
	                                              'INa', 'IK', 'IL', 'Ifb',
	                                              'gNa', 'gK'],
	                     inputs=['V', vs, 'iter', '='])
	runner.run(duration)

	fig, gs = bp.visualize.get_figure(1, 1, 4, 8)
	bp.visualize.line_plot(runner.mon.ts, runner.mon.m)
	bp.visualize.line_plot(runner.mon.ts, runner.mon.h)
	bp.visualize.line_plot(runner.mon.ts, runner.mon.n, show=True)
	fig, gs = bp.visualize.get_figure(1, 1, 4, 8)
	bp.visualize.line_plot(runner.mon.ts, runner.mon.INa, legend='INa')
	bp.visualize.line_plot(runner.mon.ts, runner.mon.IK, legend='IK')
	bp.visualize.line_plot(runner.mon.ts, runner.mon.Ifb, legend='Ifb', show=True)


def try_step_voltage2():
	steps = bm.asarray([41, 55, 70, 84, 99, 113, 127])
	steps = bm.asarray([6, 10, 20, 35, 50, 75, 100])
	vs, duration = bp.inputs.section_input([-70.67647, -70.67647 + steps],
	                                       durations=[1, 20],
	                                       return_length=True)
	runner = bp.DSRunner(SeparateNaK(steps.size), monitors=['INa', 'IK', 'IL', 'Ifb', 'gNa', 'gK'],
	                     inputs=['V', vs, 'iter', '='])
	runner.run(duration)

	fig, gs = bp.visualize.get_figure(steps.size, 2, 1, 4)
	for i in range(steps.size):
		fig.add_subplot(gs[i, 0])
		plt.plot(runner.mon.ts, runner.mon.gNa[:, i], )
		plt.ylabel(f'+{steps[i]} mV')
		if i == 0:
			plt.title('gNa')
		if i == steps.size - 1:
			plt.xlabel('Time [ms]')
		else:
			plt.xticks([])

		fig.add_subplot(gs[i, 1])
		plt.plot(runner.mon.ts, runner.mon.gK[:, i], 'r')
		if i == 0:
			plt.title('gK')
		if i == steps.size - 1:
			plt.xlabel('Time [ms]')
		else:
			plt.xticks([])
	plt.show()


def try_step_voltage_for_gNa():
	dt = 0.01
	step = 60
	vs, duration = bp.inputs.section_input(
		[-70.67647, -70.67647 + step,
		 -70.67647, -70.67647 + step,
		 -70.67647],
		durations=[1, 1.8,
		           25, 1.8,
		           10],
		return_length=True,
		dt=dt)
	runner = bp.DSRunner(SeparateNaK(1),
	                     monitors=['INa', 'IK', 'IL', 'Ifb', 'gNa', 'gK',
	                               'm', 'h', 'n'],
	                     inputs=['V', vs, 'iter', '='],
	                     dt=dt)
	runner.run(duration)

	fig, gs = bp.visualize.get_figure(3, 1, 2, 6)
	fig.add_subplot(gs[0, 0])
	bp.visualize.line_plot(runner.mon.ts, runner.mon.gNa, legend='gNa')
	fig.add_subplot(gs[1, 0])
	bp.visualize.line_plot(runner.mon.ts, runner.mon.m, legend='m')
	fig.add_subplot(gs[2, 0])
	bp.visualize.line_plot(runner.mon.ts, runner.mon.h, legend='h', show=True)


def INa_inactivation():
	dt = 0.01
	vs, duration = bp.inputs.section_input(
		[-70.67647, -70.67647 + 8, -70.67647 + 44],
		durations=[1, 10, 20],
		return_length=True,
		dt=dt)
	runner = bp.DSRunner(SeparateNaK(1),
	                     monitors=['INa', 'IK', 'IL', 'Ifb', 'gNa', 'gK',
	                               'm', 'h', 'n'],
	                     inputs=['V', vs, 'iter', '='],
	                     dt=dt)
	runner.run(duration)

	fig, gs = bp.visualize.get_figure(3, 2, 2, 5)
	fig.add_subplot(gs[0, 0])
	bp.visualize.line_plot(runner.mon.ts, runner.mon.Ifb, legend='Ifb')
	fig.add_subplot(gs[1, 0])
	bp.visualize.line_plot(runner.mon.ts, runner.mon.INa, legend='INa')
	fig.add_subplot(gs[2, 0])
	bp.visualize.line_plot(runner.mon.ts, runner.mon.IK, legend='IK')

	fig.add_subplot(gs[0, 1])
	bp.visualize.line_plot(runner.mon.ts, runner.mon.gNa, legend='gNa')
	bp.visualize.line_plot(runner.mon.ts, runner.mon.gK, legend='gK')
	fig.add_subplot(gs[1, 1])
	bp.visualize.line_plot(runner.mon.ts, runner.mon.m, legend='m')
	bp.visualize.line_plot(runner.mon.ts, runner.mon.h, legend='h')
	fig.add_subplot(gs[2, 1])
	bp.visualize.line_plot(runner.mon.ts, runner.mon.n, legend='n', show=True)


def INa_inactivation_2_0():
	dt = 0.01
	duration = 50
	lengths = [0, 5, 10, 20]

	all_vs = []
	for l in lengths:
		vs = bp.inputs.section_input([-70.67647, -70.67647 + 8, -70.67647 + 44],
		                             durations=[1, l, duration - l], dt=dt)
		all_vs.append(vs)
	all_vs = bm.vstack(all_vs).T

	runner = bp.DSRunner(SeparateNaK(len(lengths)),
	                     monitors=['INa', 'IK', 'IL', 'Ifb', 'gNa', 'gK', 'm', 'h', 'n'],
	                     inputs=['V', all_vs, 'iter', '='],
	                     dt=dt)
	runner.run(duration + 1)
	all_vs = all_vs.numpy()

	fig, gs = bp.visualize.get_figure(2, 2, 3, 4)

	for i in range(4):
		ax = fig.add_subplot(gs[i // 2, i % 2])
		ax.plot(runner.mon.ts, -runner.mon.INa[:, i], 'b')
		plt.title(f't = {lengths[i]} ms')
		ax.set_ylim([0, 1400])
		ax.tick_params('y', colors='b')
		ax.set_ylabel(r'$I_{\mathrm{Na}}$', color='b')
		ax = ax.twinx()
		ax.plot(runner.mon.ts, all_vs[:, i], 'r')
		ax.tick_params('y', colors='r')
		ax.set_ylim([-100, 100])
		ax.set_ylabel(r'Membrane potential', color='r')

	plt.show()


def INa_inactivation_2():
	dt = 0.01
	duration = 50
	lengths = list(range(0, 26, 1))
	v_list = [25, 20, 10, -10, -20, -30]

	for v in v_list:
		all_vs = []
		for l in lengths:
			vs = bp.inputs.section_input([-70.67647, -70.67647 + v, -70.67647 + 44],
			                             durations=[1, l, duration - l], dt=dt)
			all_vs.append(vs)
		all_vs = bm.vstack(all_vs).T

		runner = bp.DSRunner(SeparateNaK(len(lengths)),
		                     monitors=['INa', 'IK', 'IL', 'Ifb', 'gNa', 'gK', 'm', 'h', 'n'],
		                     inputs=['V', all_vs, 'iter', '='],
		                     dt=dt)
		runner.run(duration)

		base = runner.mon.INa[:, 0].min()
		x = runner.mon.INa

		all_min = runner.mon.INa.min(axis=0)
		plt.plot(lengths, all_min / base, label=f'$v_1$={-v} mV')
	plt.legend()
	plt.xlabel(r'$t$ [ms]')
	plt.ylabel(r'$\frac{(I_{Na})_{v_1}}{(I_{Na})_{v_0}}$', fontdict={'size': 12}, rotation=360)
	# ax.yaxis.set_label_coords(-0.1, 0.5)
	# plt.ylim(-0.1, 0.5)
	plt.show()


def INa_inactivation_steady_state1():
	dt = 0.01
	steps = bm.linspace(-30, 50, 9)
	vs, duration = bp.inputs.section_input([-70.67647, -70.67647 + steps, -70.67647 + 44],
	                                       durations=[5, 30, 20], dt=dt,
	                                       return_length=True)
	runner = bp.DSRunner(SeparateNaK(len(steps)),
	                     monitors=['INa', 'IK', 'IL', 'Ifb', 'gNa', 'gK', 'm', 'h', 'n'],
	                     inputs=['V', vs, 'iter', '='],
	                     dt=dt)
	runner.run(duration)
	vs = vs.numpy()

	fig, gs = bp.visualize.get_figure(3, 3, 3, 4)
	for i in range(9):
		ax = fig.add_subplot(gs[i // 3, i % 3])
		ax.plot(runner.mon.ts, -runner.mon.INa[:, i], 'b')
		plt.title(f'$\\Delta v$ = {steps[i]} mV')
		ax.set_ylim([-100, 1700])
		ax.tick_params('y', colors='b')
		ax.set_ylabel(r'$I_{\mathrm{Na}}$', color='b')
		ax = ax.twinx()
		ax.plot(runner.mon.ts, vs[:, i], 'r')
		ax.tick_params('y', colors='r')
		ax.set_ylim([-120, 100])
		ax.set_ylabel(r'Membrane potential', color='r')
	plt.show()


def INa_inactivation_steady_state2():
	dt = 0.01
	steps = bm.linspace(-60, 60, 201)
	steps = bm.hstack([0, steps])
	vs, duration = bp.inputs.section_input([-70.67647, -70.67647 + steps, -70.67647 + 44],
	                                       durations=[5, 30, 20], dt=dt,
	                                       return_length=True)
	runner = bp.DSRunner(SeparateNaK(len(steps)),
	                     monitors=['INa', 'IK', 'IL', 'Ifb', 'gNa', 'gK', 'm', 'h', 'n'],
	                     inputs=['V', vs, 'iter', '='],
	                     dt=dt)
	runner.run(duration)

	start = int(30 / dt)
	base = runner.mon.INa[start:, 0].min() - runner.mon.INa[-1, 0]
	scales = (runner.mon.INa[start:, 1:].min(axis=0) - runner.mon.INa[-1, 1:]) / base

	fig, gs = bp.visualize.get_figure(1, 1, 5, 6)
	ax = fig.add_subplot(gs[0, 0])
	ax.plot(steps.numpy()[1:], scales)
	ax.set_xlabel(r'$\Delta V$ [mV]', fontdict={'size': 12})
	ax.set_ylabel(r'$\frac{I_{Na}(\Delta V)}{I_{Na}(0)}$', fontdict={'size': 16}, rotation=360)
	ax.yaxis.set_label_coords(-0.13, 0.5)
	plt.ylim([-scales.max() * .05, scales.max() * 1.05])
	plt.axvline(0, linestyle='--', color='grey')
	plt.axhline(1, linestyle='--', color='grey')
	ax = ax.twinx()
	plt.yticks(bm.linspace(0, scales.max(), 11).numpy(), np.around(np.linspace(0, 1, 11), 1))
	ax.set_ylabel('$h_\infty(V+\Delta V)$', fontdict={'size': 12}, rotation=360)
	ax.yaxis.set_label_coords(1.18, 0.55)
	plt.ylim([-scales.max() * .05, scales.max() * 1.05])
	plt.show()


def n_tau_inf_alpha_beta():
	V = bm.arange(-110, 50, 0.1)
	alpha = 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
	beta = 0.125 * bm.exp(-(V + 65) / 80)
	tau = 1 / (alpha + beta)
	inf = alpha / (alpha + beta)
	fig, gs = bp.visualize.get_figure(2, 2, 3, 4.5)
	fig.add_subplot(gs[0, 0])
	plt.plot(V.numpy(), tau.numpy())
	plt.title(r'$\tau_n$')
	fig.add_subplot(gs[0, 1])
	plt.plot(V.numpy(), inf.numpy())
	plt.title(r'$n_\infty$')
	fig.add_subplot(gs[1, 0])
	plt.plot(V.numpy(), alpha.numpy())
	plt.title(r'$\alpha(n)$')
	plt.xlabel('V (mV)')
	fig.add_subplot(gs[1, 1])
	plt.plot(V.numpy(), beta.numpy())
	plt.title(r'$\beta(n)$')
	plt.xlabel('V (mV)')
	plt.show()


def Ina_tau_inf_alpha_beta():
	V = bm.arange(-110, 40, 0.1)
	alpha_m = 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
	beta_m = 4.0 * bm.exp(-(V + 65) / 18)
	tau_m = 1 / (alpha_m + beta_m)
	inf_m = alpha_m / (alpha_m + beta_m)

	alpha_h = 0.07 * bm.exp(-(V + 65) / 20.)
	beta_h = 1 / (1 + bm.exp(-(V + 35) / 10))
	tau_h = 1 / (alpha_h + beta_h)
	inf_h = alpha_h / (alpha_h + beta_h)

	V = V.numpy()

	plt.figure()
	plt.plot(V, tau_m, label='tau_m')
	plt.plot(V, tau_h, label='tau_h')
	plt.legend()

	plt.figure()
	plt.plot(V, inf_m, label='inf_m')
	plt.plot(V, inf_h, label='inf_h')
	plt.legend()
	plt.show()


def h_tau_inf_alpha_beta():
	V = bm.arange(-110, 40, 0.1)
	alpha_m = 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
	beta_m = 4.0 * bm.exp(-(V + 65) / 18)
	tau_m = 1 / (alpha_m + beta_m)
	inf_m = alpha_m / (alpha_m + beta_m)

	alpha_h = 0.07 * bm.exp(-(V + 65) / 20.)
	beta_h = 1 / (1 + bm.exp(-(V + 35) / 10))
	tau_h = 1 / (alpha_h + beta_h)
	inf_h = alpha_h / (alpha_h + beta_h)

	V = V.numpy()

	plt.figure()
	plt.plot(V, tau_m, label='tau_m')
	plt.plot(V, tau_h, label='tau_h')
	plt.legend()

	plt.figure()
	plt.plot(V, inf_m, label='inf_m')
	plt.plot(V, inf_h, label='inf_h')
	plt.legend()
	plt.show()


if __name__ == '__main__':
	pass
	# try_step_voltage()
	# try_step_voltage2()
	# try_step_voltage_for_gNa()
	# INa_inactivation()
	# INa_inactivation_2_0()
	# INa_inactivation_2()
	# INa_inactivation_steady_state1()
	INa_inactivation_steady_state2()
	# Ina_tau_inf_alpha_beta()
	# n_tau_inf_alpha_beta()
	# Ina_tau_inf_alpha_beta()
