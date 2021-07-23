import sys, math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import util
import random
import decimal
import hyperx
import traffic_generator as tg


def drange(x, y, jump):
  while x < y:
    yield float(x)
    x += decimal.Decimal(jump)


def do_routing(dimensions, link_ratio, average_hops_dal, average_hops_safety):
	hx = hyperx.HyperX(len(dimensions), dimensions, 1, 5, link_ratio)
	traffic_mat = tg.generate_uniform_traffic(hx.num_switches)
	(hops_dal, optimal_dal) = hx.route_dal(traffic_mat)
	(hops_safety, optimal_safety) = hx.safety_vectors_route()

	return hops_dal, hops_safety, optimal_dal, optimal_safety


def run():
	dimensions = [3,3,4]
	repeat_times = 10
	faulty_links_ratio = list(drange(0, 0.8, '0.05'))
	print('Links: {}'.format(faulty_links_ratio))

	average_hops_dal = []
	average_hops_safety = []
	average_optimal_dal = []
	average_optimal_safety = []

	dal_hops = 0
	safety_hops = 0
	dal_opt = 0
	safety_opt = 0

	for link_ratio in faulty_links_ratio:
		counter = 0
		while (counter < repeat_times):
			(dal, safety, optimal_dal, optimal_safety) = do_routing(dimensions, link_ratio, average_hops_dal, average_hops_safety)
			dal_hops += dal
			safety_hops += safety
			dal_opt += optimal_dal
			safety_opt += optimal_safety
			counter += 1

		dal_hops /= repeat_times
		safety_hops /= repeat_times
		dal_opt /= repeat_times
		safety_opt /= repeat_times
		average_hops_dal.append(dal_hops)
		average_hops_safety.append(safety_hops)
		average_optimal_dal.append(dal_opt)
		average_optimal_safety.append(safety_opt)


	fig = plt.figure()
	plt.plot(faulty_links_ratio, average_hops_dal)
	plt.plot(faulty_links_ratio, average_hops_safety)
	plt.title('Average hops count as link failure')
	plt.ylabel('Average hops')
	plt.xlabel('Fraction of fallen links')
	plt.legend(['dal', 'safety-vector'])

	fig = plt.figure()
	plt.plot(faulty_links_ratio, average_optimal_dal)
	plt.plot(faulty_links_ratio, average_optimal_safety)
	plt.title('Average optimal routes as link failure')
	plt.ylabel('Average optimal routes')
	plt.xlabel('Fraction of fallen links')
	plt.legend(['dal', 'safety-vector'])

	plt.show()

