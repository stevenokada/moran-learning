from __future__ import division
import networkx as nx
import pandas as pd
import numpy as np

import argparse
import matplotlib

import matplotlib.pyplot as plt

import distributed_gen_target as dgt
from pymongo import MongoClient
from collections import Counter




class OverstepError(Exception):
	pass

class GraphDisconnectedError(Exception):
	pass

def string_to_graph(s_input):

	def convert_arg(arg):
		try:
			arg = int(arg)

		except ValueError:
			arg = float(arg)

		return arg


	args = s_input.split('-')	

	constructor_name = args[0]

	if len(args) == 1:
		# builtin with no arguments
		return getattr(nx, constructor_name)()

	if len(args) == 2:
		# builtin with *args
		params = args[1].split('_')
		params = map(convert_arg, params)

		return getattr(nx, constructor_name)(*params)

	if len(args) == 3:
		params = args[1].split('_')
		params = map(convert_arg, params)

		seed = args[2].split('=')[1]
		seed = int(seed)
		return getattr(nx, constructor_name)(*params, seed=seed)


def get_connected_builtins():
	connected_builtins = []

	for graph in dgt.get_builtins():
		try:
			if not nx.is_connected(string_to_graph(graph)):
				raise GraphDisconnectedError()

			connected_builtins.append(graph)

		except GraphDisconnectedError:
			print("%s is disconnected"%graph)

	return connected_builtins






class MoranSimulation():
	def __init__(self, datapath, fitness_val, graph_type = 'data', fitness_dist = 'uniform'):
		"""datapath is pandas df without header and without edge weights

		fitness_dist == fitness distrubtion
		fitness_val = fitness of one mutant
		"""

		# MUST ONLY DEAL WITH CONNECTED GRAPH


		self.datapath = datapath
		self.status = 'ongoing'
		self.time_step = 0
		self.fitness_val = fitness_val



		# if graph_type == 'data':
		# 	df = pd.read_csv(self.datapath, sep = ' ',header = None)
		# 	edgelist = df.values.tolist()
		# 	self.graph = nx.Graph(edgelist)

		if graph_type == 'builtin':

			self.graph = nx.convert_node_labels_to_integers(string_to_graph(datapath))

			if not nx.is_connected(self.graph):
				raise GraphDisconnectedError()



		self.graph = nx.relabel_nodes(self.graph, lambda x: (x,1))

		if fitness_dist == 'uniform':

			replace_index = np.random.randint(1, high=nx.number_of_nodes(self.graph))


		self.graph = nx.relabel_nodes(self.graph, {(replace_index,1): (replace_index, fitness_val)}, copy=False)
		self.pos = nx.spring_layout(self.graph)

		self.number_of_nodes = len(self.graph.nodes())
		self.number_of_edges = len(self.graph.edges())



			# colors = ['r' if x[1] != 1 else 'b' for x in self.graph.nodes()]
			# nx.draw(self.graph, with_labels=True, node_color = colors, pos= self.pos)
			# plt.show()


	def get_node(self, node_id):
		#make this more efficient
		for node in self.graph.nodes():
			if node_id == node[0]:
				return node

	def update_pos(self):
		#make this more efficient
		nodes = self.graph.nodes()


		new_pos = {}

		for key, value in self.pos.items():

			new_fit = self.get_node(key[0])[1]
			new_pos[(key[0], new_fit)] = value

		self.pos = new_pos


	def draw_graph(self):

		self.update_pos()
		
		colors = ['r' if x[1] != 1 else 'b' for x in self.graph.nodes()]

		nx.draw(self.graph, with_labels=True, node_color = colors, pos= self.pos)
		plt.pause(0.05)
		plt.clf()



	def get_fitness_dist(self):
		nodes = self.graph.nodes()

		fitnesses = [x[1] for x in nodes]
		fitness_dist = dict(Counter(fitnesses))

		return fitness_dist

	def display_frequencies(self):
		
		f_dist = self.get_fitness_dist()




		#sorted ensures mutants is last

		axes = plt.gca()
		axes.set_ylim([0,nx.number_of_nodes(self.graph)])
		plt.bar(["normie", "mutant"], sorted(f_dist.values()), color='g')
		plt.pause(0.05)
		plt.clf()





	def run(self, max_step = float('inf'),visualize=False):

		while self.status == 'ongoing' and self.time_step <= max_step:
			self.step()

			if visualize:

				if not self.time_step % 30:
					print("Step: %d" % self.time_step)
					# self.display_frequencies()
					self.draw_graph()






		self.termination_time = self.time_step
		self.graph = nx.freeze(self.graph)

		# print("%s achieved at %d" % (self.status, self.time_step))




				
		return self.status, self.time_step



	def step(self):
		if self.status != 'ongoing':
			print("%s achieved at %d" % (self.status, self.time_step))
			raise OverstepError()

		self.time_step += 1


		to_reproduce = self.select(self.graph.nodes())
		to_replace = self.select_neighbor(to_reproduce)
		self.replace_node(to_replace, to_reproduce)


		self.update_status()

		return




	def select(self, items):
		"""Returns (node, fitness) tuple"""

		# generates prob distribution over nodes as list


		fitnesses = np.array([x[1] for x in items])


		weights = fitnesses/sum(fitnesses)


		# nodes and weights must be in same order
		selected_index = np.random.choice(len(items), p=weights)

		return items[selected_index]








	def select_neighbor(self, node):

		neighbors = list(nx.all_neighbors(self.graph, node))


		selected_index = np.random.choice(len(neighbors))

		return neighbors[selected_index]




	def replace_node(self, old, new):

		nx.relabel_nodes(self.graph, {old: (old[0], new[1])}, copy=False)

	def update_status(self):
		

		fitness_dist = self.get_fitness_dist()
		

		if len(fitness_dist) == 1 :
			surviving_val = list(fitness_dist.keys())[0]
			if  surviving_val == self.fitness_val:
				self.status = 'fixation'

			else:
				self.status = 'extinction'



def aggregate_run(datapath, fitness_val,  graph_type, number_of_runs, fitness_dist = 'uniform'):

	successes = []
	for i in range(number_of_runs):
		msim = MoranSimulation(datapath, fitness_val, graph_type = graph_type, fitness_dist=fitness_dist)
		status, termination_time = msim.run()

		number_of_nodes = msim.number_of_nodes
		number_of_edges = msim.number_of_edges

		if status == 'fixation':
			successes.append(termination_time)




	p_success = len(successes)/number_of_runs

	f_time = np.mean(successes)

	p_r = (1-1/fitness_val)/(1-(1/fitness_val)**number_of_nodes)

	if p_success < p_r:
		classification = 'S'

	if p_success >= p_r: 
		classification = 'A'

	return p_success, f_time, classification, number_of_nodes, number_of_edges








