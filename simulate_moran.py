from __future__ import division
import networkx as nx
import pandas as pd
import numpy as np

import argparse
import matplotlib

import matplotlib.pyplot as plt


from pymongo import MongoClient
from collections import Counter


class OverstepError(Exception):
	pass

class MoranSimulation():
	def __init__(self, datapath, fitness_val, fitness_dist = 'uniform'):
		"""datapath is pandas df without header and without edge weights

		fitness_dist == fitness distrubtion
		fitness_val = fitness of one mutant
		"""


		self.datapath = datapath
		self.status = 'ongoing'
		self.time_step = 0
		self.fitness_val = fitness_val


		# unfiorm fitness except for mutant
		if fitness_dist == 'uniform':
			# each node is (label, fitness) tuple

			df = pd.read_csv(self.datapath, sep = ' ',header = None)
			edgelist = df.values.tolist()

			self.graph = nx.Graph(edgelist)

			self.graph = nx.relabel_nodes(self.graph, lambda x: (x,1))


			# get node and update node

			replace_index = np.random.randint(1, high=nx.number_of_nodes(self.graph))

			print(replace_index, fitness_val)
			self.graph = nx.relabel_nodes(self.graph, {(replace_index,1): (replace_index, fitness_val)}, copy=False)

			self.pos = nx.spring_layout(self.graph)



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

		print("%s achieved at %d" % (self.status, self.time_step))




				
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


		# assuming no edge weights, if we do edge weights, we should select 

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



def aggregate_run(datapath, fitness_val,  number_of_runs, fitness_dist = 'uniform'):

	successes = []
	for i in range(number_of_runs):
		msim = MoranSimulation(datapath, fitness_val, fitness_dist=fitness_dist)
		status, termination_time = msim.run()

		if status == 'fixation':
			successes.append(termination_time)


	p_success = len(successes)/number_of_runs
	f_time = np.mean(successes)

	return p_success, f_time












# moransim = MoranSimulation('data/karate/karate.edgelist', 5)
# moransim.run(visualize=True)

# if __name__ == '__main__':
# 	parser = argparse.ArgumentParser()

# 	parser.add_argument("--input", help="datapath to input csv file")
# 	parser.add_argument("--output", help="datapath to output csv file")
# 	parser.add_argument("--fitness", nargs='+')
# 	parser.add_argument("--number_of_runs", )


# 	args = parser.parse_args()

# 	i_datapath = args.input 
# 	o_datapath = args.output


# 	results = []

# 	for i in args.fitness:
# 		i = float(i)

# 		sub_results = []


# 		for j in range(int(args.number_of_runs)):

# 			print("Run number %d for %s " % (j+1, i_datapath))
# 			msim = MoranSimulation(i_datapath, i)

# 			sub_results.append(msim.run())


# 		# results contains (status, time)

# 		# calculate prob of success and average time of success

# 		successes = list(filter(lambda x: x[0] == 'fixation', sub_results))

# 		p_success = len(successes)/len(sub_results)

# 		mean_f_time = np.mean(list(map(lambda x: x[1], successes)))

# 		results.append((i, p_success, mean_f_time))



# 	print(results)



	#write bash script to run for every file 
	# we need graph_f vec
	# open o_datapath (all files should use results.csv)
	# lookup rep_vector ofassociated with i_datapath
	# write row to o_datapath with (i_datapath, rep_vector, p_success @ f=1.1, mean_f_time @ f=1.1, p_success @ f= 1.2, ...)








