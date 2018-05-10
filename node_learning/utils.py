from __future__ import division
import utils
import subprocess
import glob
import numpy as np
import jsonschema
import pymongo
import datetime
import networkx as nx
import pandas as pd
import csv
import multiprocessing as mp
import time
import ndistributed_simulate_moran as ndsmoran
import pickle


FITNESS = 5
NUMBER_OF_RUNS = 200
NUMBER_OF_THREADS = 5
MONGO_URI = "mongodb://skokada:12345@ds115350.mlab.com:15350/moran"

client_1 = pymongo.MongoClient(MONGO_URI)
db_1 = client_1['moran']
NODE_EMBEDDINGS_COLLECTION = db_1['node_embeddings_F%s'%(str(FITNESS))]

client_2 = pymongo.MongoClient(MONGO_URI)
db_2 = client_2['moran']
NODE_DATA_FINAL = db_2['NODE_DATA_FINAL_F%s'%(str(FITNESS))]





I_DATAPATH = "data/edgelists/"
N_DATAPATH = "data/node_embeddings/"
G_DATAPATH = "data/graph_embeddings/"



SCHEMA = {"type": "object", "properties": {'graph_name': {"type": "string"}, 'rep_dim': {'type': 'int'},'node_vectors': {"type": "object"}, 'num_edges':{'type':'int'}
 ,'num_nodes': {'type': 'int'}, 'num_runs': {'type':'int'}, 'f_val': {'type': 'number'}, 'adj_matrix':{'type':'string'}}, 'required': ['graph_name', 'rep_dim','node_vectors', 'num_edges', 'num_nodes', 'num_runs','f_val']}

class OverstepError(Exception):
	pass

class GraphDisconnectedError(Exception):
	pass

def get_builtins_small():
	builtins = [
	'balanced_tree-2_2', #7
		'barbell_graph-3_3', #9
		'complete_graph-10', #10
		'complete_multipartite_graph-3_3_3', #9
		'circular_ladder_graph-5',#10
		'cycle_graph-10', #10
		'dorogovtsev_goltsev_mendes_graph-5', #6
		'grid_2d_graph-3_3', #9
		'hypercube_graph-3', #8
		'ladder_graph-5', #10
		'lollipop_graph-5_5', #10
		'path_graph-10', #10
		'star_graph-9', #10
		'wheel_graph-10', #10

		'bull_graph',
		'chvatal_graph',
		'desargues_graph',
		'diamond_graph',
		'dodecahedral_graph',
		'frucht_graph',
		'heawood_graph',
		'house_x_graph',
		'icosahedral_graph',
		'krackhardt_kite_graph',
		'moebius_kantor_graph',
		'octahedral_graph',
		'pappus_graph',
		'petersen_graph',
		'sedgewick_maze_graph',
		'tetrahedral_graph',
		'truncated_cube_graph',
		'truncated_tetrahedron_graph',
		'tutte_graph',
		'karate_club_graph',
		'davis_southern_women_graph',
		'florentine_families_graph',
		]


	def gen_rg_names(constructor_name, number_of_runs, *args):

		args = list(map(lambda x: str(x), args))
		result = []



		for x in range(number_of_runs):

			ans = constructor_name + '-' + '_'.join(args) + '-' + 'seed=' + str(x)

			result.append(ans)

		return result


	n = 10

	p_values = []
	p_values.append(1/(2*n)) #np < 1
	# p_values.append(1/n) #np = 1
	p_values.append(2/n) #np > 1
	p_values.append(np.log(n)/(2*n)) #p < ln(n)/n
	# p_values.append(np.log(n)/n)#p == ln(n)/n
	p_values.append(2*np.log(n)/n) #p > ln(n)/n

	random = []
	for p_val in p_values:
		random= random + gen_rg_names('erdos_renyi_graph', 10,n,p_val)

	for p_val in p_values:
		for k in [1,2,3]:
			random = random + gen_rg_names('newman_watts_strogatz_graph', 3, n,k, p_val)

	for m in [1]:
		random = random + gen_rg_names('barabasi_albert_graph', 10,n, m)



	builtins = builtins + random

	return builtins



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
		return nx.convert_node_labels_to_integers(getattr(nx, constructor_name)())

	if len(args) == 2:
		# builtin with *args
		params = args[1].split('_')
		params = list(map(convert_arg, params))

		return nx.convert_node_labels_to_integers(getattr(nx, constructor_name)(*params))

	if len(args) == 3:
		params = args[1].split('_')
		params = list(map(convert_arg, params))

		seed = args[2].split('=')[1]
		seed = int(seed)
		return nx.convert_node_labels_to_integers(getattr(nx, constructor_name)(*params, seed=seed))


def get_connected_builtins():
	connected_builtins = []

	for graph in get_builtins_small():
		try:
			if not nx.is_connected(string_to_graph(graph)):
				raise GraphDisconnectedError()

			connected_builtins.append(graph)

		except GraphDisconnectedError:
			print("%s is disconnected"%graph)

	return connected_builtins


BUILTINS = get_connected_builtins()


def calculate_pr(fitness, number_of_nodes):
	return (1-1/fitness)/(1-(1/fitness)**number_of_nodes)



def mongo_to_csv_target():
	client = MongoClient(utils.MONGO_URI)
	db = client['moran']
	collection = db['DATAF' + str(FITNESS) + '_FINAL']

	builtins = get_builtins_small()


	listofdicts = []
	counter = 0
	for obj in collection.find():


		target_vec = obj['target_vec']
		target_vec['graph_name'] = obj['graph_name']
		counter += 1

		# new_target_vec = {}
		# for k,v in target_vec.items():
		# 	new_target_vec[k] = [v]

		listofdicts.append(target_vec)




	print(counter)
	df = pd.DataFrame(listofdicts)
	df.to_csv('data/F%stargets.csv'%str(FITNESS), sep=' ')