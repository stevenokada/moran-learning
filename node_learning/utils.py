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
NUMBER_OF_RUNS = 1000
NUMBER_OF_THREADS = 5
REP_DIMENSIONS = 128



MONGO_URI = "mongodb://skokada:12345@ds115350.mlab.com:15350/moran"

client_1 = pymongo.MongoClient(MONGO_URI)
db_1 = client_1['moran']
NODE_EMBEDDINGS_COLLECTION = db_1['node_embeddings_F%s'%(str(FITNESS))]

client_2 = pymongo.MongoClient(MONGO_URI)
db_2 = client_2['moran']
NODE_DATA_FINAL = db_2['NODE_DATA_FINAL_F%s'%(str(FITNESS))]


TARGETS = db_2['TARGETS_F%s'%str(FITNESS)]




GRAPHWAVE_DATA_FINAL = db_2['GRAPHWAVE_F%s'%str(FITNESS)]




# node2vec hyperparamtertuning











LABELS = {'S': 0, 'A': 1}
TARGET_ORDER = ['classification']



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
		'dorogovtsev_goltsev_mendes_graph-2', #6
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
		for k in [2,4,6]:
			random = random + gen_rg_names('newman_watts_strogatz_graph', 3, n,k, p_val)

	for m in [1]:
		random = random + gen_rg_names('barabasi_albert_graph', 10,n, m)



	builtins = builtins + random

	return builtins


def get_builtins_medium():
	# def get_builtins(): 
	builtins = [
	'balanced_tree-2_5',
		'barbell_graph-50_50',
		'complete_graph-100',
		'complete_multipartite_graph-25_25_25_25',
		'circular_ladder_graph-50',
		'cycle_graph-100',
		'dorogovtsev_goltsev_mendes_graph-5',
		'grid_2d_graph-10_10',
		'hypercube_graph-6',
		'ladder_graph-50',
		'lollipop_graph-50_50',
		'path_graph-100',
		'star_graph-100',
		'wheel_graph-100',

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


	n = 100

	p_values = []
	# p_values.append(1/(2*n)) #np < 1
	# p_values.append(1/n) #np = 1
	# p_values.append(2/n) #np > 1
	# p_values.append(np.log(n)/(2*n)) #p < ln(n)/n
	# p_values.append(np.log(n)/n)#p == ln(n)/n
	p_values.append(2*np.log(n)/n) #p > ln(n)/n

	random = []
	for p_val in p_values:
		random= random + gen_rg_names('erdos_renyi_graph', 10, 100, p_val)

	for p_val in p_values:
		for k in [2,6]:
			random = random + gen_rg_names('newman_watts_strogatz_graph', 5, 100, k, p_val)

	for m in [1, 2]:
		random = random + gen_rg_names('barabasi_albert_graph', 5, 100, m)



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


def get_connected_builtins(builtins):
	connected_builtins = []

	for graph in builtins:
		try:
			if not nx.is_connected(string_to_graph(graph)):
				raise GraphDisconnectedError()

			connected_builtins.append(graph)

		except GraphDisconnectedError:
			print("%s is disconnected"%graph)

	return connected_builtins


# BUILTINS = get_connected_builtins()
# GIANT_BUILTINS = ['erdos_renyi_graph-500_0.013-seed=5', 'path_graph-500', 'star_graph-499'] #,'barabasi_albert_graph-5000_5-seed=0']


def calculate_pr(fitness, number_of_nodes):
	return (1-1/fitness)/(1-(1/fitness)**number_of_nodes)



def create_featureless_col():
	col = NODE_DATA_FINAL
	featureless_col = TARGETS


	for x in col.find():
		new_bson = dict(x)

		node_vectors = new_bson['node_vectors']
		new_bson['rep_dim'] = '?'

		for node_index in node_vectors:
			node_vectors[node_index]['feature_vec'] = []

		featureless_col.insert(new_bson)


GRAPHWAVE_GRAPHS = ['grid_2d_graph-10_10',
 'newman_watts_strogatz_graph-100_6_0.0921034037198-seed=3', 
 'newman_watts_strogatz_graph-100_6_0.0921034037198-seed=4', 
 'tutte_graph', 
 'balanced_tree-2_5', 
 'circular_ladder_graph-50', 
 'lollipop_graph-50_50', 
 'newman_watts_strogatz_graph-100_2_0.0921034037198-seed=1',
  'newman_watts_strogatz_graph-100_2_0.0921034037198-seed=4', 
  'newman_watts_strogatz_graph-100_6_0.0921034037198-seed=0', 
  'newman_watts_strogatz_graph-100_6_0.0921034037198-seed=2', 
  'barbell_graph-50_50', 
  'cycle_graph-100', 
  'ladder_graph-50', 
  'path_graph-100', 
  'newman_watts_strogatz_graph-100_2_0.0921034037198-seed=0',
   'newman_watts_strogatz_graph-100_2_0.0921034037198-seed=2', 
   'newman_watts_strogatz_graph-100_2_0.0921034037198-seed=3', 
   'barabasi_albert_graph-100_1-seed=0', 
   'barabasi_albert_graph-100_1-seed=3']

def filter(collection, list_of_graphs):
	# removes documents that don't have graph_name in list_of_graphs
	collection.remove({"graph_name": {"$nin": list_of_graphs}})


def find_mismatch(collection1, collection2):
	# returns in collection 1 not in collection2

	not_found = []
	for x in collection1.find():
		if collection2.find_one({"graph_name": x['graph_name']}) is None:
			not_found.append(x['graph_name'])

	return not_found

def update_with_targets(collection):

	for x in collection.find():
		node_vectors = x['node_vectors']

		for node_index in node_vectors:

			graph_targets = utils.TARGETS.find_one({'graph_name': x['graph_name']})
			target_vec = graph_targets['node_vectors'][str(node_index)]



			collection.update_one({'graph_name': x['graph_name']}, {'$set': {'node_vectors.%s.target_vec'%node_index: target_vec}})
