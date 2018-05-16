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


def thread_node_sim_upload_general(graph_name, q_lock, q, v_lock, v, return_dict):
	print("%s started" % mp.current_process().name)

	while not q.empty():
		with q_lock:
			node_index = q.get()


		# result = {'p_success':, 'f_time', 'classification': }

		result = ndsmoran.MoranNodeSimulation(graph_name, utils.FITNESS, utils.NUMBER_OF_RUNS).single_node_run(node_index, early_stop=True)

		with v_lock:
			return_dict[str(node_index)] = result


		with v_lock:
			v.value += 1

		print("Node %s is uploaded" % v.value)

	print("%s finished" % mp.current_process().name)


def sim_upload_graphs(list_of_graphs):
	q = mp.Queue()
	q_lock = mp.Lock()
	v = mp.Value('i', 0)
	v_lock = mp.Lock()

	for graph_name in list_of_graphs:
		v.value = 0
		return_dict = mp.Manager().dict()

		graph = utils.string_to_graph(graph_name)
		
		for i in range(graph.number_of_nodes()):
			q.put(i)

		processes = [mp.Process(name="thread %s"%x, target=thread_node_sim_upload_general, args=(graph_name, q_lock, q, v_lock, v, return_dict)) for x in range(utils.NUMBER_OF_THREADS)]

		for process in processes:
			process.start()

		while not q.empty():
			pass

		for process in processes:
			process.join()


		# upload to targets database

		bson_object = {}
		bson_object['graph_name'] = graph_name
		bson_object['adj_matrix'] = pickle.dumps(nx.adjacency_matrix(graph))
		bson_object['rep_dim'] = '?'
		bson_object['f_val'] = utils.FITNESS
		bson_object['num_runs'] = utils.NUMBER_OF_RUNS
		bson_object['num_nodes'] = graph.number_of_nodes()
		bson_object['num_edges'] = graph.number_of_edges()
		bson_object['node_vectors'] = dict(return_dict)


		utils.TARGETS.insert_one(bson_object)



if '__main__' == __name__:
	sim_upload_graphs(utils.GRAPHWAVE_GRAPHS)
	