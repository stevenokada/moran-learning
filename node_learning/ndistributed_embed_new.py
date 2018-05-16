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


def thread_embed_upload(r,s,q_lock, q, v_lock, v):

	print("%s started" % mp.current_process().name)
	while not q.empty():

		with q_lock:
			graph_name = q.get()

		try: 
			csv_filename = graph_name + 'r=%s_s=%s'%(r,s) +  '.csv'
			node_emb_filename = graph_name + 'r=%s_s=%s'%(r,s)'.nemb'
			g = utils.string_to_graph(graph_name)

			if not nx.is_connected(g):
				raise utils.GraphDisconnectedError()

			edgelist_path = utils.I_DATAPATH + csv_filename
			nx.write_edgelist(g, edgelist_path, data=False)

			node_emb_path = utils.N_DATAPATH + node_emb_filename

			command = ['python2.7', '../node2vec/src/main.py', '--input', edgelist_path,
						'--output', node_emb_path, '--dimensions', str(utils.REP_DIMENSIONS), '--window-size', str(20), '--q', str(r), '--p', str(s)]
			subprocess.call(command)




			#handle uploading part now

			rs_map = {(1,8): '0125', (1,4): '025', (1,2): '05', (1,1): '1', (2,1): '2', (4,1): '4', (8,1): '8'}
			collection = pymongo.MongoClient(utils.MONGO_URI)['moran']['NODE_qp_'+rs_map[(r,s)]]

			bson_object = {}
			bson_object['num_nodes'] = g.number_of_nodes()
			bson_object['num_edges'] = g.number_of_edges()
			bson_object['adj_mat'] = pickle.dumps(nx.adjacency_matrix(g))
			bson_object['num_runs'] = utils.NUMBER_OF_RUNS
			bson_object['f_val'] = utils.FITNESS



			with open(node_emb_path) as csvfile:
				node_reader = csv.reader(csvfile, delimiter = ' ')

				_, rep_dim = next(node_reader)

				
				bson_object['rep_dim'] = int(rep_dim)

				node_vectors = {}
				for node_vec in node_reader:


					node_vectors[node_vec[0]] = {'feature_vec': list(map(lambda x: float(x), node_vec[1:])), 'target_vec': {}}

			bson_object['node_vectors'] = node_vectors
			

			collection.insert_one(bson_object)
			print("Done with uploading %s!" % graph_name)





		except utils.GraphDisconnectedError:
			print("Graph is disconnected")

		with v_lock:
			v.value += 1

		print("Graph %s is processed" % v.value)

	print("%s finished" % mp.current_process().name)





# q -> r, p -> s

def embed_upload(r, s, list_of_graphs):

	q = mp.Queue()
	q_lock = mp.Lock()
	v = mp.Value('i', 0)
	v_lock = mp.Lock()

	for graph_name in list_of_graphs:
		q.put(graph_name)


	processes = [mp.Process(name='thread %s'%x, target=thread_embed_upload, args=(r,s,q_lock, q, v_lock, v)) for x in range(utils.NUMBER_OF_THREADS)]

	for process in processes:
		process.start()

	while not q.empty():
		pass

	for process in processes:
		process.join()



if '__main__': == __name__:

	rs_map = {(1,8): '0125', (1,4): '025', (1,2): '05', (1,1): '1', (2,1): '2', (4,1): '4', (8,1): '8'}

	for r,s in rs_map:

		embed_upload(r, s, utils.GRAPHWAVE_GRAPHS)
		collection = pymongo.MongoClient(utils.MONGO_URI)['moran']['NODE_qp_'+rs_map[(r,s)]]

		utils.update_with_targets(collection)







