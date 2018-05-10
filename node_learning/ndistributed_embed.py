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


def thread_embed(q_lock, q, v_lock, v):

	print("%s started" % mp.current_process().name)
	while not q.empty():

		with q_lock:
			graph_name = q.get()

		try: 
			csv_filename = graph_name + '.csv'
			node_emb_filename = graph_name + '.nemb'
			g = utils.string_to_graph(graph_name)

			if not nx.is_connected(g):
				raise utils.GraphDisconnectedError()

			edgelist_path = utils.I_DATAPATH + csv_filename
			nx.write_edgelist(g, edgelist_path, data=False)

			node_emb_path = utils.N_DATAPATH + node_emb_filename

			command = ['python2.7', '../node2vec/src/main.py', '--input', edgelist_path,
						'--output', node_emb_path, '--dimensions', '128']
			subprocess.call(command)



		except utils.GraphDisconnectedError:
			print("Graph is disconnected")

		with v_lock:
			v.value += 1

		print("Graph %s is processed" % v.value)

	print("%s finished" % mp.current_process().name)




def thread_sim_upload(q_lock, q, v_lock, v):
	

	while not q.empty():
		with q_lock:
			file_name = q.get()
			#file_name is path to node_emb

		graph_name = file_name.split("/")[-1]
		graph_name = graph_name.split(".nemb")[0]


		adj_mat = nx.adjacency_matrix(utils.string_to_graph(graph_name))
		pickled_adj_mat = pickle.dumps(adj_mat)

		# results['0': {'p_success':, 'f_time', 'classification': }]

		results = ndsmoran.MoranNodeSimulation(graph_name, utils.FITNESS, utils.NUMBER_OF_RUNS).node_run()


		list_of_nodes = []
		for node, target_vec in results:
			node_row = {"node": node, **target_vec}
			list_of_nodes.append(node_row)

		df = pd.DataFrame(list_of_nodes)
		# check this line below
		df.to_csv('data/node_targets/%s.nvec' % graph_name, sep=' ',index=False)


		# write nodetargetvecs to csv CHECK
		# pickle and store adjacency CHECK
		# make sure we keep node order consistent



		with open(file_name) as csvfile:
			node_reader = csv.reader(csvfile, delimiter = ' ')

			num_nodes, rep_dim = next(node_reader)

			num_nodes = int(num_nodes)
			num_edges = len(utils.string_to_graph(graph_name).edges())
			rep_dim = int(rep_dim)

			node_vectors = {}
			for node_vec in node_reader:
				node_vectors[node_vec[0]] = {'feature_vec': list(map(lambda x: float(x), node_vec[1:])), 'target_vec': results[node_vec[0]]}


		

		utils.NODE_DATA_FINAL.update_one({"graph_name": graph}, {"$set": {"node_vectors": node_vectors, "rep_dim": rep_dim, "num_nodes": num_nodes, 'num_edges': num_edges,
			'f_val': utils.FITNESS, 'num_runs': utils.NUMBER_OF_RUNS, "adj_matrix": pickled_adj_mat}},upsert=True)

		with v_lock:
			v.value += 1

		# print(graph, "is uploaded")
		print("Graph %s is uploaded" % v.value)

	print("%s finished" % mp.current_process().name)


# node_data = [{'graph_name': graph_name, 'node_features': {'node1': [...],  ...}, 'node_targets': {'node1': {'p_success': ,'f_time': }, ...}, 'num_edges': ,'num_nodes': , 'num_runs': }]






if __name__ == '__main__':

	q = mp.Queue()
	q_lock = mp.Lock()

	v = mp.Value('i', 0)
	v_lock = mp.Lock()

	for graph_name in utils.BUILTINS:
		q.put(graph_name)

	processes = [mp.Process(name="thread %s"%x, target = thread_embed, args = (q_lock, q, v_lock, v)) for x in range(utils.NUMBER_OF_THREADS)]

	for process in processes:
		process.start()

	while not q.empty():
		pass

	print("Done with embedding!")

	time.sleep(10)




	v.value = 0

	for file in glob.glob(utils.N_DATAPATH + "*"):
		q.put(file)

	processes = [mp.Process(name="thread %s"%x, target = thread_sim_upload, args = (q_lock, q, v_lock, v)) for x in range(utils.NUMBER_OF_THREADS)]

	for process in processes:
		process.start()

	while not q.empty():
		pass

	print("Done with uploading!")