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
						'--output', node_emb_path, '--dimensions', str(utils.REP_DIMENSIONS)]
			subprocess.call(command)



		except utils.GraphDisconnectedError:
			print("Graph is disconnected")

		with v_lock:
			v.value += 1

		print("Graph %s is processed" % v.value)

	print("%s finished" % mp.current_process().name)




def thread_node_sim_upload(graph_name, q_lock, q, v_lock, v, return_dict):
	
	print("%s started" % mp.current_process().name)

	while not q.empty():
		with q_lock:
			node_index = q.get()
			#file_name is path to node_emb

		graph_name = file_name.split("/")[-1]
		graph_name = graph_name.split(".nemb")[0]


		
		# result = {'p_success':, 'f_time', 'classification': }

		result = ndsmoran.MoranNodeSimulation(graph_name, utils.FITNESS, utils.NUMBER_OF_RUNS).single_node_run(node_index, early_stop=True)

		with v_lock:
			return_dict[str(node_index)] = result


		with v_lock:
			v.value += 1

		print("Node %s is uploaded" % v.value)

	print("%s finished" % mp.current_process().name)











if __name__ == '__main__':

	q = mp.Queue()
	q_lock = mp.Lock()

	v = mp.Value('i', 0)
	v_lock = mp.Lock()

	for graph_name in utils.GRAPHWAVE_GRAPHS:
		q.put(graph_name)

	processes = [mp.Process(name="thread %s"%x, target = thread_embed, args = (q_lock, q, v_lock, v)) for x in range(utils.NUMBER_OF_THREADS)]

	for process in processes:
		process.start()

	while not q.empty():
		pass

	for process in processes:
		process.join()

	print("Done with embedding!")

	time.sleep(10)







	# big graphs, distributed over nodes

	for file_name in glob.glob(utils.N_DATAPATH + "*"):


		graph_name = file_name.split("/")[-1]
		graph_name = graph_name.split(".nemb")[0]

		if graph_name in utils.GRAPHWAVE_GRAPHS:
			v.value = 0


			graph = utils.string_to_graph(graph_name)
			for i in range(graph.number_of_nodes()):
				q.put(i)

			adj_mat = nx.adjacency_matrix(graph)
			pickled_adj_mat = pickle.dumps(adj_mat)

			manager = mp.Manager()
			return_dict = manager.dict()


			processes = [mp.Process(name="thread %s"%x, target = thread_node_sim_upload, args = (graph_name, q_lock, q, v_lock, v, return_dict)) for x in range(utils.NUMBER_OF_THREADS)]

			for process in processes:
				process.start()

			while not q.empty():
				pass

			for process in processes:
				process.join()


			list_of_nodes = []
			for node, target_vec in return_dict.items():
				node_row = {"node": node, **target_vec}
				list_of_nodes.append(node_row)

			df = pd.DataFrame(list_of_nodes)
			# check this line below
			df.to_csv('data/node_targets/%s.nvec' % graph_name, sep=' ',index=False)

			with open(file_name) as csvfile:
				node_reader = csv.reader(csvfile, delimiter = ' ')

				num_nodes, rep_dim = next(node_reader)

				num_nodes = int(num_nodes)
				num_edges = len(graph.edges())
				rep_dim = int(rep_dim)

				node_vectors = {}
				for node_vec in node_reader:
					node_vectors[node_vec[0]] = {'feature_vec': list(map(lambda x: float(x), node_vec[1:])), 'target_vec': return_dict[node_vec[0]]}
			

			utils.NODE_DATA_FINAL.update_one({"graph_name": graph_name}, {"$set": {"node_vectors": node_vectors, "rep_dim": rep_dim, "num_nodes": num_nodes, 'num_edges': num_edges,
				'f_val': utils.FITNESS, 'num_runs': utils.NUMBER_OF_RUNS, "adj_matrix": pickled_adj_mat}},upsert=True)


			print("Done with uploading %s!" % graph_name)


			for process in processes:
				process.join()