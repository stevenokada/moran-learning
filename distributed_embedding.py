from __future__ import division
import subprocess
import glob
import simulate_moran as smoran
import distributed_gen_training
import numpy as np
import jsonschema
from pymongo import MongoClient
import datetime
import networkx as nx
import pandas as pd
import csv

from multiprocessing import Process, Lock, Queue, Value, current_process


I_DATAPATH = "data/edgelists/"
N_DATAPATH = "data/node_embeddings/"
G_DATAPATH = "data/graph_embeddings/"

MONGO_URI = "mongodb://skokada:12345@ds115350.mlab.com:15350/moran"

NUMBER_OF_THREADS = 5

builtins = distributed_gen_training.get_builtins()
dimensions = '128'

# non distributed version

def generate_embeddings():

	#GENERATE EMBEDDINGS NOT FOR  DISCONNECTED GRAPHS

	# can use this to change dimensions

	counter = 0
	for graph in builtins:


		try: 
			csv_filename = graph + '.csv'
			node_emb_filename = graph + '.nemb'
			graph_emb_filename = graph + '.gemb'
			g = nx.convert_node_labels_to_integers(smoran.string_to_graph(graph))

			if not nx.is_connected(g):
				raise smoran.GraphDisconnectedError()


			nx.write_edgelist(g, I_DATAPATH + csv_filename, data=False)

			command = ['python', 'node2vec/src/main.py', '--input', I_DATAPATH + csv_filename,
						'--output', N_DATAPATH + node_emb_filename, '--dimensions', dimensions]
			subprocess.call(command)



			# sum node reps to get graph_rep

			node_emb = N_DATAPATH + node_emb_filename

			with open(node_emb) as csvfile:
				# skip over (# nodes, dimension) header
				# next(csvfile)
				node_reader = csv.reader(csvfile, delimiter = ' ')

				for row in node_reader:
					# header
					if len(row) == 2:
						graph_rep = np.array([float(0)]*int(row[1]))
					else:

						to_add = np.array(map(lambda x: float(x), row[1:]))

						graph_rep += to_add


			graph_rep = np.reshape(graph_rep, (1, len(graph_rep)))
			graph_emb = G_DATAPATH + graph + '.gemb'

			np.savetxt(graph_emb, graph_rep, delimiter = ' ')

		except smoran.GraphDisconnectedError:
			print("Graph %s is disconnected" % counter)

		counter += 1
		print("Graph %s is processed" % counter)



def upload_graph_embeddings():

	client = MongoClient(MONGO_URI)

	db = client['moran']
	#change this to data when data script is done
	col = db['embeddings']

	counter = 0

	for file in glob.glob(G_DATAPATH + "*"):
		with open(file) as csvfile:
			graph_reader = csv.reader(csvfile, delimiter = ' ')

			gemb = next(graph_reader)

			gemb = map(lambda x: float(x), gemb)

		#parse file name into graph_name
		#update all graph documents s.t. graph name is same

		graph = file.split("/")[-1]
		graph = graph.split(".gemb")[0]



		col.update_many({"graph_name": graph}, {"$set": {"feature_vec": gemb}},upsert=True)

		counter += 1

		print("Graph %s is uploaded" % counter)




def thread_embed(q_lock, q, v_lock, v):

	print("%s started" % current_process().name)
	while not q.empty():

		with q_lock:
			graph_name = q.get()

		try: 
			csv_filename = graph_name + '.csv'
			node_emb_filename = graph_name + '.nemb'
			graph_emb_filename = graph_name + '.gemb'
			g = nx.convert_node_labels_to_integers(smoran.string_to_graph(graph_name))

			if not nx.is_connected(g):
				raise smoran.GraphDisconnectedError()


			nx.write_edgelist(g, I_DATAPATH + csv_filename, data=False)

			node_emb = N_DATAPATH + node_emb_filename

			command = ['python', 'node2vec/src/main.py', '--input', I_DATAPATH + csv_filename,
						'--output', node_emb, '--dimensions', dimensions]
			subprocess.call(command)



			# sum node reps to get graph_rep


			with open(node_emb) as csvfile:
				# skip over (# nodes, dimension) header
				# next(csvfile)
				node_reader = csv.reader(csvfile, delimiter = ' ')



				for row in node_reader:
					# header
					if len(row) == 2:
						graph_rep = np.array([float(0)]*int(row[1]))
					else:

						to_add = np.array(map(lambda x: float(x), row[1:]))

						graph_rep += to_add


			graph_rep = np.reshape(graph_rep, (1, len(graph_rep)))
			graph_emb = G_DATAPATH + graph_name + '.gemb'

			np.savetxt(graph_emb, graph_rep, delimiter = ' ')

		except smoran.GraphDisconnectedError:
			print("Graph is disconnected")

		with v_lock:
			v.value += 1

		print("Graph %s is processed" % v.value)

	print("%s finished" % current_process().name)




def thread_upload(q_lock, q, v_lock, v):
	client = MongoClient(MONGO_URI)
	db = client['moran']
	col = db['data_test']

	while not q.empty():
		with q_lock:
			file_name = q.get()


		with open(file_name) as csvfile:
			graph_reader = csv.reader(csvfile, delimiter = ' ')

			#skip over the line

			gemb = next(graph_reader)

			#skip over next line

			gemb = map(lambda x: float(x), gemb)

		#parse file name into graph_name
		#update all graph documents s.t. graph name is same

		graph = file_name.split("/")[-1]
		graph = graph.split(".gemb")[0]



		col.update_many({"graph_name": graph}, {"$set": {"feature_vec": gemb}},upsert=True)

		with v_lock:
			v.value += 1

		# print(graph, "is uploaded")
		print("Graph %s is uploaded" % v.value)

	print("%s finished" % current_process().name)







if __name__ == '__main__':
	# generate_embeddings()
	# upload_graph_embeddings()

	q = Queue()
	q_lock = Lock()

	v = Value('i', 0)
	v_lock = Lock()
	# for graph_name in builtins:
	# 	q.put(graph_name)

	# processes = [Process(name="thread %s"%x, target = thread_embed, args = (q_lock, q, v_lock, v)) for x in range(NUMBER_OF_THREADS)]

	# for process in processes:
	# 	process.start()

	# while not q.empty():
	# 	pass

	# print("Done with embedding!")




	v.value = 0

	for file in glob.glob(G_DATAPATH + "*"):
		q.put(file)

	processes = [Process(name="thread %s"%x, target = thread_upload, args = (q_lock, q, v_lock, v)) for x in range(NUMBER_OF_THREADS)]

	for process in processes:
		process.start()

	while not q.empty():
		pass

	print("Done with uploading!")





	# original -> 73m33s
	# 4 threads -> 24m
	# 8 threads -> 36m36s



















#generates graph embeddings for node embeddings



