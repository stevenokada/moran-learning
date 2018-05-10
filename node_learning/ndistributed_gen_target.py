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
import ndistributed_simulate_moran as ndsmoran


def thread_simulate(q_lock, q, c_lock, counter):
	print("%s starting to generate data" % mp.current_process().name)
	collection = utils.NODE_DATA_FINAL


	try:
		
		while not q.empty():

			with q_lock:

				graph_name = q.get()


			try: 


				# p_success, f_time, classification, number_of_nodes, number_of_edges = smoran.aggregate_run(graph_name, FITNESS, 'builtin', NUMBER_OF_RUNS)

				# bson_obj = {"number_of_nodes": number_of_nodes, "number_of_edges": number_of_edges,"graph_name": graph_name, "datatype": "data", "feature_vec": [], "f_val": FITNESS, 
				# "target_vec":{"classification" : classification, "p_success": p_success, "f_time":f_time}, "number_of_runs": NUMBER_OF_RUNS, "date": datetime.datetime.now()}



SCHEMA = {"type": "object", "properties": {'graph_name': {"type": "string"}, 'rep_dim': {'type': 'int'},'node_vectors': {"type": "object"}, 'num_edges':{'type':'int'}
 ,'num_nodes': {'type': 'int'}, 'num_runs': {'type':'int'}, 'f_val': {'type': 'number'}}, 'required': ['graph_name', 'rep_dim','node_vectors', 'num_edges', 'num_nodes', 'num_runs','f_val']}

				results = ndsmoran.aggregate_run(graph_name, utils.FITNESS, utils.NUMBER_OF_RUNS)
				bson_obj = collection.update_one({'graph_name': graph_name}, {'$set': {'num_runs': utils.NUMBER_OF_RUNS, 'f_val': utils.FITNESS}})



				# results['node_1': {'p_success':, 'f_time', 'classification': }]





				try: 
					jsonschema.validate(bson_obj, schema)
					collection.replace_one({"graph_name": graph_name, "f_val": FITNESS}, bson_obj, upsert=True)

				except jsonschema.exceptions.ValidationError:
					print("Malformed input")


				print("graph_name: %s, p_success: %s, f_time: %s, classification %s" % (graph_name, p_success, f_time,classification))


			
			except smoran.GraphDisconnectedError:
				print("Graph is disconnected")

			with c_lock:
				counter.value += 1

			print("Graph %s processed" % counter.value)



			# print("%s processed %s" % (current_process().name, graph_name))

	finally:
		client.close()







if __name__ == '__main__':
	q = mp.Queue()
	c = mp.Value('i', 0)
	q_lock = mp.Lock()
	c_lock = mp.Lock()
	
	for graph in utils.BUILTINS:
		q.put(x)


	processes = [mp.Process(name='thread %s'%x,target=thread_simulate, args=(q_lock,q, c_lock, c)) for x in range(utils.NUMBER_OF_THREADS)]

	for process in processes:
		process.start()

	# processes are running and q is not empty
	while not q.empty():
		pass

	print("Done!")



		








