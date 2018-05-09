from multiprocessing import Process, Lock, Queue, Value, current_process
from pymongo import MongoClient
import simulate_moran as smoran
import datetime
import jsonschema
import csv
import numpy as np
import pandas as pd


FITNESS = 5
NUMBER_OF_RUNS = 200
NUMBER_OF_THREADS = 5
MONGO_URI = "mongodb://skokada:12345@ds115350.mlab.com:15350/moran"

schema = {"type": "object", "properties": {"number_of_nodes": {"type":"integer"}, "number_of_edges": {"type": "integer"},"graph_name": {"type": "string"}, 
"datatype": {"type": "string"}, "feature_vec": {"type": "array"}, "f_val": {"type": "number", "minimum": 1}, "target_vec":{"type": "object"}, 
"number_of_runs": {"type": "integer", "minimum": 0} }, "required": ["graph_name", "datatype", 
 "feature_vec", "f_val", "target_vec", "number_of_runs", "number_of_nodes", "number_of_edges"]}


#non distributed
# if __name__ == '__main__':

# 	try: 
# 		print("Starting to generate data")
# 		client = MongoClient(MONGO_URI)
# 		db = client['moran']
# 		collection = db['data']




# 		counter = 1

# 		# for file in glob.glob(I_DATAPATH + "*"):
# 		# 	# command = ["python", "node2vec/src/main.py", "--input", file, "--output", file.split(".")[0] + ".rep"]
# 		# 	# subprocess.call(command)

# 		# 	# training_data_@fitness.csv

# 		# 	# make row with file_name, feature vec, p_success @ given fitness, fixation_success @ given fitness


# 		# 	for fitness in FITNESSES:

# 		# 		p_success, f_time, number_of_nodes, number_of_edges = smoran.aggregate_run(file, fitness, 'data', NUMBER_OF_RUNS)

# 		# 		bson_obj = {"number_of_nodes": number_of_nodes, "number_of_edges": number_of_edges,"graph_name": file, "datatype": "real", "feature_vec": {}, "f_val": fitness, 
# 		# 		"target_vec":{"p_success": p_success, "f_time":f_time}, "number_of_runs": NUMBER_OF_RUNS, "date": datetime.datetime.now()}


# 		# 		try: 
# 		# 			jsonschema.validate(bson_obj, schema)
# 		# 			collection.replace_one({"graph_name": file, "f_val": fitness}, bson_obj, upsert=True)

# 		# 		except jsonschema.exceptions.ValidationError:
# 		# 			print("Malformed input")


# 		# 		print("p_success: %s, f_time: %s" % (p_success, f_time))


# 		# 	print("Graph %s processed" % counter)

# 		# 	counter += 1


# 		for graph in builtins:
# 			# command = ["python", "node2vec/src/main.py", "--input", file, "--output", file.split(".")[0] + ".rep"]
# 			# subprocess.call(command)

# 			# training_data_@fitness.csv

# 			# make row with file_name, feature vec, p_success @ given fitness, fixation_success @ given fitness


# 			try: 
# 				for fitness in FITNESSES:

# 					p_success, f_time, classification, number_of_nodes, number_of_edges = smoran.aggregate_run(graph, fitness, 'builtin', NUMBER_OF_RUNS)

# 					bson_obj = {"number_of_nodes": number_of_nodes, "number_of_edges": number_of_edges,"graph_name": graph, "datatype": "data", "feature_vec": [], "f_val": fitness, 
# 					"target_vec":{"classification" : classification, "p_success": p_success, "f_time":f_time}, "number_of_runs": NUMBER_OF_RUNS, "date": datetime.datetime.now()}


# 					try: 
# 						jsonschema.validate(bson_obj, schema)
# 						collection.replace_one({"graph_name": graph, "f_val": fitness}, bson_obj, upsert=True)

# 					except jsonschema.exceptions.ValidationError:
# 						print("Malformed input")


# 					print("p_success: %s, f_time: %s, classification %s" % (p_success, f_time,classification))


# 				# THIS PRINT DOESN'T HANDLE DISCONNECTED GRAPHS
# 				print("Graph %s processed" % counter)

# 				counter += 1

# 			except smoran.GraphDisconnectedError:
# 				print("Graph %s is disconnected" % counter)




# 	finally:
# 		client.close()


def get_builtins(): 
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

		args = map(lambda x: str(x), args)
		result = []



		for x in range(number_of_runs):

			ans = constructor_name + '-' + '_'.join(args) + '-' + 'seed=' + str(x)

			result.append(ans)

		return result


	n = 100

	p_values = []
	p_values.append(1/(2*n)) #np < 1
	p_values.append(1/n) #np = 1
	p_values.append(2/n) #np > 1
	p_values.append(np.log(n)/(2*n)) #p < ln(n)/n
	p_values.append(np.log(n)/n)#p == ln(n)/n
	p_values.append(2*np.log(n)/n) #p > ln(n)/n

	random = []
	for p_val in p_values:
		random= random + gen_rg_names('erdos_renyi_graph', 50, 100, p_val)

	for p_val in p_values:
		for k in [0, 5, 20, 50, 75]:
			random = random + gen_rg_names('newman_watts_strogatz_graph', 10, 100, k, p_val)

	for m in [1, 2, 4, 8, 10]:
		random = random + gen_rg_names('barabasi_albert_graph', 60, 100, m)



	builtins = builtins + random

	return builtins




def simulate(q_lock, q, c_lock, counter):
	print("%s starting to generate data" % current_process().name)
	client = MongoClient(MONGO_URI)
	db = client['moran']
	collection = db['F' + str(FITNESS) + '_noemb']

	try:
		
		while not q.empty():

			with q_lock:

				graph_name = q.get()


			try: 


				p_success, f_time, classification, number_of_nodes, number_of_edges = smoran.aggregate_run(graph_name, FITNESS, 'builtin', NUMBER_OF_RUNS)

				bson_obj = {"number_of_nodes": number_of_nodes, "number_of_edges": number_of_edges,"graph_name": graph_name, "datatype": "data", "feature_vec": [], "f_val": FITNESS, 
				"target_vec":{"classification" : classification, "p_success": p_success, "f_time":f_time}, "number_of_runs": NUMBER_OF_RUNS, "date": datetime.datetime.now()}


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


def mongo_to_csv_target():
	client = MongoClient(MONGO_URI)
	db = client['moran']
	collection = db['DATAF' + str(FITNESS) + '_FINAL']

	builtins = get_builtins()


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




if __name__ == '__main__':
	pass
	q = Queue()
	c = Value('i', 0)
	q_lock = Lock()
	c_lock = Lock()
	
	builtins = smoran.get_connected_builtins()

	for x in builtins:
		q.put(x)


	# works with single fitness value

	# q will store builtin function names

	# each thread will establish mongo connection (to be closed when terminates)

	# will call smoran.aggregate_run on inputs


	

	processes = [Process(name='thread %s'%x,target=simulate, args=(q_lock,q, c_lock, c)) for x in range(NUMBER_OF_THREADS)]

	for process in processes:
		process.start()

	# processes are running and q is not empty
	while not q.empty():
		pass

	print("Done!")



		








