from __future__ import division
import subprocess
import glob
import simulate_moran as smoran
import numpy as np
import jsonschema
from pymongo import MongoClient
import datetime
import networkx as nx




I_DATAPATH = "data/"
FITNESSES = [2,5,10]
NUMBER_OF_RUNS = 500
MONGO_URI = "mongodb://skokada:12345@ds115350.mlab.com:15350/moran"

#networkx functions
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
		return getattr(nx, constructor_name)()

	if len(args) == 2:
		# builtin with *args
		params = args[1].split('_')
		params = map(convert_arg, params)

		return getattr(nx, constructor_name)(*params)

	if len(args) == 3:
		params = args[1].split('_')
		params = map(convert_arg, params)

		seed = args[2].split('=')[1]
		seed = int(seed)
		return getattr(nx, constructor_name)(*params, seed=seed)



