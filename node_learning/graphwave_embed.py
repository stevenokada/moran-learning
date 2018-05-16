import os
import sys
import numpy as np
import pandas as pd
import seaborn as sb
import networkx as nx 
import utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

sys.path.append('../graphwave/')
from GraphWave.graphwave import graphwave
from GraphWave.shapes import build_graph

# from GraphWave import *
# from GraphWave import shapes, graphwave, heat_diffusion, characteristic_functions
# from GraphWave.shapes import build_graph,shapes
# from GraphWave.heat_diffusion import *
# from GraphWave.utils.graph_tools  import *
# from GraphWave.utils.utils import *
# from GraphWave.characteristic_functions import *

np.random.seed(123)


for graph_name in utils.get_connected_builtins(utils.get_builtins_medium()):
	graph = utils.string_to_graph(graph_name)

	try:
		chi, heat_print, taus=graphwave(graph, 'automatic', verbose=False)

		for node_index, feature_vec in enumerate(chi):

			feature_vec = list(feature_vec)
			utils.GRAPHWAVE_DATA_FINAL.update_one({"graph_name": graph_name},{"$set":{"node_vectors.%s.feature_vec"%node_index: feature_vec, 
				"rep_dim": len(feature_vec)}}, upsert=True)




	except:
		print("Embedding failed for %s" % graph_name)

