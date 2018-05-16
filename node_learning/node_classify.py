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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

#DON'T TOUCH
import imp
coarsening = imp.load_source('coarsening', '../cnn_graph/lib/coarsening.py')
cnn_utils = imp.load_source('cnn_utils', '../cnn_graph/lib/utils.py')
graph = imp.load_source('graph','../cnn_graph/lib/graph.py')
models = imp.load_source('models','../cnn_graph/lib/models.py')





# def load_data, put things into [{'feature_vec': [--node1--, --node2--, ...], 'target_vec':[--nodetarget1--, --nodetarget], 'adjacency': binary}, ...]



def load_single_graph(graph_name, metric):
	# returns numpy array
	col = utils.NODE_DATA_FINAL

	graph_data = col.find_one({'graph_name': graph_name})

	node_vectors = graph_data['node_vectors']

	X = []
	Y = []
	A = pickle.loads(graph_data['adj_matrix'])
	for node_vec in sorted(node_vectors.items()):
		node_data = node_vec[1]
		X.append(node_data['feature_vec'])

		if node_data['target_vec'][metric] in utils.LABELS:
			Y.append(utils.LABELS[node_data['target_vec'][metric]])

		else:
			Y.append(node_data['target_vec'][metric])


	X = np.array(X)
	Y = np.array(Y)

	return X,Y,A




def load_data(metric):

	col = utils.NODE_DATA_FINAL

	X = []
	Y = []
	A = []

	for graph in col.find().sort('graph_name', pymongo.ASCENDING):

		graph_data = {'feature_vecs': [], 'target_vecs': [], 'adj_matrix': None}
		graph_data['adj_matrix'] = pickle.loads(graph['adj_matrix'])

		node_vectors = graph['node_vectors']

		for node_vec in sorted(node_vectors.items()):

			node_data = node_vec[1]
			node_features = node_data['feature_vec']

			node_targets_dict = node_data['target_vec']

			
			# assume we want all the targets in below order (if we have muliple)

			if metric == 'target_order':
				node_targets = [node_targets_dict[x] for x in utils.TARGET_ORDER]

			elif metric == 'p_success' or metric == 'classification':
				node_targets = [node_targets_dict[metric]]

			# maps classification according to utils.LABELS

			node_targets = list(map(lambda x: utils.LABELS[x] if x in utils.LABELS else x, node_targets))



			# if we only have one target, don't make it a vector (might change later)
			if len(node_targets) == 1:
				node_targets = node_targets[0]



			graph_data['feature_vecs'].append(node_features)
			graph_data['target_vecs'].append(node_targets)


		X.append(graph_data['feature_vecs'])
		Y.append(graph_data['target_vecs'])
		A.append(graph_data['adj_matrix'])


	return X,Y,A


def pca_visualize_graph(node_features, node_targets):
	pca = PCA(n_components=3)
	node_features_pca = pca.fit_transform(node_features)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')


	num_nodes = len(node_features)
	pr = utils.calculate_pr(utils.FITNESS, num_nodes)

	for node_index, node in enumerate(node_features_pca):
		# assuming just p_success is target

		p_success = node_targets[node_index]

		if p_success < pr:
			c= 'r'
			# S

		else:
			c= 'b'
			# A

		ax.scatter(*node, c=c)

	plt.show()

def pca_visualize_graphs(list_node_features, list_node_targets):

	flattened_features = [node for graph in list_node_features for node in graph]
	flattened_features = preprocessing.scale(flattened_features)
	flattened_targets = [node for graph in list_node_targets for node in graph]

	pca_visualize_graph(flattened_features, flattened_targets)





np.random.seed(0)


X, Y, A = load_data('p_success')


# for graph_index in range(len(X)):
# 	X_nodes = np.array(X[graph_index])
# 	Y_targets = np.array(Y[graph_index])

# 	X_scaled = preprocessing.scale(X_nodes)
# 	X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_targets, test_size=0.2, random_state=42)

# 	rfr = RandomForestRegressor()
# 	rfr.fit(X_train, Y_train)
# 	print(rfr.score(X_test, Y_test))

# 	break



for graph_name in utils.get_connected_builtins(utils.get_builtins_medium()):
	X, Y, _ = load_single_graph(graph_name, 'classification')
	X_scaled = preprocessing.scale(X)
	X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.33, random_state=42)

	rfr = KNeighborsClassifier(n_neighbors=2)
	rfr.fit(X_train, Y_train)

	predictions = rfr.predict(X_test)
	ground_truth = Y_test
	print("F1 Score: %s" % f1_score(ground_truth, predictions))


	print('Graph_name: %s, Number of nodes: %s, Accuracy: %s, Percentage A: %s ' %(graph_name, len(Y), rfr.score(X_test, Y_test), np.count_nonzero(Y_test)/len(Y_test)))
