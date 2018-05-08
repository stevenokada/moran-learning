from __future__ import division


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input
from keras.models import Model
from keras.utils import to_categorical




import subprocess
import glob
import simulate_moran as smoran
import numpy as np
import jsonschema
from pymongo import MongoClient
import datetime
import networkx as nx
import pandas as pd
import csv







I_DATAPATH = "data/edgelists/"
N_DATAPATH = "data/node_embeddings/"
G_DATAPATH = "data/graph_embeddings/"
MONGO_URI = "mongodb://skokada:12345@ds115350.mlab.com:15350/moran"


# load training data
def load_data(fitness_val = 5):
	# returns feature matrix, output matrix tuple

	client = MongoClient(MONGO_URI)
	col = client['moran']['DATAF5_FINAL']

	feature_mat = []
	target_mat = []

	# CAREFUL THAT FIND DOES NOT RETURN SPECIFIC ORDER SO TRAINING DATA + MATRIX DATA IS RANDOM
	# WE ARE LOSING GRAPH_NAME DATA

	for match in col.find({"f_val": fitness_val}):

		feature_vec = match["feature_vec"]

		# target_vec is dictionary at this point
		target_vec = match["target_vec"]

		# classification = target_vec['classification']
		# p_success = target_vec['p_success']
		# f_time = target_vec['f_time']

		# target_vec = [classification, p_success, f_time]


		feature_mat.append(feature_vec)
		target_mat.append(target_vec)

	return feature_mat, target_mat




def select_target_val(Y, arg='classification'):
	# args is 3-vector that specifies which parts of output we want learning on


	# Y = [['A', 0.9, 2300], [...],...]

	Y_new = []

	for target_vec in Y:
		# map each vec to new_vec
		# new_vec = [x for i, x in enumerate(target_vec) if args[i] == 1]

		new_vec = target_vec[arg]




		Y_new.append(new_vec)

	return Y_new


def predict_on_graph_name(model, graph_name):
	client = MongoClient(MONGO_URI)
	col = client['moran']['DATAF5_FINAL']

	graph = col.find_one({'graph_name':graph_name})

	graph = np.array(graph)

	res = model.predict(graph)

	res = 'A' if res==[0,1] else 'S'

	return res



# [['A'], ['B']

if __name__ == '__main__':

	# might want to normalize feature vectors

	dimensions = 128
	task = 'classification'


	X, Y = load_data(fitness_val=5)


	Y = select_target_val(Y, arg=task)
	# Y = ['A','S','S',..]

	Y = map(lambda x: 0 if x == 'S' else 1, Y)
	Y = to_categorical(Y)


	# X = map(lambda x: [x for x in range(128)], X)


	X_scaled = preprocessing.scale(X)

	X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.33, random_state=42)


	# 380 A, 260 S, 60% A, 40% S
	# A -> 1 -> [0,1]



	inputs = Input(shape=(dimensions, ))

	x = Dense(10, activation='relu')(inputs)
	x = Dense(10, activation='relu')(x)
	predictions = Dense(2, activation='softmax')(x)


	model = Model(inputs=inputs, outputs=predictions)
	model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
	model.fit(X_train, Y_train, epochs = 300)
	loss_and_metrics = model.evaluate(X_test, Y_test)
	print(loss_and_metrics)


	#map(lambda x: 'A' if x[0] > x[1] else 'S', model.predict(X_train)).count("A")

	res_train = zip(model.predict(X_train), Y_train)
	res_test = zip(model.predict(X_test),Y_test)





