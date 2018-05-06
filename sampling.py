import pandas as pd
import numpy as np

np.random.seed(1)

def count_nodes(df):
	"""Takes in dataframe, Returns int"""
	return pd.concat([df[x] for x in df]).nunique()
def filter_nodes(df, s_rate=None, subN=None):
	"""Takes in dataframe, Returns dataframe"""

	assert s_rate is None or subN is None


	if s_rate is not None:
		mask = []

		for x in range(no_nodes):
			mask.append(True if np.random.binomial(1,s_rate) else False)

	if subN is not None:
		assert no_nodes >= subN

		mask = [True]*subN + [False]*(no_nodes - subN) 
		mask = np.random.permutation(mask)


	# might have to modify if we use weights so that all does not include weight

	result = df[df.apply(lambda x: all([mask[x[header]-1] for header in ["node1", "node2"]]),  axis=1)]

	return result


def filter_edges(df, s_rate=None, subE=None):
	"""Takes in dataframe, Returns dataframe"""

	assert s_rate is None or subE is None

	no_edges = len(df)

	if s_rate is not None:
		mask = []
		for x in range(no_edges):
			mask.append(True if np.random.binomial(1,s_rate) else False)

	if subE is not None:

		mask = [True]*subE + [False]*(no_edges - subE) 
		mask = np.random.permutation(mask)


	result = df[mask]

	return result


	

datapath = 'data/karate/karate.edgelist'

df = pd.read_csv(datapath, sep = ' ', names=["node1", "node2"])

no_nodes = count_nodes(df)
no_edges = len(df)



for i in range(10):
	for subN in [0.5*no_nodes, 0.75*no_nodes]:
		res = filter_nodes(df, subN = int(subN))

		res.to_csv(datapath.split('.')[0] + '_subN_' + str(int(subN)) + '_' + str(i) + '.csv', sep= ' ', header=False, index=False)


	for subE in [0.5*no_edges, 0.75*no_edges]:
		res = filter_edges(df, subE = int(subE))

		res.to_csv(datapath.split('.')[0] + '_subE_' + str(int(subE)) + '_' + str(i) + '.csv', sep= ' ', header = False,index = False)







