# from lib import models, graph, coarsening, utils
import numpy as np
import matplotlib.pyplot as plt

n= 1000
d=100
c=5
X = np.random.normal(0, 1, (n, d)).astype(np.float32)
X += np.linspace(0, 1, c).repeat(d // c)