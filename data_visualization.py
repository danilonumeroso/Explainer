from utils import io_utils
import matplotlib.pyplot as plt
import networkx as nx

graphs = io_utils.read_graphfile('data', 'Tox21_AHR')

G = graphs[2]
nx.draw(G)
plt.show()
