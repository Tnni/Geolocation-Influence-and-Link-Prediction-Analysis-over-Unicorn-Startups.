import pickle
import networkx
from node2vec import Node2Vec

with open('company_investor_undirected.pkl', 'rb') as file:
    G = pickle.load(file)

embedding = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
