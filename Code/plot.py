import openpyxl
import networkx as nx
import matplotlib.pyplot as plt
import pickle

with open('company_investor_undirected.pkl', 'rb') as file:
    G = pickle.load(file)

with open('cluster_coloring.pkl', 'rb') as file:
    cluster_coloring = pickle.load(file)

colors = []
for node in G.nodes:
    if G.nodes[node]['type'] == 'investor':
        colors.append('red')
    elif G.nodes[node]['type'] == 'company':
        colors.append('green')
    elif G.nodes[node]['type'] == 'company_investor':
        colors.append('blue')

nx.draw(G, with_labels=False, edge_color='black', alpha=0.3, node_size=3, width = 0.2, cmap=plt.cm.Reds, node_color=colors)
plt.show()