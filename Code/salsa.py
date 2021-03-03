import networkx as nx
import pickle

def ppr(key):
    personalization = ref_personalization.copy()
    personalization[key] = 1.0
    return nx.pagerank(G, personalization=personalization)

with open('company_investor_undirected.pkl', 'rb') as file:
    G = pickle.load(file)


CIRCLE_SIZE = 7
ref_personalization = {}

for node in G.nodes:
    if G.nodes[node]['type'] == 'investor':
        ref_personalization[node] = 0.0

betweenness_centrality = nx.betweenness_centrality(G)
betweenness_centrality = sorted(betweenness_centrality.items(), key=lambda x: x[1])
print(betweenness_centrality)
print(G.nodes[betweenness_centrality[-1][0]]['type'])
# Tencent Holdings has the highest betweenness_centrality and degree_centrality and closeness_centrality and eigenvector_centrality

# ppr for all investors
# for key in ref_personalization:
#     personalization = ref_personalization.copy()
#     personalization[key] = 1.0
#     ppr = nx.pagerank(G, personalization=personalization)
#     print(ppr)

query = 'Sequoia Capital'
th_ppr = ppr(query)
th_ppr = sorted(th_ppr.items(), key=lambda x: x[1], reverse=True)
hubs = []
for ppr in th_ppr:
    if G.nodes[ppr[0]]['type'] == 'investor' or G.nodes[ppr[0]]['type'] == 'company_investor':
        hubs.append(ppr[0])
    if len(hubs) == CIRCLE_SIZE:
        break
auths = set([])
for h in hubs:
    for child in G.neighbors(h):
        if G.nodes[child]['type'] == 'company':
            auths.add(child)

sub_graph = nx.subgraph(G, hubs+list(auths))
print(sub_graph.number_of_nodes())
print(sub_graph.number_of_edges())
hub_scores, auth_scores = nx.hits(sub_graph, max_iter=20000)
auth_scores = sorted(auth_scores.items(), key=lambda x: x[1], reverse=True)
for a in auth_scores:
    if G.nodes[a[0]]['type'] == 'company' and a[0] not in G.neighbors(query):
        shared_hubs = 0
        for child in G.neighbors(a[0]):
            if child in hubs:
                shared_hubs += 1
        print(a[0], str(shared_hubs/len(hubs)*100) + '%')