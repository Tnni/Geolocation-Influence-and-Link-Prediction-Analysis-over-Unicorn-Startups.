import random
import openpyxl
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from datetime import datetime
from collections import OrderedDict
import pandas as pd
import numpy as np
import datetime

# fof_ind_date = pd.read_pickle("./fof_ind_date.pkl")
fof_ind_date = pd.read_pickle("./fof_ind_date.pkl")

# list of dataframe columns
cols = list(fof_ind_date.columns.values.tolist())

# ==== Get the top 5 most invested industries.
# this is so we can have only them as attributes, instead of 130 attributes for each node.

top_5_ind_dict = {}
for i in cols[2:]:
    top_5_ind_dict[i] = len(fof_ind_date[fof_ind_date[i].notnull()])
    # print(len(fof_ind_date[fof_ind_date[i].notnull()]))
top_5_ind_dict = dict(
    sorted(top_5_ind_dict.items(), key=lambda item: item[1], reverse=True))
# this is the list of top 5 industries
top_5_ind = list(top_5_ind_dict.keys())[:5]
print(top_5_ind[:5])

print(fof_ind_date)


# ====== MAKING THE NETWORK
G = nx.DiGraph()

# == adding nodes
# G.add_node('Tencent Holdings', {'Mobile': 2018-10-16 15:16:21 , 'Software': 2018-10-16 15:16:21, ...})
for i, row in fof_ind_date.iterrows():
    G.add_node(row[0])
    for ind in top_5_ind:
        # attrs[ind] = row[ind]
        G.nodes[row[0]][ind] = row[ind]

# == adding edges
for i, row in fof_ind_date.iterrows():
    for frnd in row.friends:
        G.add_edge(row[0], frnd)

nx.draw(G, with_labels=False, node_size=2, width=0.1,
        node_color='g', edge_color='black')
print(len(G.edges))
# nodesAt5 = [x for x, y in G.nodes(data=True) isinstance(y, datetime.date)]
# print(nodesAt5)

# T = nx.dfs_tree(G, source='Tencent Holdings')
# print(list(T.edges()))

# Source = G.nodes['Tencent Holdings']['Software']


def active(node):
    # return (np.isnat(np.datetime64(str(G.nodes[n2]['Software']))))
    # 'node' is like 'G.nodes[n2]['Software']'
    return not (np.isnat(np.datetime64(str(node))))


def shuf(dict):
    values = list(dict.values())
    random.shuffle(values)
    for key, val in zip(dict, values):
        dict[key] = val
    return dict


# === actual results
res_og_sh = {}
res_sh_test = {}
# === actual results


allog = []
allsh = []
og_sh = {}
sh_test = {}
for src in G.nodes():
    result = []
    act_cnt = 0
    inf_cnt = 0
    act_nodes = {}

    # ========= OG CHECK
    for n1, n2 in list(nx.bfs_edges(G, source=src)):
        if (active(G.nodes[n2]['Software'])):
            act_cnt = act_cnt + 1
            act_nodes[n2] = G.nodes[n2]['Software']

            if G.nodes[src]['Software'] < G.nodes[n2]['Software']:
                inf_cnt = inf_cnt + 1
                result.append(n2)

    og_sh[src] = str(inf_cnt)+"/"+str(act_cnt)

    allog.append([act_cnt, inf_cnt, src])

    print(">>> From ", src)
    print(np.unique(result))
    print(len(np.unique(result)))
    print(str(inf_cnt)+"/"+str(act_cnt))
    print("act_nodes: ", act_nodes)
    if (act_cnt == 0):
        res_og_sh[src] = 0
    else:
        res_og_sh[src] = inf_cnt / act_cnt
    # adding source to make the shuffle 'more interesting'.
    # if(active(G.nodes[src]['Software'])):
    act_nodes[src] = G.nodes[src]['Software']

    # # ========= SHUFFLE CHECK
    print("========= SHUFFLE CHECK")
    result = []
    act_cnt = 0
    inf_cnt = 0
    if len(act_nodes) > 0:
        print("shuf(act_nodes): ", shuf(act_nodes))

    for n1, n2 in list(nx.bfs_edges(G, source=src)):
        if (active(G.nodes[n2]['Software'])):
            act_cnt = act_cnt + 1

            if act_nodes[src] < act_nodes[n2]:
                inf_cnt = inf_cnt + 1
                result.append(n2)
    print(">>> From ", src)
    print(np.unique(result))
    print(len(np.unique(result)))
    print(str(inf_cnt)+"/"+str(act_cnt))
    print("act_nodes: ", act_nodes)
    sh_test[src] = str(inf_cnt)+"/"+str(act_cnt)
    if (act_cnt == 0):
        res_sh_test[src] = 0
    else:
        res_sh_test[src] = inf_cnt / act_cnt
    allsh.append([act_cnt, inf_cnt, src])
    # break
print(">>>> og_sh")
print(og_sh)
print(">>>> sh_test")
print(sh_test)


# chart
x_og_label = []
y_og_label = []
label_og = []
x_sh_label = []
y_sh_label = []
label_sh = []
sorted_allog = sorted(allog, key=lambda x: x[0])
sorted_allsh = sorted(allsh, key=lambda x: x[0])
plt.title('Original vs. Random Activation for Software', fontsize=20)
plt.xlabel('# Nodes Activated')
plt.ylabel('# Nodes Activated due to Social Influence')
for i in range(len(sorted_allog)):
    if sorted_allog[i][0] in x_og_label:
        if sorted_allog[i][1] > y_og_label[x_og_label.index(sorted_allog[i][0])]:
            y_og_label[x_og_label.index(
                sorted_allog[i][0])] = sorted_allog[i][1]
            label_og[x_og_label.index(sorted_allog[i][0])] = sorted_allog[i][2]
    else:
        x_og_label.append(sorted_allog[i][0])
        y_og_label.append(sorted_allog[i][1])
        label_og.append(sorted_allog[i][2])

for i in range(len(sorted_allsh)):
    if sorted_allsh[i][0] in x_sh_label:
        if sorted_allsh[i][1] > y_sh_label[x_sh_label.index(sorted_allsh[i][0])]:
            y_sh_label[x_sh_label.index(
                sorted_allsh[i][0])] = sorted_allsh[i][1]
            label_sh[x_sh_label.index(sorted_allsh[i][0])] = sorted_allsh[i][2]
    else:
        x_sh_label.append(sorted_allsh[i][0])
        y_sh_label.append(sorted_allsh[i][1])
        label_sh.append(sorted_allog[i][2])

plt.plot(x_og_label, y_og_label, color="y",
         marker='o', markersize=2, label='Original')
plt.plot(x_sh_label, y_sh_label, color="r",
         marker='o', markersize=2, label='Shuffled')
for i, txt in enumerate(label_og):
    plt.annotate(txt, (x_og_label[i], y_og_label[i]))
for i, txt in enumerate(label_sh):
    plt.annotate(txt, (x_sh_label[i], y_sh_label[i]))
plt.legend(loc=2)
plt.show()


#     if G.nodes['Tencent Holdings']['Software'] < G.nodes[n2]['Software']:
#         result.append(n2)
# print(">>> From ", 'Tencent Holdings')
# print(np.unique(result))
# print(len(np.unique(result)))
# print(list(nx.bfs_successors(G, 'Tencent Holdings')))


# for src in G.nodes():
#     result = []
#     for n1, n2 in list(nx.bfs_edges(G, source=src)):
#         if G.nodes[src]['Software'] < G.nodes[n2]['Software']:
#             result.append(n2)
#     print(">>> From ", src)
#     print(np.unique(result))
#     print(len(np.unique(result)))
#     # print(list(nx.bfs_successors(G, 'Tencent Holdings')))


# T = nx.DiGraph()
# T.add_node(0)
# T.add_node(1)
# T.add_node(2)
# T.add_node(3)
# T.add_node(4)
# T.add_node(5)
# T.add_node(6)
# T.add_node(7)
# T.add_node(8)
# T.add_node(9)
# T.add_node(10)
# T.add_node(11)

# T.add_edge(0, 1)
# T.add_edge(0, 2)
# T.add_edge(0, 3)
# T.add_edge(0, 4)
# T.add_edge(1, 5)
# T.add_edge(1, 6)
# T.add_edge(2, 7)
# T.add_edge(3, 8)
# T.add_edge(3, 9)
# T.add_edge(8, 9)
# T.add_edge(8, 10)
# T.add_edge(9, 10)
# T.add_edge(11, 10)
# T.add_edge(7, 11)

# print(list(nx.bfs_tree(T, source=0).edges(data=True)))
# print(list(nx.bfs_edges(T, source=0)))


# ORIGINAL - SHUFFLE > 0
inf_res = {}
for i in res_og_sh:
    if (res_og_sh[i] - res_sh_test[i]) > 0:
        inf_res[i] = res_og_sh[i] - res_sh_test[i]
for i in inf_res:
    print(i)
    print(inf_res[i])
