import argparse
import copy
import datetime
import gc
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import ITrackerData_person_tensor as data_gen
import swpathnet_func_eyetracker
import networkx as nx
import matplotlib.pyplot as plt

# n_layer = 4
#
# # Directed Graph
# G = nx.DiGraph()
# # G.add_nodes_from([i for i in range(n_layer * 2)])
#
# edges = []
# edges.append(('start', 0, 1))
# edges.append(('start', 1, 10))
# for i in range(n_layer * 2 - 2):
#     edges.append((i, (i // 2) * 2 + 2, 0.5))
#     edges.append((i, (i // 2) * 2 + 3, 0.5))
# G.add_weighted_edges_from(edges)
#
# # 1 : trainable
# # 0 : frozen
# atr = {n: 1 if n % 2 != 0 else 0
#        for i, n in enumerate(G.nodes) if isinstance(n, int)}
#
# nx.set_node_attributes(G, atr, 'labels')
#
# print(dict(G.edges))
#
# weights = 0
# next = []
# edges = G.out_edges('start')
# for x in edges:
#     weights += G.edges[x]['weight']
#
# probability = []
# for x in edges:
#     probability.append(G.edges[x]['weight'] / weights)
#
# next = np.random.choice([edge[1] for edge in edges], p=probability)
# print("next is {}".format(next))
#
# paths = []
# print('node label is {}'.format(G.nodes[next]['labels']))
# paths.append(G.nodes[next]['labels'])
#
# print(probability)
# for i in range(n_layer - 1):
#     print(G.out_edges(next))
#     weights = 0
#     edges = G.out_edges(next)
#     for x in edges:
#         weights += G.edges[x]['weight']
#
#     probability = []
#     for x in edges:
#         probability.append(G.edges[x]['weight'] / weights)
#
#     next = np.random.choice([edge[1] for edge in edges], p=probability)
#     # print(a)
#
#     paths.append(G.nodes[next]['labels'])
#
# print(paths)
#
# # set node's position
# pos = {}
# for i, n in enumerate(G.nodes):
#     if isinstance(n, int):
#         pos[n] = (n // 2 + 1, n % 2 + 1)
#     else:
#         pos[n] = (0, 1.5)

# show graph
# nx.draw_networkx(G, pos, with_labels=True)
# plt.show()


class BinaryAntColony:
    def __init__(self, n_layer,
                 evapolate_rate=0.9,
                 initialize_method='uniform',
                 offset=0.1):
        # レイヤ数の設定
        self.n_layer = n_layer

        # 蒸発率の設定
        self.evapolate_rate = evapolate_rate

        if initialize_method == 'uniform':
            initial_parameter = 0.5
        else:
            initial_parameter = 0

        edges = []

        # make graph
        self.G = nx.DiGraph()
        edges.append(('start', 0, initial_parameter))
        edges.append(('start', 1, initial_parameter))
        for k in range(n_layer * 2 - 2):
            edges.append((k, (k // 2) * 2 + 2, initial_parameter))
            edges.append((k, (k // 2) * 2 + 3, initial_parameter))
        self.G.add_weighted_edges_from(edges)

        # 1 : trainable
        # 0 : frozen
        atr = {n: 1 if n % 2 != 0 else 0
               for i, n in enumerate(self.G.nodes) if isinstance(n, int)}

        nx.set_node_attributes(self.G, atr, 'labels')

    # フェロモンでpathを生成
    def gen_path(self):
        weights_sum = 0
        next = []
        edges = self.G.out_edges('start')
        for x in edges:
            weights_sum += self.G.edges[x]['weight']

        probability = []
        for x in edges:
            probability.append(self.G.edges[x]['weight'] / weights_sum)

        next = np.random.choice([edge[1] for edge in edges], p=probability)
        print("next is {}".format(next))

        path = []
        print('node label is {}'.format(self.G.nodes[next]['labels']))
        path.append(self.G.nodes[next]['labels'])

        print(probability)
        for i in range(self.n_layer - 1):
            print(self.G.out_edges(next))
            weights_sum = 0
            edges = self.G.out_edges(next)
            for x in edges:
                weights_sum += self.G.edges[x]['weight']

            probability = []
            for x in edges:
                probability.append(self.G.edges[x]['weight'] / weights_sum)

            next = np.random.choice([edge[1] for edge in edges], p=probability)
            # print(a)

            path.append(self.G.nodes[next]['labels'])

        print(path)

        return path

    # フェロモンの更新
    def update_pheromone(self, path, acc):

        # evaporation
        weights = nx.get_edge_attributes(self.G, 'weight')
        for k,v in weights.items():
            weights[k] = v * self.evapolate_rate

        nx.set_edge_attributes(self.G, values=weights)

        pheromone = 1/acc
        pre = -1
        for node in path:
            if pre >= 0:
                self.G.edges[(pre, node)]['weight'] += 1 / acc
                pre = node
            else:
                self.G.edges[('start', node)]['weight'] += 1 / acc
                pre = node
