import random
import networkx as nx
import numpy as np


# for each edge type, construct a RWGraph, generate random works for training sequences
class RWGraph():
    def __init__(self, nx_G, node_type=None):
        self.G = nx_G
        self.node_type = node_type

    def walk(self, walk_length, start, schema=None):
        # Simulate a random walk starting from start node.
        G = self.G

        rand = random.Random()

        if schema:
            schema_items = schema.split('-')
            assert schema_items[0] == schema_items[-1]

        walk = [start]
        while len(walk) < walk_length:
            cur = walk[-1]
            candidates = []
            for node in G[cur].keys():
                if schema == None or self.node_type[node] == schema_items[len(walk) % (len(schema_items) - 1)]:
                    candidates.append(node)
            if candidates:
                walk.append(rand.choice(candidates))  # randomly pick up one from candidates
            else:
                break
        return [str(node) for node in walk]

    def simulate_walks(self, num_walks, walk_length, schema=None):
        G = self.G
        walks = []  # each row is a group of random walks for all nodes, with start node shuffled
        nodes = list(G.nodes())
        # print('Walk iteration:')
        if schema is not None:
            schema_list = schema.split(',')
        for walk_iter in range(num_walks):
            print('the ' +str(walk_iter)+'th walk')
            random.shuffle(nodes)
            for node in nodes:
                if schema is None:
                    print("is None")
                    exit()
                    walks.append(self.walk(walk_length=walk_length, start=node))
                else:
                    for schema_iter in schema_list:
                        if schema_iter.split('-')[0] == self.node_type[node]:
                            walks.append(self.walk(walk_length=walk_length, start=node, schema=schema_iter))
                            #print(walks)

        return walks
