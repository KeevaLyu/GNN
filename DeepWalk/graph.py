import logging
import sys
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
import random
from random import shuffle
from itertools import product,permutations
from scipy.io import loadmat
from scipy.sparse import issparse

class Graph(defaultdict):
    def __init__(self):
        super(Graph, self).__init__(list)

    def nodes(self):
        return self.keys()

    def adjacency_iter(self):
        return self.iteritmes()

    def subgraph(self, nodes={}):
        subgraph = Graph()
        for n in nodes:
            if n in self:
                subgraph[n] = [x for x in self[n] if x in nodes]
        return subgraph

    def make_undirected(self):
        for v in list(self):
            for other in self[v]:
                if v != other:
                    self[other].append(v)
        self.make_consistent()

    def make_consistent(self):
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))
        self.remove_self_loops()
        return self

    def remove_self_loops(self):
        for x in self:
            if x in self[x]:
                self[x].remove(x)
        return self

    def check_self_loops(self):
        for x in self:
            for y in self[x]:
                if x==y:
                    return True
        return False

    def has_edge(self, v1, v2):
        if v2 in self[v1] or v1 in self[v2]:
            return True
        return False

    def degree(self, nodes=None):
        if isinstance(nodes, Iterable):
            return {v:len(self[v]) for v in nodes}
        else:
            return len(self[nodes])

    def order(self):
        return len(self)

    def number_of_edges(self):
        return sum([self.degree(x) for x in self.keys()])/2

    def number_of_nodes(self):
        return self.order()

    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        G = self
        if start:
            path = [start]
        else:
            path = [rand.choice(list(G.keys()))]

        while len(path) < path_length:
            i = path[-1]
            if len(G[i]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(G[i]))
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]

def build_deepwalk_corpus(G, num_paths, path_length, alpha=0, rand=random.Random(0)):
    walks = []
    nodes = list(G.nodes())
    for i in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
    return walks

def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0, rand=random.Random(0)):
    nodes = list(G.nodes())
    for i in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)

# def clique(size):
#     return from_adjlist(permutations(range(1,size+1)))

def grouper(n, iterable, padvalue=None):
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def parse_adjacencylist_unchecked(f):
    adjlist = []
    for l in f:
        if l and l[0] != "#":
            adjlist.extend([[int(x) for x in l.strip().split()]])
    return adjlist

# def parse_adjacencylist(f):
#     adjlist = []
#     for l in f:
#         if l and l[0] != "#":
#             introw = [int(x) for x in l.strip().split()]
#             row = [introw[0]]
#             row.extend(set(sorted(introw[1:])))
#             adjlist.extend([row])
#     return adjlist

def load_adjacencylist(file, undirected=False, chunksize=10000, unchecked=True):
    if unchecked:
        parse_func = parse_adjacencylist_unchecked
        convert_func = from_adjlist_unchecked
    # else:
    #     parse_func = parse_adjacencylist
    #     convert_func = from_adjlist

    adjlist = []
    with open(file) as f:
        for i, adj_chunk in enumerate(map(parse_func, grouper(int(chunksize), f))):
            adjlist.extend(adj_chunk)
    G = convert_func(adjlist)

    if undirected:
        G = G.make_undirected()

    return G

def load_edgelist(file, undirected=True):
    G = Graph()
    with open(file) as f:
        for l in f:
            x, y = l.strip().split()[:2]
            x = int(x)
            y = int(y)
            G[x].append(y)
            if undirected:
                G[y].append(x)
    G.make_consistent()
    return G

def load_matfile(file, name="network", undirected=True):
    mat = loadmat(file)
    mat_matrix = mat[name]
    return from_numpy(mat_matrix, undirected)

# def from_networkx(G_input, undirected=True):
#     G = Graph()
#     for idx, x in enumerate(G_input.nodes()):
#         for y in iterkeys(G_input[x]):
#             G[x].append(y)
#
#     if undirected:
#         G.make_undirected()
#
#     return G

def from_numpy(x, undirected=True):
    G = Graph()
    if issparse(x):
        cx = x.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
        raise Exception("Dense matrices not yet supported")

    if undirected:
        G.make_undirected()

    G.make_consistent()
    return G

# def from_adjlist(adjlist):
#     G = Graph()
#     for row in adjlist:
#         node = row[0]
#         neighbors = row[1:]
#         G[node] = list(sorted(set(neighbors)))
#     return G

def from_adjlist_unchecked(adjlist):
    G = Graph()
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = list(sorted(set(neighbors)))
    return G