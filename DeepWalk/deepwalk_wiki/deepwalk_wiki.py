
import numpy as np
import random
import networkx as nx
from gensim.models import Word2Vec


# Start random walk from start_node
def deepwalk_walk(walk_length, start_node):
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = list(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            walk.append(random.choice(cur_nbrs))
        else:
            break
    return walk

# Create Embedding
def _simulate_walks(nodes, num_walks, walk_length):
    walks = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for v in nodes:
            walks.append(deepwalk_walk(walk_length=walk_length, start_node=v))
    return walks

if __name__ == "__main__":
    G = nx.read_edgelist('Wiki_edgelist.txt',
                         create_using=nx.DiGraph(),
                         nodetype=None,
                         data=[('weight', int)])
    #Get nodes
    nodes = list(G.nodes())
    #Random walks
    walks = _simulate_walks(nodes, num_walks=80, walk_length=10)
    #Skipgram
    w2v_model = Word2Vec(walks, sg=1, hs=1)
    print(w2v_model.wv["1"])

