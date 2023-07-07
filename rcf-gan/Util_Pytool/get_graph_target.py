from utility import get_graph_target
import pickle as pkl
import scipy.sparse as sp
import torch
import torchvision

target_source = 'cora'
adj, self_features = get_graph_target(target_source)

print(adj)

