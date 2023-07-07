import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

#change nhid to adjust layers;

#class GCN(nn.Module):
#    def __init__(self, nfeat, nhid, nclass, dropout):
#        super(GCN, self).__init__()
#
#        self.gc1 = GraphConvolution(nfeat, nhid)
#        self.gc2 = GraphConvolution(nhid, nclass)
#        self.dropout = dropout
#
#    def forward(self, x, adj):
#        x = F.relu(self.gc1(x, adj))
#        x = F.dropout(x, self.dropout, training=self.training)
#        x = self.gc2(x, adj)
#        return F.log_softmax(x, dim=1)


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, num_layers):
        super(GCN, self).__init__()

        self.gc_layers = nn.ModuleList()
        self.gc_layers.append(GraphConvolution(nfeat, nhid))

        for _ in range(num_layers - 2):
            self.gc_layers.append(GraphConvolution(nhid, nhid))

        self.gc_layers.append(GraphConvolution(nhid, nclass))
        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, x, adj):
        for i in range(self.num_layers - 1):
            x = F.relu(self.gc_layers[i](x, adj))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc_layers[self.num_layers - 1](x, adj)
        return F.log_softmax(x, dim=1)
