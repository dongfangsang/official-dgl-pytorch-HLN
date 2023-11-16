import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

class GCNLayer(nn.Module):
    def __init__(self, in_size, out_size, dropout):
        super().__init__()

        hidden_size = 16

        self.layers = nn.ModuleList()
        # two-layer GCN_lastAttention
        self.layers.append(
            GraphConv(in_size, hidden_size, activation=F.relu)
        )
        self.layers.append(GraphConv(hidden_size, out_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h




class GCN(nn.Module):
    def __init__(
            self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout
    ):
        super(GCN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            GCNLayer(
                #  hidden_size作为GCN的输出维度，最后的预测全连接层输出维度才是 分类数量，ACM数据集中这里hidden_size默认为8，分类数out_size 3
                in_size, hidden_size, dropout
            )
        )

        # 上一个简单的全连接层
        self.predict = nn.Linear(hidden_size, out_size)

    # 只有一层
    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)  # 这里g是多个元路径子图，Graph类型，即多个邻接矩阵
            # 输出的是∶节点，meta-path数量，embedding; Returns:节点HAN后输出的embedding

        return self.predict(h)  # HAN输出节点embedding后接—个Linear层
