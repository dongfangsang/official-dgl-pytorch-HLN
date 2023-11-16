import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        # 两个全连接层
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)   # 每个节点在metapath维度的均值; mean(0):每个meta-path上的均值(/|V|);
        # N是节点数量，M是元路径数
        beta = torch.softmax(w, dim=0)  # (M, 1) # 归一化
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)  # 扩展到N个节点上的metapath的值
        # sum（1）将在第一个维度上求和，即在 metapath 维度上求和
        return (beta * z).sum(1)  # (N, D * K)


class GCNLayer(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN_lastAttention
        self.layers.append(
            GraphConv(in_size,hidden_size, activation=F.relu)
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


class MultiLayerGCN(nn.Module):
    def __init__(self, num_meta_paths, in_size, out_size, dropout):
        super(MultiLayerGCN, self).__init__()

        hidden_size = 1000

        self.layers = nn.ModuleList()
        for _ in range(num_meta_paths):
            self.layers.append(GCNLayer(in_size, hidden_size, out_size, dropout))

        self.semantic_attention = SemanticAttention(
            in_size=out_size
        )

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            semantic_embeddings.append(self.layers[i](g, h).flatten(1))

        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)


class GCNSemantic(nn.Module):
    def __init__(
            self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout
    ):
        super(GCNSemantic, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            MultiLayerGCN(
                #  hidden_size作为GCN的输出维度，最后的预测全连接层输出维度才是 分类数量，ACM数据集中这里hidden_size默认为8，分类数out_size 3
                num_meta_paths, in_size, hidden_size, dropout
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
