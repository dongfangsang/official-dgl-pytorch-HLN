import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv


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


class HLNLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : list[DGLGraph]
        List of graphs
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(
            self, num_meta_paths, in_size, out_size, layer_num_heads, dropout
    ):
        super(HLNLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()

        # 一个meta-path 一次GAT, ACM中的两个GAT 的维度一样
        for i in range(num_meta_paths):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                )
            )

        # 多元路径的注意力，它的input size 跟 上面GAT输出的output size 保持一致
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        # 单元路径的嵌入
        semantic_embeddings = []

        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HLN(nn.Module):
    def __init__(
            self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout
    ):
        super(HLN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            HLNLayer(
                num_meta_paths, in_size, hidden_size, num_heads[0], dropout
            )
        )
        # num是一维
        for l in range(1, len(num_heads)):
            self.layers.append(
                HLNLayer(
                    num_meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
            )

        # 上一个简单的全连接层
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    # 只有一层
    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)   # 这里g是多个元路径子图，Graph类型，即多个邻接矩阵
            # 输出的是∶节点，meta-path数量，embedding; Returns:节点HAN后输出的embedding

        return self.predict(h)  # HAN输出节点embedding后接—个Linear层

