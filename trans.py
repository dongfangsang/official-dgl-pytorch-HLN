import utils
import dgl
import matplotlib.pyplot as plt
import networkx as nx
if __name__ == '__main__':
    import pandas as pd
    import pickle

    # 读取.pkl文件

    args = []
    (
        g,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
    ) = utils.load_data("ACM")

    # 转换为NetworkX图，因为DGL目前不直接支持GraphML或GEXF导出
    nx_graph = g[0].to_networkx()

    # 保存为GraphML格式
    # nx.write_graphml(nx_graph, 'your_graph.graphml')

    # 保存为GEXF格式
  #  nx.write_gexf(nx_graph, 'your_graph.gexf')





    print("done")
