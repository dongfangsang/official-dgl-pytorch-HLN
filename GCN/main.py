import torch
from sklearn.metrics import f1_score
from utils import EarlyStopping, load_data


def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)  #
    micro_f1 = f1_score(labels, prediction, average="micro")
    macro_f1 = f1_score(labels, prediction, average="macro")

    return accuracy, micro_f1, macro_f1


def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1


def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
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
    ) = load_data(args["dataset"])

    # 转bool类型
    if hasattr(torch, "BoolTensor"):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    # 放GPU
    features = features.to(args["device"])
    labels = labels.to(args["device"])
    train_mask = train_mask.to(args["device"])
    val_mask = val_mask.to(args["device"])
    test_mask = test_mask.to(args["device"])

    if args["hetero"]:  # 构建异质网络的邻居节点
        print("DGL dataset work,waiting to be done")
    else:
        from model import GCN

        model = GCN(
            num_meta_paths=len(g),   # g就是元图
            in_size=features.shape[1],
            hidden_size=args["hidden_units"],
            out_size=num_classes,
            num_heads=args["num_heads"],
            dropout=args["dropout"],
        ).to(args["device"])
        # 元路径子图转GPU并拼接到一个变量list
        g = [graph.to(args["device"]) for graph in g]
        g = g[1] # 取第一个元路径子图

    stopper = EarlyStopping(patience=args["patience"])

    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    )
    # 保存用以画图
    train_losses = []
    train_accuracies = []
    train_micro_f1s = []
    train_macro_f1s = []

    val_losses = []
    val_accuracies = []
    val_micro_f1s = []
    val_macro_f1s = []


    for epoch in range(args["num_epochs"]):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(
            logits[train_mask], labels[train_mask]
        )

        train_losses.append(loss.item())
        train_accuracies.append(train_acc)
        train_micro_f1s.append(train_micro_f1)
        train_macro_f1s.append(train_macro_f1)

        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(
            model, g, features, labels, val_mask, loss_fcn
        )

        val_losses.append(val_loss.item())
        val_accuracies.append(val_acc)
        val_micro_f1s.append(val_micro_f1)
        val_macro_f1s.append(val_macro_f1)

        early_stop = stopper.step(val_loss.data.item(), val_acc, model)


        print(
            "Epoch {:d} | Train Loss {:.4f} | Train Acc {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | "
            "Val Loss {:.4f} | Val Acc {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}".format(
                epoch + 1,
                loss.item(),
                train_acc,
                train_micro_f1,
                train_macro_f1,
                val_loss.item(),
                val_acc,
                val_micro_f1,
                val_macro_f1,
            )
        )


        if early_stop:
            break
    # 结束训练 作图
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    df = pd.DataFrame({
        'Epoch': np.arange(len(train_losses)),
        'Train Loss': train_losses,
        'Train Accuracy': train_accuracies,
        'Train Micro F1': train_micro_f1s,
        'Train Macro F1': train_macro_f1s
    })
    # 保存指标数据
    df.to_csv(os.path.join(args['log_dir'], 'training_metrics.csv'), index=False)
    # 转长格式，利于可视化
    df_melted = df.melt(id_vars=['Epoch'], var_name='Metric', value_name='Value')

    sns.set(style="whitegrid")  # 设置样式，可根据需要修改

    pic = sns.relplot(
        data=df_melted,
        x='Epoch',
        y='Value',
        hue="Metric",
        kind='line',
        height=5,
        aspect=1.5,
        palette='husl',
        markers=True,
        dashes=False,
        col='Metric',  # 以列的形式显示不同的指标
        col_wrap = 2
    )

    # 设置图表标题
    pic.fig.suptitle('Training Metrics Over Epochs', y=1.02)

    # 显示图形
    plt.show()

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(
        model, g, features, labels, test_mask, loss_fcn
    )
    print(
        "Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(
            test_loss.item(), test_micro_f1, test_macro_f1
        )
    )
    torch.save(model.state_dict(), os.path.join(args['log_dir'], 'saved_parameters.pth'))

if __name__ == "__main__":
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser("HAN")
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "-ld",
        "--log-dir",
        type=str,
        # 指定日志目录
        default="results",
        help="Dir for saving training results",
    )
    parser.add_argument(
        "--hetero",
        action="store_true",
        help="Use metapath coalescing with DGL's own dataset",
    )
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)
