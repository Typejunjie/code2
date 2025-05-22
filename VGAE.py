import torch
import torch.nn.functional as F
from GAE_model import VGAE
from sklearn.metrics import roc_auc_score 
import argparse
import pickle
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch_geometric.utils import negative_sampling
from sklearn.metrics import precision_recall_curve, auc, f1_score, roc_auc_score, recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(conf_matrix):
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'], 
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Nowplaying', help='diginetica/Nowplaying/Tmall/yoochoose1_64')
parser.add_argument('--dim', default=100)
parser.add_argument('--batch_size', default=100)
parser.add_argument('--len', default=3)
parser.add_argument('--KL_weight', default=0.1)
opt = parser.parse_args()

# def init_seed(seed=None):
#     if seed is None:
#         seed = int(time.time() * 1000 // 1000)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
# init_seed(2025)
if opt.dataset == 'diginetica':
    num_node = 43098
    n_category = 996    #(995+1)
elif opt.dataset == 'Nowplaying':
    num_node = 60417
    n_category = 11462 #(11461 + 1)
elif opt.dataset == 'Tmall':
    num_node = 40728
    n_category = 712 #711 + 1
elif opt.dataset == 'yoochoose1_64':
    num_node = 37484
    n_category = 61 # 60 + 1
else:
    raise KeyError

def random_graph(data):
    num_nodes = data.num_nodes
    node_indices = torch.randperm(num_nodes)
    data.x = data.x[node_indices]
    data.edge_index = node_indices[data.edge_index]
    return data

def vgae_loss(pred_adj, target_adj, mu, logvar, pos_weight):
    # 加权二元交叉熵
    bce_loss = F.binary_cross_entropy(
        pred_adj, 
        target_adj,
        weight=torch.where(target_adj > 0, pos_weight, 1.0)  # 正样本权重
    )
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return bce_loss + opt.KL_weight * kl_loss  # 可选：降低KL散度权重

def get_pos_weight(edge_index, num_nodes):
    num_pos = edge_index.size(1)          # 正样本数量
    num_neg = num_nodes * num_nodes - num_pos  # 负样本数量（近似）
    pos_weight = num_neg / max(num_pos, 1)    # 权重=负样本数/正样本数
    return torch.tensor([pos_weight])

embedding = torch.nn.Embedding(num_embeddings=num_node, embedding_dim=opt.dim)
# 加载图数据
train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))[0]
test_data = pickle.load(open('./datasets/' + opt.dataset + '/test.txt', 'rb'))[0]

train_list = []
for seq in tqdm(train_data, leave=False):
    if len(seq) < opt.len:
        continue
    u_input = np.array(seq)
    nodes = np.unique(u_input)
    alias_input = [np.where(nodes == i)[0][0] for i in seq]
    x = torch.tensor(nodes.reshape(-1, 1), dtype=torch.long)
    source = alias_input[:-1]
    target = alias_input[1:]
    edge_index = torch.tensor([source, target], dtype=torch.int32).view(2, -1)
    train_list.append(random_graph(Data(x=x, edge_index=edge_index)))
    # train_list.append(Data(x=x, edge_index=edge_index))

test_list = []
for seq in tqdm(test_data, leave=False):
    if len(seq) < opt.len:
        continue
    u_input = np.array(seq)
    nodes = np.unique(u_input)
    alias_input = [np.where(nodes == i)[0][0] for i in seq]
    x = torch.tensor(nodes.reshape(-1, 1), dtype=torch.long)
    source = alias_input[:-1]
    target = alias_input[1:]
    edge_index = torch.tensor([source, target], dtype=torch.int32).view(2, -1)
    test_list.append(random_graph(Data(x=x, edge_index=edge_index)))
    # test_list.append(Data(x=x, edge_index=edge_index))

train_loader = DataLoader(train_list, batch_size=opt.batch_size, shuffle=True)
test_loader = DataLoader(test_list, batch_size=opt.batch_size, shuffle=False)

# 初始化模型和优化器
model = VGAE(embedding, in_dim=opt.dim, hidden_dim=128, out_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

def train(train_loader):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, leave=False):
        optimizer.zero_grad()
        # VGAE返回 (adj_pred, mu, logvar)
        adj_pred, mu, logvar = model(batch.x, batch.edge_index)
        pos_weight = get_pos_weight(batch.edge_index, batch.num_nodes)
        # 构建目标邻接矩阵（仅正样本边为1）
        target_adj = torch.zeros_like(adj_pred)
        target_adj[batch.edge_index[0], batch.edge_index[1]] = 1
        
        # 使用VGAE的损失函数（重构损失 + KL散度）
        loss = vgae_loss(adj_pred, target_adj, mu, logvar, pos_weight)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

def test(loader):
    model.eval()
    preds, labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            # VGAE返回 (adj_pred, mu, logvar)
            adj_pred, mu, _ = model(batch.x, batch.edge_index)  # 测试时忽略logvar
            
            # 正样本预测（真实边）
            pos_pred = adj_pred[batch.edge_index[0], batch.edge_index[1]]
            pos_labels = torch.ones(pos_pred.size(0))
            
            # 负样本采样
            neg_edge_index = negative_sampling(
                batch.edge_index, 
                num_nodes=batch.num_nodes,
                num_neg_samples=batch.edge_index.size(1)  # 保持正负样本平衡
            )
            neg_pred = adj_pred[neg_edge_index[0], neg_edge_index[1]]
            neg_labels = torch.zeros(neg_pred.size(0))
            
            # 收集预测值和标签
            preds.append(torch.cat([pos_pred, neg_pred]))
            labels.append(torch.cat([pos_labels, neg_labels]))

    # 合并所有batch的结果
    preds = torch.cat(preds).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()

    flat_values = preds.flatten()
    mu = np.mean(flat_values)
    sigma = np.std(flat_values)
    cv_global = (sigma / mu) * 100
    
    # 计算指标
    roc_auc = roc_auc_score(labels, preds)
    precision, recall, _ = precision_recall_curve(labels, preds)
    pr_auc = auc(recall, precision)
    f1 = f1_score(labels, (preds > 0.9).astype(int))  # 阈值0.5
    recall_value = recall_score(labels, (preds > 0.9).astype(int))
    conf_matrix = confusion_matrix(labels, (preds > 0.9).astype(int))
    
    return roc_auc, pr_auc, f1, recall_value, conf_matrix, cv_global

# 训练和测试
for epoch in tqdm(range(5)):
    loss = train(train_loader)
    tqdm.write('--------------------------------------')
    tqdm.write(f'Epoch {epoch + 1}, Loss: {loss}')
    roc_auc, pr_auc, f1, recall, m, cv = test(train_loader)
    roc_auc_, pr_auc_, f1_, recall_, m_, cv_ = test(test_loader)
    tqdm.write('Train:')
    tqdm.write(f'roc_auc: {roc_auc}, pr_auc: {pr_auc}, f1: {f1}, recall: {recall}, CV: {cv}')
    tqdm.write('Test:')
    tqdm.write(f'roc_auc: {roc_auc_}, pr_auc: {pr_auc_}, f1: {f1_}, recall: {recall_}, CV: {cv_}')

plot_confusion_matrix(m_)
# pickle.dump(m_, open(f'./data/m_VGAE-{opt.dataset}', 'wb'))
# torch.save(model.state_dict(), f'./dict/VGAE-{opt.dataset}.model')