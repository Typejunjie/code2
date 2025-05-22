import torch
import torch.nn.functional as F
from GAE_model import GAE
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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

def plot_confusion_matrix(conf_matrix):
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'], 
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='diginetica/Nowplaying/Tmall/yoochoose1_64')
parser.add_argument('--dim', default=100)
parser.add_argument('--batch_size', default=100)
parser.add_argument('--len', default=3)
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
model = GAE(embedding, in_dim=opt.dim, hidden_dim=128, out_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
def compute_f1_score(pos_pred, neg_pred, threshold=0.5):
    # 合并正负样本的预测值和标签
    y_pred = torch.cat([pos_pred, neg_pred]).cpu().numpy()
    y_true = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)]).cpu().numpy()
    
    # 根据阈值进行分类
    y_pred_class = (y_pred > threshold).astype(int)
    
    # 计算 F1 分数
    f1 = f1_score(y_true, y_pred_class)
    return f1

def train():
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, leave=False):
        optimizer.zero_grad()
        adj_pred = model(batch.x, batch.edge_index)
        target_adj = torch.zeros_like(adj_pred)
        target_adj[batch.edge_index[0], batch.edge_index[1]] = 1

        pred_values = adj_pred[batch.edge_index[0], batch.edge_index[1]]
        neg_edge_index = negative_sampling(batch.edge_index, num_nodes=batch.num_nodes)
        neg_pred_values = adj_pred[neg_edge_index[0], neg_edge_index[1]]

        pos_weight = neg_edge_index.size(1) / batch.edge_index.size(1)
        neg_weight = 1
        pos_weight_tensor = torch.full_like(pred_values, fill_value=pos_weight)
        neg_weight_tensor = torch.full_like(neg_pred_values, fill_value=neg_weight)
        pos_loss = F.binary_cross_entropy(pred_values, torch.ones(batch.edge_index.size(1)), weight=pos_weight_tensor)
        neg_loss = F.binary_cross_entropy(neg_pred_values, torch.zeros(neg_edge_index.size(1)), weight=neg_weight_tensor)
        # print(batch.edge_index.size(1), neg_edge_index.size(1))
        # raise KeyboardInterrupt
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return loss

def test(loader):
    model.eval()
    preds, labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            # 编码器和解码器
            adj_pred = model(batch.x, batch.edge_index)
            pos_pred = adj_pred[batch.edge_index[0], batch.edge_index[1]]
            pos_labels = torch.ones(pos_pred.size(0))
            neg_edge_index = negative_sampling(batch.edge_index, num_nodes=batch.num_nodes)
            neg_pred = adj_pred[neg_edge_index[0], neg_edge_index[1]]
            neg_labels = torch.zeros(neg_pred.size(0))
            
            pos_weight = neg_edge_index.size(1) / batch.edge_index.size(1)
            neg_weight = 1
            pos_weight_tensor = torch.full_like(pos_pred, fill_value=pos_weight)
            neg_weight_tensor = torch.full_like(neg_pred, fill_value=neg_weight)
            pos_loss = F.binary_cross_entropy(pos_pred, torch.ones(batch.edge_index.size(1)), weight=pos_weight_tensor)
            neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros(neg_edge_index.size(1)), weight=neg_weight_tensor)
            loss = pos_loss + neg_loss

            preds.append(torch.cat([pos_pred, neg_pred]))
            labels.append(torch.cat([pos_labels, neg_labels]))

    cosine_scheduler.step()
    plateau_scheduler.step(loss)
    # 合并所有 batch 的预测值和标签
    preds = torch.cat(preds).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    
    # 计算 ROC AUC
    roc_auc = roc_auc_score(labels, preds)
    
    # 计算 PR 曲线和 PR-AUC
    precision, recall, _ = precision_recall_curve(labels, preds)
    pr_auc = auc(recall, precision)
    
    # 计算 F1 分数
    f1 = f1_score(labels, (preds > 0.9).astype(int))  # 默认阈值为 0.5
    
    # 计算召回率（默认阈值 0.5）
    recall_value = recall_score(labels, (preds > 0.9).astype(int))
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(labels, (preds > 0.9).astype(int))
    
    # 输出混淆矩阵
    return roc_auc, pr_auc, f1, recall_value, conf_matrix

# 训练和测试
for epoch in tqdm(range(5)):
    loss = train()
    tqdm.write('--------------------------------------')
    tqdm.write(f'Epoch {epoch + 1}, Loss: {loss}')
    roc_auc, pr_auc, f1, recall, m = test(train_loader)
    roc_auc_, pr_auc_, f1_, recall_, m_ = test(test_loader)
    tqdm.write('Train:')
    tqdm.write(f'roc_auc: {roc_auc}, pr_auc: {pr_auc}, f1: {f1}, recall: {recall}')
    tqdm.write('Test:')
    tqdm.write(f'roc_auc: {roc_auc_}, pr_auc: {pr_auc_}, f1: {f1_}, recall: {recall_}')

plot_confusion_matrix(m_)
pickle.dump(m_, open(f'./data/m_GAE-{opt.dataset}', 'wb'))
# torch.save(model.state_dict(), f'./dict/GAE-{opt.dataset}.model')