import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as ge_Data
from scipy.sparse import csr_matrix

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

def handle_data(inputData, train_len=None):
    len_data = [len(nowData) for nowData in inputData]
    if train_len is None:
        max_len = max(len_data)
    else:
        max_len = train_len
    # reverse the sequence
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
               for upois, le in zip(inputData, len_data)]
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
               for le in len_data]
    return us_pois, us_msks, max_len


def handle_adj(adj_dict, n_entity, sample_num, num_dict=None):
    adj_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    num_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    for entity in range(1, n_entity):
        neighbor = list(adj_dict[entity])
        neighbor_weight = list(num_dict[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
        adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
        num_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])

    return adj_entity, num_entity

def pro_inputs(category,inputs):   #为种类设置特征变量，先读出每个序列对应的种类ID序列
    inputs_ID = []
    for item in inputs:
       if item == 0:
          inputs_ID += [0]
       else:
          inputs_ID += [category[item]]
    return inputs_ID 

class Data(Dataset):
    def __init__(self, data, category, opt, model, train=True):

        inputs, mask, max_len = handle_data(data[0])
        self.category = category
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.mask = np.asarray(mask)
        self.length = len(data[0])
        self.max_len = max_len
        self.opt = opt
        self.model = model
        self.train = train

    def __getitem__(self, index):
        u_input, mask, target = self.inputs[index], self.mask[index], self.targets[index]

        input_ID = pro_inputs(self.category, u_input)
        total = np.append(u_input, input_ID)
        total = total[total > 0]

        max_n_node = self.max_len
        node = np.unique(u_input)
        total_node = np.unique(total)
        if len(total_node)<max_n_node*2:
          total_node= np.append(total_node,0)
          
        items = node.tolist() + (max_n_node - len(node)) * [0]
        total_items = total_node.tolist() + (max_n_node * 2 - len(total_node)) * [0]
        total_adj = np.zeros((max_n_node*2, max_n_node * 2))

        for i in np.arange(len(u_input) - 1):
            u = np.where(total_node == u_input[i])[0][0]
            c = np.where(total_node == self.category[u_input[i]])[0][0]
            total_adj[u][u] = 1
            total_adj[c][c] = 4
            total_adj[u][c]= 2
            total_adj[c][u]= 3
            if u_input[i + 1] == 0:
                break          
            u2 = np.where(total_node == u_input[i + 1])[0][0]
            c2 = np.where(total_node == self.category[u_input[i + 1]])[0][0]
            total_adj[u][u2] = 1
            total_adj[u2][u] = 1
            
            total_adj[c][c2] = 4
            total_adj[c2][c] = 4

        if self.train == False and len(u_input[u_input > 0]) > 2:
            total_adj = add_edge(u_input, self.model, total_adj, self.opt)
            # total_adj = jaccard_similarity(u_input, total_adj)

        alias_items = [np.where(total_node == i)[0][0] for i in u_input]
        alias_category = [np.where(total_node == i)[0][0] for i in input_ID]   #对应ID的相对位置
        
        return [torch.tensor(items),torch.tensor(mask), torch.tensor(target),
                torch.tensor(alias_items),torch.tensor(alias_category), 
                torch.tensor(total_adj),torch.tensor(total_items)]

    def __len__(self):
        return self.length


def get_N(items):

    nonzero_indices = torch.nonzero(items, as_tuple=False)
    last_nonzero_indices = torch.zeros(items.size(0), dtype=torch.long)
    for i in range(items.size(0)):
        row_indices = nonzero_indices[nonzero_indices[:, 0] == i, 1]
        if len(row_indices) > 0:
            last_nonzero_indices[i] = row_indices[-1]

    return max(last_nonzero_indices).item() + 1

def get_overlap(sessions):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)
            for j in range(i+1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap))/float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0]*len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0/degree)
        return matrix, degree

def get_N(items):
    N = 0
    items = items.numpy()
    for i in items:
        n = sum(i > 0)
        if n > N:
            N = n
    return N + 1

def add_edge(u_input, model, adj, opt):
    adj = torch.tensor(adj)
    seq = u_input[u_input > 0]
    nodes = np.unique(u_input)
    alias_input = [np.where(nodes == i)[0][0] for i in seq]
    x = torch.tensor(nodes.reshape(-1, 1), dtype=torch.long)
    source = alias_input[:-1]
    target = alias_input[1:]
    edge_index = torch.tensor([source, target], dtype=torch.int32).view(2, -1)
    data = ge_Data(x=x, edge_index=edge_index)

    try:
        model.load_state_dict(torch.load(f'./dict/VGAE-{opt.dataset}.model'))
        with torch.no_grad():
            pre_adj, _, _ = model(data.x, data.edge_index)
    except:
        model.load_state_dict(torch.load(f'./dict/GAE-{opt.dataset}.model'))
        with torch.no_grad():
            pre_adj = model(data.x, data.edge_index)

    # with torch.no_grad():
    #     pre_adj, _, _ = model(data.x, data.edge_index)

    model.eval()
    adj_item = adj[:pre_adj.shape[0], :pre_adj.shape[1]]
    pre_adj[adj_item < 1] = 0
    pre_adj[pre_adj < opt.threshold] = 0
    pre_adj.fill_diagonal_(0)

    # ---------------------------
    pre_adj[pre_adj > opt.threshold] = 1
    adj[:pre_adj.shape[0], :pre_adj.shape[1]] += pre_adj * opt.weight

    # ---------------------------
    # dynamic_weights = local_path_weight(adj_item, pre_adj)
    # print(dynamic_weights)
    # adj[:pre_adj.shape[0], :pre_adj.shape[1]] += dynamic_weights * opt.weight * 10

    # ---------------------------
    # dynamic_weights = degree_normalized_weight(adj_item, pre_adj)
    # adj[:pre_adj.shape[0], :pre_adj.shape[1]] += dynamic_weights

    return adj.numpy()

def jaccard_similarity(u_input, adj):
    nodes = np.unique(u_input)
    adj = torch.tensor(adj)
    adj_item = adj[:len(nodes), :len(nodes)]
    A = csr_matrix(adj_item.numpy())
    """计算稀疏邻接矩阵A的Jaccard系数矩阵"""
    A_bin = (A > 0).astype(int)  # 二值化
    intersection = A_bin @ A_bin.T  # 交集大小
    row_sum = A_bin.sum(axis=1)
    union = row_sum + row_sum.T - intersection  # 并集大小
    J = np.nan_to_num(intersection / union)
    J[J < 0.5] = 0
    J = torch.tensor(J)
    J.fill_diagonal_(0)
    adj[:len(nodes), :len(nodes)] += J * 3
    return adj.numpy()

def degree_normalized_weight(adj, pre_adj, threshold=0.9):
    """
    :param adj: 原始邻接矩阵（float32或float64）
    :param pre_adj: 链路预测的边矩阵（float32或float64）
    :param threshold: 筛选边的阈值
    :return: 动态权重矩阵（与pre_adj类型一致）
    """
    # 统一数据类型为float32
    adj = adj.float()
    pre_adj = pre_adj.float()
    
    degrees = adj.sum(dim=1) + 1e-6  # float32
    edge_mask = (pre_adj >= threshold).float()
    src, dst = (pre_adj >= threshold).nonzero(as_tuple=True)
    weights = 1.0 / (degrees[src] + degrees[dst])  # float32
    weight_matrix = torch.zeros_like(pre_adj, dtype=torch.float32)
    weight_matrix[src, dst] = weights
    return edge_mask * weight_matrix

def local_path_weight(adj, pre_adj, threshold=0.9, k=2):
    """
    考虑k跳路径的权重（k=2时类似共同邻居的扩展）
    """
    adj_matrix = adj.float()
    path_k = torch.matrix_power(adj_matrix, k)  # k跳路径数
    src, dst = (pre_adj >= threshold).nonzero(as_tuple=True)
    weights = path_k[src, dst] / (path_k.sum(dim=1)[src] + 1e-6)  # 归一化
    weight_matrix = torch.zeros_like(pre_adj)
    weight_matrix[src, dst] = weights
    return weight_matrix * (pre_adj >= threshold).float()