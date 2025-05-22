import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import *
from torch.nn import Module, Parameter
import torch.nn.functional as F
from utils import *
import pickle


class CombineGraph(Module):
    def __init__(self, opt, num_node, n_category, category):
        super(CombineGraph, self).__init__()
        self.opt = opt
        # self.data = []

        self.batch_size = opt.batch_size
        self.num_node = num_node
        # self.num_total = num_total
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample
        self.print = False

        self.n_category = n_category
        self.category = category
        # Aggregator
        self.LineConv = LineConv(layers=opt.layers, emb_size=self.dim)
        self.local_agg_mix_1 = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)

        # Item representation & Position representation
        self.embedding = nn.Embedding(num_node + n_category - 1, self.dim)
        # self.cat_embedding = nn.Embedding(n_category, self.dim)
        self.pos = nn.Embedding(200, self.dim)

        # Parameters_1
        self.w_1 = nn.Parameter(torch.Tensor(3 * self.dim, 2 * self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(2 * self.dim, 1))
        self.glu1 = nn.Linear(2 * self.dim, 2 * self.dim)
        self.glu2 = nn.Linear(2 * self.dim, 2 * self.dim, bias=False)
        
        self.bbb = Parameter(torch.Tensor(1))
        self.ccc = Parameter(torch.Tensor(1))

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()
        
        item = []
        for x in range(1,num_node):
            item += [category[x]]
        item = np.asarray(item)
        self.item =  trans_to_cuda(torch.Tensor(item).long())

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden1_mix, hidden2_mix, mask):
        hidden1 = hidden1_mix
        hidden2 = hidden2_mix

        hidden = torch.cat([hidden1, hidden2],-1)
        
        mask = mask.float().unsqueeze(-1)
        batch_size = hidden1.shape[0]
        len = hidden1.shape[1]
        
        pos_emb = self.pos.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        b = self.embedding.weight[1:self.num_node]
        item_category = self.embedding(self.item)
        t = torch.cat([b,item_category],-1)

        # scores = torch.matmul(select, t.transpose(1, 0))
        return select, t

    def forward(self, total_items, total_adj):
        
        hidden_mix = self.embedding(total_items)
        
        hidden_mix = self.local_agg_mix_1(hidden_mix, total_adj)

        hidden_mix = F.dropout(hidden_mix, self.dropout_local, training=self.training)

        return hidden_mix


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        # return variable
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data):
    (items, mask, targets,
    alias_items, alias_category, total_adj, total_items)= data

    N = get_N(total_items)

    alias_items = trans_to_cuda(alias_items).long()
    alias_category = trans_to_cuda(alias_category).long()

    N_seq = alias_category.shape[1]

    total_adj = trans_to_cuda(total_adj).float()[:,:N, :N]
    total_items = trans_to_cuda(total_items).long()[:,:N]
    
    mask = trans_to_cuda(mask).long()[:,:N_seq]

    A_hat, D_hat = get_overlap(items.cpu().numpy())
    A_hat = trans_to_cuda(torch.Tensor(A_hat))
    D_hat = trans_to_cuda(torch.Tensor(D_hat))

    hidden_mix = model(total_items, total_adj)

    get1_mix = lambda i: hidden_mix[i][alias_items[i]] 
    seq_hidden1_mix = torch.stack([get1_mix(i) for i in torch.arange(len(alias_items)).long()])
    get2_mix = lambda i: hidden_mix[i][alias_category[i]] 
    seq_hidden2_mix = torch.stack([get2_mix(i) for i in torch.arange(len(alias_category)).long()])
    
    select, b = model.compute_scores(seq_hidden1_mix, seq_hidden2_mix, mask)
    select = model.LineConv(select, D_hat, A_hat)
    item_scores = torch.matmul(select, b.transpose(1, 0))
    
    # for i in items:
    #     if 21066 in i.numpy():
    #         if model.print == False:
    #             index = 80
    #             print(items[index])
    #             print(total_adj[index][: 10, : 10])
    #             print(item_scores[index].topk(20)[1].detach().numpy())
    #             print(targets[index])
    #             model.print = True
    #             raise KeyboardInterrupt
    # for i in range(select.shape[0]):
    #     cache = [select[i].detach().numpy(), targets[i].item()]
    #     model.data.append(cache)

    return targets, item_scores

def train_test(model, train_data, test_data, writer, epoch, topk):
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=0, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    length = len(train_loader)
    for i, data in enumerate(tqdm(train_loader, colour='green', desc=f'Epoch {epoch}', leave=False)):
        model.optimizer.zero_grad()
        targets, item_scores = forward(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(item_scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()

        if i % 10 == 0:
            writer.add_scalar('loss/train_batch_loss', loss.item(), epoch * length + i)
            
    tqdm.write('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    writer.add_scalar('loss/train_loss', total_loss, epoch)

    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=0, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    hit, mrr = [[] for i in topk], [[] for i in topk]
    total_loss = 0.0

    for data in tqdm(test_loader, colour='green', desc='Estimating', leave=False):
        # targets, scores, con_loss = forward(model, data)
        targets, item_scores = forward(model, data)
        loss = model.loss_function(item_scores, trans_to_cuda(targets).long() - 1)
        total_loss += loss.item()
        targets = targets.numpy()

        for index, i in enumerate(topk):
            sub_scores = item_scores.topk(i)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target in zip(sub_scores, targets):
                hit[index].append(np.isin(target - 1, score))
                if len(np.where(score == target - 1)[0]) == 0:
                    mrr[index].append(0)
                else:
                    mrr[index].append(1 / (np.where(score == target - 1)[0][0] + 1))

    for index, j in enumerate(topk):
        hit[index] = np.mean(hit[index]) * 100
        mrr[index] = np.mean(mrr[index]) * 100
        writer.add_scalar(f'index/hit@{j}', hit[index], epoch)
        writer.add_scalar(f'index/mrr@{j}', mrr[index], epoch)

    writer.add_scalar('loss/test_loss', total_loss, epoch)
    return hit, mrr