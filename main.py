import numpy as np
import os
import pandas as pd
import csv
import torch
from model import *
from utils import *
import torch.optim as optim
from sklearn.metrics import accuracy_score
import argparse
import random

parser = argparse.ArgumentParser(description='MHADTI')
parser.add_argument('--seed', type=int, default=223, help='Random seed.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--gat_noutput', type=int, default=128,
                    help='GAT output feature dim and the input dim of Decoder')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate (1 - keep probability).')
###############################################################
# Model hyper setting
# Protein_NN
parser.add_argument('--target_ninput', type=int, default=40,
                    help='target vector size')
parser.add_argument('--tnn_nlayers', type=int, default=1,
                    help='Target_nn layers num')
parser.add_argument('--tnn_nhid', type=str, default='[]',
                    help='tnn hidden layer dim, like [200,100] for tow hidden layers')
# Drug_NN
parser.add_argument('--drug_ninput', type=int, default=881,
                    help='Drug fingerprint dimension')
parser.add_argument('--dnn_nlayers', type=int, default=1,
                    help='dnn_nlayers num')
parser.add_argument('--dnn_nhid', type=str, default='[]',
                    help='dnn hidden layer dim, like [200,100] for tow hidden layers')
# Decoder
parser.add_argument('--DTI_nn_nhid', type=list, default=[128, 128, 128],
                    help='DTI_nn hidden layer dim, like [200,100] for tow hidden layers')
parser.add_argument('--DTI_nn_nlayers', type=int, default=3,
                    help='Protein_nn layers num')
args = parser.parse_args()
np.random.seed(args.seed)

torch.manual_seed(args.seed)
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)

dnn_hyper = [args.drug_ninput, args.dnn_nhid, args.gat_noutput, args.dnn_nlayers]
tnn_hyper = [args.target_ninput, args.tnn_nhid, args.gat_noutput, args.tnn_nlayers]
Deco_hyper = [args.gat_noutput, args.DTI_nn_nhid, args.DTI_nn_nlayers]
"""取出全部数据集，标签全为1"""

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
nd = 4358
nt = 2407

drug_feature = np.loadtxt("data/drug_feature.txt", delimiter=",")
target_feature = np.loadtxt("data/target_feature.txt", delimiter=",")
fea_list = [drug_feature, target_feature]
fea_list = [torch.tensor(fea, dtype=torch.float32).to(device) for fea in fea_list]


def convert_input(file_directory):
    adj_list_drug = [np.loadtxt("data/{}/DTD.txt".format(file_directory), delimiter=","),
                     np.loadtxt("data/{}/DTTD.txt".format(file_directory), delimiter=",")]
    adj_list_target = [np.loadtxt("data/{}/TDT.txt".format(file_directory), delimiter=","),
                       np.loadtxt("data/{}/TDDT.txt".format(file_directory), delimiter=",")]
    adj_list_drug = [adj for adj in adj_list_drug]
    adj_list_target = [adj for adj in adj_list_target]
    biases_list_drug = [torch.from_numpy(adj).to(device) for adj in adj_list_drug]
    biases_list_target = [torch.from_numpy(adj).to(device) for adj in adj_list_target]
    return biases_list_drug, biases_list_target


f1 = "heterogeneous1"
f2 = "heterogeneous2"
f3 = "heterogeneous3"
biases_list_drug1, biases_list_target1 = convert_input(f1)
print(1, biases_list_drug1[0].shape, biases_list_target1[0].shape)
biases_list_drug2, biases_list_target2 = convert_input(f2)
print(2, biases_list_drug2[0].shape, biases_list_target2[0].shape)
biases_list_drug3, biases_list_target3 = convert_input(f3)
print(3, biases_list_drug3[0].shape, biases_list_target3[0].shape)

ft_size_drug = 881
ft_size_target = 40
batch_size = 1
patience = 100
lr = 0.005
l2_coef = 0.0005
n_heads = [8, 1]
residual = False

biases_list = [[biases_list_drug1, biases_list_target1], [biases_list_drug2, biases_list_target2],
               [biases_list_drug3, biases_list_target3]]
model = HeteGAT_multi(inputs_list=fea_list, DNN_hyper=dnn_hyper,
                      TNN_hyper=tnn_hyper, Deco_hyper=Deco_hyper, attn_drop=0.5, ffd_drop=0.0,
                      bias_mat_list=biases_list,
                      n_heads=n_heads, dropout=args.dropout, device=device, activation=nn.ELU(),
                      residual=False)

optimizer = optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
model.to(device)


def main():
    print("训练节点个数：", len(edge_index))
    for epoch in range(200):
        loss_score, acc_score, auc_score, aupr_score = train()
        print("epoch:{:03d}".format(epoch), 'loss_train: {:.4f}'.format(loss_score),
              'acc_train: {:.4f}'.format(acc_score), "auc_train: {:.4f}".format(auc_score),
              "aupr_train: {:.4f}".format(aupr_score))


def train():
    model.train()
    correct = 0
    output = model(fea_list, edge_index)
    loss = model.bceloss(output, label_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    acc_value = accuracy(output, label_train)
    auc_value = auc(output, label_train)
    aupr_value = aupr(output, label_train)
    return loss, acc_value, auc_value, aupr_value


if __name__ == '__main__':
    drug_list = [v for v in open("data/drug.txt", "r").read().split("\n")]
    target_list = [v for v in open("data/target.txt", "r").read().split("\n")]
    f_inter = "data/DT_interaction.txt"
    lt_inter = get_inter(f_inter, drug_list, target_list)
    # print(lt_inter)
    postive_train = lt_inter
    negative_train = []
    i = 0
    """构造负样本"""
    while i < len(postive_train):
        # print(i, end=" ")
        drug_index = random.choice(range(nd))
        target_index = random.choice(range(nt))
        neg_pos = [drug_index, target_index]
        if neg_pos in lt_inter or neg_pos in negative_train:
            continue
        negative_train.append(neg_pos)
        i = i + 1
    edge_index = []  # 所有的训练样本
    for v in postive_train:
        edge_index.append(v)
    for v in negative_train:
        edge_index.append(v)
    torch.tensor(edge_index).to(device)
    l_train_pos = torch.ones(len(postive_train))
    l_train_neg = torch.zeros(len(negative_train))
    label_train = torch.hstack((l_train_pos, l_train_neg))
    main()
