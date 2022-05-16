import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import ssl

import torch
from torch import nn


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(
            torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data,
                                gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SimpleAttLayer(nn.Module):
    def __init__(self, inputs, attention_size, time_major=False, return_alphas=False):
        super(SimpleAttLayer, self).__init__()
        self.hidden_size = inputs
        self.return_alphas = return_alphas
        self.time_major = time_major
        self.w_omega = nn.Parameter(torch.Tensor(self.hidden_size, attention_size))
        self.b_omega = nn.Parameter(torch.Tensor(attention_size))
        self.u_omega = nn.Parameter(torch.Tensor(attention_size, 1))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_omega)
        nn.init.zeros_(self.b_omega)
        nn.init.xavier_uniform_(self.u_omega)

    def forward(self, x):
        v = self.tanh(torch.matmul(x, self.w_omega) + self.b_omega)
        vu = torch.matmul(v, self.u_omega)
        alphas = self.softmax(vu)
        output = torch.sum(x * alphas.reshape(alphas.shape[0], -1, 1), dim=1)
        if not self.return_alphas:
            return output
        else:
            return output, alphas


class NN(nn.Module):
    def __init__(self, ninput, nhidden, noutput, nlayers, dropout):
        # 之前的dropout是0.5
        """
        """
        super(NN, self).__init__()
        self.dropout = dropout
        self.encode = torch.nn.ModuleList([
            torch.nn.Linear(ninput if l == 0 else nhidden[l - 1], nhidden[l] if l != nlayers - 1 else noutput) for l in
            range(nlayers)])
        self.BatchNormList = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(num_features=nhidden[l] if l != nlayers - 1 else noutput) for l in range(nlayers)])
        self.LayerNormList = torch.nn.ModuleList([
            torch.nn.LayerNorm(nhidden[l] if l != nlayers - 1 else noutput) for l in range(nlayers)])

    def forward(self, x):
        for l, linear in enumerate(self.encode):
            x = F.relu(linear(x))
            x = self.BatchNormList[l](x)
            x = self.LayerNormList[l](x)
            x = F.dropout(x, self.dropout)
        return x


class DTI_Decoder(nn.Module):
    def __init__(self, Nodefeat_size, nhidden, nlayers, dropout=0.5):
        super(DTI_Decoder, self).__init__()
        self.Protein_num = 2407
        self.Drug_num = 4358
        self.dropout = dropout
        self.nlayers = nlayers
        self.decode = torch.nn.ModuleList([
            torch.nn.Linear(Nodefeat_size if l == 0 else nhidden[l - 1], nhidden[l]) for l in
            range(nlayers)])
        self.BatchNormList = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(num_features=nhidden[l]) for l in range(nlayers)])
        self.linear = torch.nn.Linear(nhidden[nlayers - 1], 1)

    def forward(self, final_emb_drug, final_emb_target, edge_index):
        protein_index = [v[1] for v in edge_index]
        drug_index = [v[0] for v in edge_index]
        drug_features = final_emb_drug[drug_index]
        protein_features = final_emb_target[protein_index]
        pair_nodes_features = protein_features * drug_features
        for l, dti_nn in enumerate(self.decode):
            # print(l, dti_nn)
            pair_nodes_features = F.dropout(pair_nodes_features, self.dropout)
            pair_nodes_features = F.relu(dti_nn(pair_nodes_features))
            pair_nodes_features = self.BatchNormList[l](pair_nodes_features)
        pair_nodes_features = F.dropout(pair_nodes_features, self.dropout)
        output = self.linear(pair_nodes_features)
        return torch.sigmoid(output)


def combine(self, fea):
    num_fea = len(fea)
    drug_num = fea[0].shape[0]
    att_size = fea[0].shape[1]
    all_features = torch.cat(fea, axis=0)
    temp_features = torch.matmul(all_features, self.a_omega)
    temp_value = temp_features.view(num_fea, drug_num, 1)
    beta = self.softmax(temp_value)
    features = all_features.view(num_fea, drug_num, att_size)
    final_features = torch.zeros_like(fea[0])
    for i in range(num_fea):
        t_features = beta[i] * features[i]
        final_features = final_features + t_features
    return final_features
