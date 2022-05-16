import torch
import torch.nn as nn
from layer import *
import numpy as np


class HeteGAT_multi(nn.Module):
    def __init__(self, inputs_list, DNN_hyper, TNN_hyper, Deco_hyper, attn_drop, ffd_drop,
                 bias_mat_list, n_heads, dropout, device, activation=nn.ELU(), residual=False):
        super(HeteGAT_multi, self).__init__()
        self.inputs_list_drug = inputs_list[0]
        self.inputs_list_target = inputs_list[1]
        self.dropout = dropout
        self.drug_nn = NN(DNN_hyper[0], DNN_hyper[1], DNN_hyper[2], DNN_hyper[3], dropout)
        self.target_nn = NN(TNN_hyper[0], TNN_hyper[1], TNN_hyper[2], TNN_hyper[3], dropout)
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.bias_mat_list = bias_mat_list
        self.bias_mat_list_drug = [bias_mat_list[i][0] for i in range(len(self.bias_mat_list))]
        self.bias_mat_list_target = [bias_mat_list[i][1] for i in range(len(self.bias_mat_list))]
        self.n_heads = n_heads
        self.activation = activation
        self.residual = residual
        self.mp_att_size = 128
        self.device = device
        self.layers = GraphAttentionLayer(128, 16,  self.dropout, 0.2)
        self.simpleAttLayer = SimpleAttLayer(128, self.mp_att_size, time_major=False, return_alphas=True)
        self.a_omega = torch.nn.Parameter(torch.randn(128, 1))
        self.DTI_Decoder = DTI_Decoder(Deco_hyper[0], Deco_hyper[1], Deco_hyper[2])
        self.softmax = torch.nn.Softmax(dim=0)

    def bceloss(self, scores, labels):
        loss = nn.BCELoss()

        return loss(scores, labels)

    def forward(self, x, edge_index):
        drug_all = []
        target_all = []
        """对每个异构图做操作"""
        for j in range(len(self.bias_mat_list)):  # 3个异构图
            fea_list_nn = self.drug_nn(x[0])
            embed_list_drug = []
            for i, biases in enumerate(self.bias_mat_list_drug[j]):
                attns = []
                for _ in range(self.n_heads[0]):
                    attns.append(self.layers(fea_list_nn, biases))
                h_1 = torch.cat(attns, dim=1)  # 多头拼接
                embed_list_drug.append(h_1.reshape(-1, 1, h_1.shape[-1]))
            multi_embed = torch.cat(embed_list_drug, dim=1)

            final_embed_drug, att_val = self.simpleAttLayer(multi_embed)
            drug_all.append(final_embed_drug)

            # 对于靶标
            fea_list_nn = self.target_nn(x[1])
            embed_list_target = []
            for i, biases in enumerate(self.bias_mat_list_target[j]):
                attns = []
                for _ in range(self.n_heads[0]):
                    attns.append(self.layers(fea_list_nn, biases))
                h_1 = torch.cat(attns, dim=1)
                embed_list_target.append(h_1.reshape(-1, 1, h_1.shape[-1]))
            multi_embed = torch.cat(embed_list_target, dim=1)
            final_embed_target, att_val = self.simpleAttLayer(multi_embed)
            target_all.append(final_embed_target)


        ##########
        final_drug = combine(self, drug_all)  # 元图级注意力层
        final_target = combine(self, target_all)

        """写两个嵌入的相似度比较"""
        output = self.DTI_Decoder(final_drug, final_target, edge_index).view(-1)
        return output