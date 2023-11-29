from typing import Optional
from math import sqrt
import shutil
import torch.nn.functional as F

import os
import time
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from tqdm import trange
from torch_geometric.nn import MessagePassing
from tap import Tap

class ExplainerParser(Tap):
    t0: float = 5.0                   # temperature denominator
    t1: float = 1.0                   # temperature numerator
    coff_size: float = 0.01           # constrains on mask size
    coff_ent: float = 5e-4            # constrains on smooth and continuous mask
    lr = 0.001

    training_patient: int = 10  #  用来early stop
    training_epochs: int = 200
    saving_epochs: int = 10
    EPS = 1e-6

explainer_args = ExplainerParser().parse_args(known_only=True)

def save_best(model, model_path, model_name, is_best_loss, device): # PGExplainer就包括2层全连接网络
    print('saving....')
    model.to('cpu')

    pth_name = f"pgexplainer_{model_name}_latest.pth"
    best_loss_pth_name = f'pgexplainer_{model_name}_best.pth'
    ckpt_path = os.path.join(model_path, pth_name)
    torch.save(model.state_dict(), ckpt_path)

    if is_best_loss:
        shutil.copy(ckpt_path, os.path.join(model_path, best_loss_pth_name))
    model.to(device)

def inv_sigmoid(t: torch.Tensor):
    """ except the case t is 0 or 1 """
    if t.shape[0] != 0:
        if t.min().item() == 0 or t.max().item() == 1:
            t = 0.99 * t + 0.005
    ret = - torch.log(1 / t - 1)
    return ret


class PGExplainer(nn.Module):
    def __init__(self, model, epochs: int = explainer_args.training_epochs, lr: float = explainer_args.lr,
                 num_hops: Optional[int] = None, device = "cpu"):
        # lr=0.005, 0.003
        super(PGExplainer, self).__init__()
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.__num_hops__ = num_hops
        self.device = device

        self.coff_size = explainer_args.coff_size
        self.coff_ent = explainer_args.coff_ent
        self.init_bias = 0.0
        self.t0 = explainer_args.t0
        self.t1 = explainer_args.t1

        self.elayers = nn.ModuleList()
        input_feature = self.model.input_size * 2  # 2
        # 总共2层全连接网络
        self.elayers.append(nn.Sequential(nn.Linear(input_feature, 64), nn.ReLU()))
        self.elayers.append(nn.Linear(64, 1))
        self.elayers.to(self.device)



    def set_train_loader(self, train_loader):
        self.train_loader = train_loader


    def __set_masks__(self, x, edge_index, edge_mask=None):
        """ Set the weights for message passing """
        (N, F), E = x.size(), edge_index.size(1) # N: num_node  F:input_feature_dim  E: edge_num
        std = 0.1
        init_bias = self.init_bias
        self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * std)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))

        if edge_mask is None:
            self.edge_mask = torch.randn(E) * std + init_bias
        else:
            self.edge_mask = edge_mask

        self.edge_mask.to(self.device)
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        """ clear the edge weights to None """
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.node_feat_masks = None
        self.edge_mask = None


    def __loss__(self, prob, ori_pred):
        """
        the pred loss encourages the masked graph with higher probability,
        the size loss encourage small size edge mask,
        the entropy loss encourage the mask to be continuous.
        """
        logit = prob[ori_pred]  # mask后预测的概率
        logit = logit + explainer_args.EPS
        pred_loss = -torch.log(logit)

        # size
        edge_mask = torch.sigmoid(self.mask_sigmoid)
        size_loss = self.coff_size * torch.sum(edge_mask)

        # entropy
        edge_mask = edge_mask * 0.99 + 0.005
        mask_ent = - edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)
        mask_ent_loss = self.coff_ent * torch.mean(mask_ent)

        loss = pred_loss + size_loss + mask_ent_loss
        return loss


    def get_model_output(self, x, edge_index, edge_mask=None):
        """ return the model outputs with or without (w/wo) edge mask  """
        self.model.eval()
        self.__clear_masks__()
        if edge_mask is not None:
            self.__set_masks__(x, edge_index, edge_mask.to(self.device))

        with torch.no_grad():
            data = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
            data.to(self.device)
            outputs = self.model(data=data)

        self.__clear_masks__()
        return outputs


    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        """ Sample from the instantiation of concrete distribution when training
        \epsilon \sim  U(0,1), \hat{e}_{ij} = \sigma (\frac{\log \epsilon-\log (1-\epsilon)+\omega_{i j}}{\tau})

        log_alpha: [num_node * num_node]
        beta = τ
        """
        if training:
            random_noise = torch.rand(log_alpha.shape) # ϵ
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta  # =σ((logϵ−log(1−ϵ)+ωij​)/τ)
            gate_inputs = gate_inputs.sigmoid() # e_ij
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs


    def forward(self, inputs, training=None):
        x, embed, edge_index, tmp = inputs  # X向量，Z向量，边，τ
        nodesize = embed.shape[0]
        feature_dim = embed.shape[1]
        f1 = embed.unsqueeze(1).repeat(1, nodesize, 1).reshape(-1, feature_dim)
        f2 = embed.unsqueeze(0).repeat(nodesize, 1, 1).reshape(-1, feature_dim)

        # using the node embedding to calculate the edge weight
        f12self = torch.cat([f1, f2], dim=-1)
        h = f12self.to(self.device)
        for elayer in self.elayers:
            h = elayer(h)
        values = h.reshape(-1)  # value = w_ij，[num_node * num_node]
        values = self.concrete_sample(values, beta=tmp, training=training) # value = e_ij
        self.mask_sigmoid = values.reshape(nodesize, nodesize) # e^ij​=σ((logϵ−log(1−ϵ)+ωij​)/τ)

        # set the symmetric edge weights
        sym_mask = (self.mask_sigmoid + self.mask_sigmoid.transpose(0, 1)) / 2
        edge_mask = sym_mask[edge_index[0], edge_index[1]]

        # inverse the weights before sigmoid in MessagePassing Module
        edge_mask = inv_sigmoid(edge_mask)
        self.__clear_masks__()
        self.__set_masks__(x, edge_index, edge_mask)

        # the model prediction with edge mask
        data = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
        data.to(self.device)
        outputs = self.model(data=data)
        return outputs[0].squeeze(), edge_mask

    # input: a set of graph data
    def train_explanation_network(self, trainset, model_path, model_name):
        """ train the explantion network for graph classification task """
        optimizer = Adam(self.elayers.parameters(), lr=self.lr, betas=(0.9, 0.999))

        # collect the embedding of nodes
        emb_dict = {}  # 用来保存每个结点的Z向量
        ori_pred_dict = {}
        with torch.no_grad():
            self.model.eval()
            for gid in trange(len(trainset)):
                data = trainset[gid].to(self.device)
                prob, emb = self.get_model_output(data.x, data.edge_index)  # 获取模型的输出结果

                _, prediction = torch.max(prob, -1)
                if prediction.cpu() == 0: # 跳过分类错误的
                    continue

                emb_dict[gid] = emb.data.cpu() # 获取Z向量
                ori_pred_dict[gid] = prediction.cpu() # Yo，是分类值，不是概率

        # train the mask generator
        duration = 0.0

        min_loss = 3.4
        early_stop_count = 0


        for epoch in range(self.epochs):
            print('epoch:{}'.format(epoch + 1))

            tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs)) # τ
            self.elayers.train()
            optimizer.zero_grad()
            tic = time.perf_counter()
            loss_list = list()
            for gid in trange(len(trainset)):
                if gid not in emb_dict.keys(): # 跳过分类错误的
                    continue
                data = trainset[gid]
                data = data.to(self.device)
                self.model = self.model.to(self.device)

                prob, edge_mask = self.forward((data.x, emb_dict[gid], data.edge_index, tmp), training=True)
                prob = F.softmax(prob)

                # edge_mask: shape = [edge_num]
                loss_tmp = self.__loss__(prob, ori_pred_dict[gid])  # 计算loss

                loss_tmp.backward()
                loss_list.append(loss_tmp.item())


            optimizer.step()
            duration += time.perf_counter() - tic

            loss = np.nanmean(np.array(loss_list))

            print(f'Epoch: {epoch} | Loss: {loss}')

            is_best_loss = (loss < min_loss)

            if loss < min_loss:
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count > explainer_args.training_patient:
                break


            if is_best_loss:
                min_loss = loss
                early_stop_count = 0

            if is_best_loss or epoch % explainer_args.saving_epochs == 0:
                save_best(self.elayers, model_path, model_name, is_best_loss, self.device)

            # torch.save(self.elayers.cpu().state_dict(), self.ckpt_path)
            # self.elayers.to(self.device)
        print(f"training time is {duration:.5}s")


    def explain(self, x, edge_index, **kwargs):
        data = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
        data = data.to(self.device)
        with torch.no_grad():
            prob, emb = self.get_model_output(data.x, data.edge_index)
            _, edge_mask = self.forward((data.x, emb, data.edge_index, 1.0), training=False)
            sorted_indices = edge_mask.sort(descending=True)
        return sorted_indices, False  # [num_node, num_node]


    def set_cpg(self, cpgs: list):
        self.cpgs = cpgs


