import os
import os.path as osp
import sys
g_pwd = osp.abspath(osp.dirname(osp.dirname(os.getcwd())))
sys.path.append(g_pwd)

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
from KG_Exp.KG_dataset import FB15K237_Dataset

from KG_Exp.KG_method import TransE_Conv
from KG_Exp.KG_method import DistMult_Conv

from KG_Exp.KG_explain import KEX_mask, calculate_KEX_Metrics
from KG_Exp.KG_explain import Gradient_CAM_mask, calculate_GC_Metrics
from KG_Exp.KG_explain import Saliency_Map_mask, calculate_SM_Metrics
from KG_Exp.KG_explain.metric import calculate_metric
from KG_Exp.KG_explain.visual import visual_node

import pickle


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str,
                            choices=['train', 'explain', 'metric'])
    parser.add_argument('--relation_id', type=int)
    parser.add_argument('--conv', type=str,
                            choices=['transe', 'distmult'])
    parser.add_argument('--batch_size', type=int,
                        default=100)
    parser.add_argument('--epoch', type=int,
                        default=100)
    parser.add_argument('--node', type=int,
                        default=0)
    args = parser.parse_args()
    return args


class Net(torch.nn.Module):
    def __init__(self, ent_embedding, rel_embedding, relation_num, class_num, conv):
        super(Net, self).__init__()
        self.ent_embedding = ent_embedding
        self.rel_embedding = torch.cat([rel_embedding, rel_embedding], dim=0)
        
        self.in_channels = self.ent_embedding.shape[1]
        self.out_channels = self.in_channels
        
        if conv == 'transe':
            conv_func = TransE_Conv
        else:
            conv_func = DistMult_Conv
            
        self.conv1 = conv_func(in_channels=self.in_channels, out_channels=50,
                              num_relations=relation_num, num_baSses=5, root_weight=True)
        self.conv2 = conv_func(in_channels=50, out_channels=class_num,
                              num_relations=relation_num, num_bases=5, root_weight=True)

        self.rel_weight = nn.Parameter(torch.rand([100, 50]))


    def forward(self, x, edge_index, edge_type, target_node=None):

        if target_node == None:
            return
        x[target_node] = torch.zeros(x[target_node].shape).to(x.device)

        edge_embeddings = self.rel_embedding[edge_type]
        x = self.conv1(x, edge_index, edge_embeddings, edge_type)
        
        # For Gradient-CAM
        self.layer2_save = x.detach()

        self.rel_embedding2 = torch.matmul(self.rel_embedding, self.rel_weight)
        edge_embeddings = self.rel_embedding2[edge_type]
        x = self.conv2(x, edge_index, edge_embeddings, edge_type)

        return F.log_softmax(x, dim=1)


def train():
    model.train()

    train_index_tuple = torch.split(data.train_idx, batch_size, dim=0)
    total_loss = 0
    for i, sub_train_index in enumerate(train_index_tuple):
        optimizer.zero_grad()
        node_index, edge_index, node_mapping, edge_mask = k_hop_subgraph(node_idx=sub_train_index, num_hops=2,
                                                                    edge_index=data.edge_index, relabel_nodes=True)
        edge_type = data.edge_type[edge_mask]
        out = model(ent_embedding[node_index], edge_index, edge_type, target_node=node_mapping)
        loss = F.nll_loss(out[node_mapping], data.train_y[i * batch_size: min((i+1) * batch_size, data.train_y.shape[0])])
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss


@torch.no_grad()
def test():
    model.eval()

    pred = torch.tensor([]).to(device)
    test_index_tuple = torch.split(torch.cat([data.train_idx, data.test_idx]), batch_size, dim=0)
    for i, sub_test_index in enumerate(test_index_tuple):
        node_index, edge_index, node_mapping, edge_mask = k_hop_subgraph(node_idx=sub_test_index, num_hops=2,
                                                                   edge_index=data.edge_index, relabel_nodes=True)
        edge_type = data.edge_type[edge_mask]
        out = model(ent_embedding[node_index], edge_index, edge_type, target_node=node_mapping)[node_mapping].argmax(dim=-1)
        pred = torch.cat([pred, out])

    train_acc = pred[:data.train_idx.shape[0]].eq(data.train_y).to(torch.float).mean()
    test_acc = pred[data.train_idx.shape[0]:].eq(data.test_y).to(torch.float).mean()

    if 'max_acc' in globals():
        path = osp.dirname(osp.realpath(__file__)) + "/../../model_save/fb15k237_relation" + "_" + str(args.relation_id) + "_" + args.conv + "_conv.pt"
        torch.save(model.state_dict(), path)
        globals()['max_acc'] = test_acc.item()

    return train_acc.item(), test_acc.item()


def pred_node(target_node, model, data):
    sub_node_index, sub_edge_index, mapping, edge_mask = k_hop_subgraph(target_node, 2, data.edge_index,
                                                                        relabel_nodes=True)
    sub_edge_type = data.edge_type[edge_mask] 

    if sub_edge_type.size(0) == 0:
        return -1

    out = model(ent_embedding[sub_node_index], sub_edge_index, sub_edge_type, target_node=mapping)

    pred = out.argmax(dim=-1)[mapping].item()
    return pred


args = argument()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'fb15k237_' + str(args.relation_id))
dataset = FB15K237_Dataset(path, args.relation_id)
data = dataset[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = args.batch_size
epoch = args.epoch


if args.conv == 'distmult':
    ent_embedding = torch.tensor(np.load(osp.dirname(osp.realpath(__file__)) + "/./TransE_preTrain/ent_embeddings.npy")).to(device)
    rel_embedding = torch.tensor(np.load(osp.dirname(osp.realpath(__file__)) + "/./TransE_preTrain/rel_embeddings.npy")).to(device)
else:
    ent_embedding = torch.tensor(np.load(osp.dirname(osp.realpath(__file__)) + "/./DistMult_preTrain/ent_embeddings.npy")).to(device)
    rel_embedding = torch.tensor(np.load(osp.dirname(osp.realpath(__file__)) + "/./DistMult_preTrain/rel_embeddings.npy")).to(device)

model = Net(ent_embedding, rel_embedding,
            (data.edge_type.max() + 1).item(), (data.train_y.max() + 1).item(), args.conv).to(device)
data = data.to(device)


if args.action == 'train':   
    max_acc = 0
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(1, epoch):
        loss = train()
        train_acc, test_acc = test()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f} '
              f'Test: {test_acc:.4f}')

    print("success save")

    path = osp.dirname(osp.realpath(__file__)) + "/../../model_save/fb15k237_relation" + "_" + str(args.relation_id) + "_" + args.conv + "_conv.pt"
    model.load_state_dict(torch.load(path))
    train_acc, test_acc = test()
    print(f'Train: {train_acc:.4f} '
          f'Test: {test_acc:.4f}')
    


if args.action == 'explain':
    
    path = osp.dirname(osp.realpath(__file__)) + "/../../model_save/fb15k237_relation" + "_" + str(args.relation_id) + "_" + args.conv + "_conv.pt"
    
    if os.path.exists(path) == False:
        sys.exit(0)

    model.load_state_dict(torch.load(path))
    ground_truth = data.test_y

    same_GE_list, pred_decay_GE_list, sparity_GE_list = [], [], []
    same_SM_list, pred_decay_SM_list, sparity_SM_list = [], [], []
    same_GC_list, pred_decay_GC_list, sparity_GC_list = [], [], []

    pred_true_node_list = []
    result_dict = {}
    
    save_path = osp.dirname(osp.realpath(__file__)) + "/../../explain_result/fb15k237_relation" + "_" + str(args.relation_id) + "_" + args.conv + "_explain_result.json"
    
    for i in range(0, data.test_idx.shape[0]):

        target_node = data.test_idx[i].item()
        
        pred = pred_node(target_node, model, data)
        if pred != ground_truth[i].item():
            continue
        print("node: " + str(target_node))

        edge_mask_score_GE = KEX_mask(target_node, pred_tail=None, model=model, data=data)
        same_GE, pred_decay_GE, sparity_GE = calculate_KEX_Metrics(target_node, model, data, edge_mask_score_GE, threshold=0.5)
        same_GE_list.append(same_GE)
        pred_decay_GE_list.append(pred_decay_GE)
        sparity_GE_list.append(sparity_GE)

        node_mask_score_SM = Saliency_Map_mask(target_node, model=model, data=data)
        same_SM, pred_decay_SM, sparity_SM = calculate_SM_Metrics(target_node, model, data, node_mask_score_SM, threshold=0.1)
        same_SM_list.append(same_SM)
        pred_decay_SM_list.append(pred_decay_SM)
        sparity_SM_list.append(sparity_SM)

        node_mask_score_GC = Gradient_CAM_mask(target_node, model=model, data=data)
        same_GC, pred_decay_GC, sparity_GC = calculate_GC_Metrics(target_node, model, data, node_mask_score_GC, threshold=0.1)
        same_GC_list.append(same_GC)
        pred_decay_GC_list.append(pred_decay_GC)
        sparity_GC_list.append(sparity_GC)

        pred_true_node_list.append(target_node)

        node_dict = dict()
        node_dict["node"] = target_node
        node_dict["label"] = pred

        node_dict["same_GE"] = same_GE
        node_dict["pred_decay_GE"] = pred_decay_GE
        node_dict["sparity_GE"] = sparity_GE

        node_dict["same_SM"] = same_SM
        node_dict["pred_decay_SM"] = pred_decay_SM
        node_dict["sparity_SM"] = sparity_SM

        node_dict["same_GC"] = same_GC
        node_dict["pred_decay_GC"] = pred_decay_GC
        node_dict["sparity_GC"] = sparity_GC

        result_dict[target_node] = node_dict
        print("\n")

        if i % 50 == 0:
            with open(save_path, 'w') as f:
                f.write('\n')
                json.dump(result_dict, f)

    with open(save_path, 'w') as f:
        f.write('\n')
        json.dump(result_dict, f)

    result_path = osp.dirname(osp.realpath(__file__)) + "/../../explain_result/fb15k237_relation" + "_" + str(args.relation_id) + "_" + args.conv + "_explain_result.json"
    with open(result_path, 'r') as f:
        node_dict = json.load(f)
        
    result_dict = calculate_metric(node_dict)


if args.action == 'visual':
    path = osp.dirname(osp.realpath(__file__)) + "/../../model_save/fb15k237_relation" + "_" + str(args.relation_id) + "_" + args.conv + "_conv.pt"
    model.load_state_dict(torch.load(path))
    target_node = args.node
    visual_node(target_node, pred_tail=None, model=model, data=data)
