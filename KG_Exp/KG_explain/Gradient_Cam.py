import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph, subgraph
import numpy as np

import math


def filt_node(node_index, edge_index, grad_list, backward_node):

    node_index_filter = list()
    grad_list_draw = list()

    edge_set = set(map(tuple, edge_index.T.detach().cpu().numpy().tolist()))
  
    grad_list_sort = torch.tensor(grad_list)
    grad_list_sort, sort = torch.sort(grad_list_sort, descending=True)
    sort = sort.detach().cpu().numpy()
    node_index_sort = node_index[sort]

    for i in range(len(grad_list_sort)):
        grad = grad_list_sort[i].item()
        node_id = node_index_sort[i].item()

        if node_id == backward_node:
            node_index_filter.append(node_id)
            grad_list_draw.append(1)
            continue

        if i < 5 or (i % 10 == 0 and grad > 0.02) or (backward_node, i) in edge_set or (i, backward_node) in edge_set:
            node_index_filter.append(node_id)
            grad_list_draw.append(grad)

    edge_index, edge_attr = subgraph(torch.tensor(node_index_filter), edge_index)
    edge_index_list = edge_index.detach().cpu().numpy().tolist()
    
    node_in_edge_index = set(edge_index_list[0] + edge_index_list[1])
    node_save = list()
    grad_save = list()
    for i, node in enumerate(node_index_filter):
        if node in node_in_edge_index:
            node_save.append(node)
            grad_save.append(grad_list_draw[i])

    node_index_filter = node_save
    grad_list_draw = grad_save

    return node_index_filter, edge_index_list, grad_list_draw


def Gradient_CAM_mask(target_node, model, data, id_entity=None, edge_relation=None):

    sub_node_index, sub_edge_index, mapping, edge_mask = k_hop_subgraph(target_node, 2, data.edge_index,
                                                                        relabel_nodes=True)
    sub_edge_type = data.edge_type[edge_mask]

    model.ent_embedding.requires_grad = True
    x = model.ent_embedding[sub_node_index]
    out = model(x, sub_edge_index, sub_edge_type, target_node=mapping)

    pred = out.argmax(dim=-1)[mapping]
    print("model pred label: " + str(pred.item()))

    out[mapping.item()][pred.item()].backward()
    model.ent_embedding.requires_grad = False
    
    feature_weight = torch.mean(model.layer2_save, dim=0)
    grad_list = list()
    
    grad = model.ent_embedding.grad
    sub_node_features = model.layer2_save
    
    for index in range(grad.shape[0]):
        if (index not in sub_node_index) or (index == target_node):
            grad_list.append(0)
        else:
            loc = (sub_node_index ==index).nonzero().item()
            score = F.relu(torch.sum(sub_node_features[loc] * feature_weight)).item()
            grad_list.append(score)

    if torch.from_numpy(np.array(grad_list)).max() != 0:
        grad_list = list(grad_list / np.linalg.norm(np.array(grad_list), keepdims=True))
    node_mask_score = torch.from_numpy(np.array(grad_list))

    return node_mask_score


def calculate_GC_Metrics(target_node, model, data, node_mask_score, threshold=0.1):

    sub_node_index, sub_edge_index, mapping, edge_mask = k_hop_subgraph(target_node, 2, data.edge_index,
                                                                        relabel_nodes=True)

    sub_edge_type = data.edge_type[edge_mask]
    node_mask_score = node_mask_score[sub_node_index]

    if sub_edge_type.size(0) == 0:
        print("node" + str(target_node) + " is an isolated point")
        return -1, -1, -1

    x = model.ent_embedding[sub_node_index]
    
    with torch.no_grad():
        out = model(x, sub_edge_index, sub_edge_type, target_node=mapping)
        pred1 = out[mapping].argmax().item()
        prob1 = out[mapping].reshape(-1)[pred1].item()

    total_node = (sub_edge_index.max() + 1).item()
    sub_node_index = sub_node_index[(node_mask_score < threshold)]

    sub_edge_index, sub_edge_type = subgraph(sub_node_index, data.edge_index, edge_attr=data.edge_type,
                                             relabel_nodes=True)
    x = model.ent_embedding[sub_node_index]
    mapping = (sub_node_index == target_node).nonzero().item()

    with torch.no_grad():
        out = model(x, sub_edge_index, sub_edge_type, target_node=mapping)

    pred2 = out[mapping].argmax().item()
    prob2 = out[mapping].reshape(-1)[pred1].item()

    same = True if pred1 == pred2 else False
    prob_decay = math.exp(prob1) - math.exp(prob2)

    import_node = (node_mask_score > threshold).nonzero().shape[0]
    sparity = import_node / total_node

    return same, prob_decay, sparity