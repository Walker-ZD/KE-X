import torch
from torch_geometric.utils import k_hop_subgraph
from KG_Exp.KG_explain.explainer import Explainer
import copy

import math


def edge_info(edge_list, edge_relation_list, edge_color):
    print("\n")
    for i, edge in enumerate(edge_list):
        
        print(str(edge[0]) + "    " + str(edge_relation_list[i]) + "    " + str(edge[1])
              + "   value: " + str(edge_color[i]))
        print("\n")
    print("\n")


def edge_process(edge_mask, edge_index, edge_relation):

    edge_mask, sort = torch.sort(edge_mask, descending=True)
    sort = sort.detach().cpu().numpy()
    edge_index[0] = edge_index[0][sort]
    edge_index[1] = edge_index[1][sort]
    edge_relation = edge_relation[sort]

    edge_list = list()
    edge_color = list()
    edge_relation_list = list()
    
    edge_mask = edge_mask[:150]
    index = []
    
    for i in range(15):
      
        index.append(i)
         
        edge_color.append(edge_mask[i].item())
        edge_list.append((int(edge_index[0][i].item()), int(edge_index[1][i].item())))
        edge_relation_list.append(edge_relation[i].item())

    edge_mask = edge_mask[index]
    return edge_list, edge_color, edge_mask, edge_relation_list


def edge_recover(edge_list, edge_relation_list, relation_number):
    for i in range(len(edge_list)):
        if edge_relation_list[i] >= relation_number:
            edge_relation_list[i] = edge_relation_list[i] - relation_number
            edge_list[i] = tuple([edge_list[i][1], edge_list[i][0]])

    return edge_list, edge_relation_list


def KEX_mask(target_node, pred_tail, model, data, id_entity=None, id_relation=None):

    y = torch.cat((data.train_y, data.test_y)).detach().cpu().numpy().tolist()
  
    explainer = Explainer(model, epochs=50, return_type='log_prob', lr=0.5, num_hops=2)
    node_feat_mask, edge_mask = explainer.explain_node(target_node, model.ent_embedding, data.edge_index,
                                                       label_given=None,
                                                        edge_type=data.edge_type)  

    edge_mask_return = copy.deepcopy(edge_mask)
    return edge_mask_return


def calculate_KEX_Metrics(target_node, model, data, edge_mask_score, threshold=0.1):
  
    sub_node_index, sub_edge_index, mapping, edge_mask = k_hop_subgraph(target_node, 2, data.edge_index,
                                                                        relabel_nodes=True)

    sub_edge_type = data.edge_type[edge_mask]
    edge_mask_score = edge_mask_score[edge_mask]

    if sub_edge_type.size(0) == 0:
        print("node" + str(target_node) + " is an isolated point")
        return -1, -1, -1

    x = model.ent_embedding[sub_node_index]
    
    with torch.no_grad():
        out = model(x, sub_edge_index, sub_edge_type, target_node=mapping)
        pred1 = out[mapping].argmax().item()  
        prob1 = out[mapping].reshape(-1)[pred1].item()
 
    total_node = (sub_edge_index.max() + 1).item()
    sub_edge_index = (sub_edge_index.T)[(edge_mask_score < threshold)].T
    sub_edge_type = (sub_edge_type.T)[(edge_mask_score < threshold)].T

    with torch.no_grad():
        out = model(x, sub_edge_index, sub_edge_type, target_node=mapping)

    
    pred2 = out[mapping].argmax().item()
    prob2 = out[mapping].reshape(-1)[pred1].item()

    same = True if pred1 == pred2 else False
    prob_decay = math.exp(prob1) - math.exp(prob2)

    import_node = (edge_mask_score > threshold).nonzero().shape[0]
    sparity = import_node / total_node

    return same, prob_decay, sparity


