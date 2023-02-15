import torch
from torch_geometric.utils import k_hop_subgraph
from KG_Exp.KG_explain.explainer import Explainer
import numpy as np
import networkx as nx
import networkx.drawing as draw
import matplotlib.pyplot as plt
import math
from matplotlib import cm


def edge_info(edge_list, edge_relation_list, edge_color, id_relation, id_entity):
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


def NormMinandMax(npdarr, min=0, max=1):
    arr = npdarr.flatten()
    Ymax = np.max(arr)
    Ymin = np.min(arr) 
    k = (max - min) / (Ymax - Ymin)
    last = min + k * (arr - Ymin)
    return last


def visual_node(target_node, pred_tail, model, data, id_entity=None, id_relation=None):
    explainer = Explainer(model, epochs=50, return_type='log_prob', lr=0.5, num_hops=2)
    node_feat_mask, edge_mask = explainer.explain_node(target_node, model.ent_embedding, data.edge_index,
                                                       label_given=None,
                                                       edge_type=data.edge_type)  
   
    edge_index = (data.edge_index.T)[(edge_mask > 0)].T
    edge_relation = data.edge_type[edge_mask > 0]
    edge_mask = edge_mask[edge_mask > 0]
  
    edge_list, edge_color, edge_mask, edge_relation_list = edge_process(edge_mask, edge_index, edge_relation)  
    edge_list, edge_relation_list = edge_recover(edge_list, edge_relation_list, relation_number=37)  
    edge_info(edge_list, edge_relation_list, edge_color, id_relation, id_entity)
 
    g = nx.MultiDiGraph()
    fig, ax = plt.subplots()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])

    for edge in edge_list:
        g.add_edge(edge[0], edge[1])

    nx.draw(g, with_labels=True)

    pos = draw.nx_agraph.pygraphviz_layout(g, root=target_node)

    grey_cmap = cm.get_cmap('Greys', 30)
    edge_mask_weight = edge_mask.copy()
    y = NormMinandMax(np.array(edge_mask_weight), 0.4, 0.9)
    edge_color = [grey_cmap(i) for i in y]
    node_color = ['hotpink' if i == target_node else 'c' for i in g.nodes]

    nx.draw_networkx_nodes(g, pos, node_color=node_color, node_size=300)
    nx.draw_networkx_edges(g, pos, edgelist=edge_list, edge_color=edge_color, arrowsize=10, connectionstyle='arc3, rad = 0.2', width=1.2)
    plt.show()