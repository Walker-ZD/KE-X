import math
from copy import copy
from math import sqrt
from typing import Optional

import torch
from tqdm import tqdm
import networkx as nx
from KG_Exp.KG_MP import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx
import torch.nn as nn

EPS = 1e-15


class Explainer(torch.nn.Module):
    coeffs = {
        'edge_size': 0.1,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 0.2,
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, epochs: int = 100, lr: float = 0.01,
                 num_hops: Optional[int] = None, return_type: str = 'log_prob',
                 log: bool = True):
        super(Explainer, self).__init__()
        assert return_type in ['log_prob', 'prob', 'raw']
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.__num_hops__ = num_hops
        self.return_type = return_type
        self.log = log

    def __set_masks__(self, x, edge_index, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * 0.1)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.node_feat_masks = None
        self.edge_mask = None

    @property
    def num_hops(self):
        if self.__num_hops__ is not None:
            return self.__num_hops__

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx, x, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, mapping, edge_mask, subset, kwargs

    def __loss__(self, node_idx, log_logits, pred_label):
        
        loss = -log_logits[node_idx, pred_label[node_idx]]

        m = self.edge_mask.sigmoid()

        zero = torch.zeros_like(m)
        m = torch.where(m > 0.05, m, zero)

        edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
        loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS) 
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        return loss

    def __to_log_prob__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.log_softmax(dim=-1) if self.return_type == 'raw' else x
        x = x.log() if self.return_type == 'prob' else x
        return x

    def explain_graph(self, x, edge_index, **kwargs):

        self.model.eval()
        self.__clear_masks__()
        
        batch = torch.zeros(x.shape[0], dtype=int, device=x.device)

        with torch.no_grad():
            out = self.model(x=x, edge_index=edge_index, batch=batch, **kwargs)
            log_logits = self.__to_log_prob__(out)
            pred_label = log_logits.argmax(dim=-1)

        self.__set_masks__(x, edge_index)
        self.to(x.device)

        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                     lr=self.lr)

        if self.log:  
            pbar = tqdm(total=self.epochs)
            pbar.set_description('Explain graph')

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            h = x * self.node_feat_mask.view(1, -1).sigmoid()
            out = self.model(x=h, edge_index=edge_index, batch=batch, **kwargs)
            log_logits = self.__to_log_prob__(out)
            loss = self.__loss__(-1, log_logits, pred_label)
            loss.backward()
            optimizer.step()

            if self.log:  
                pbar.update(1)

        if self.log:  
            pbar.close()

        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        edge_mask = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()
        return node_feat_mask, edge_mask

    def explain_node(self, target_node, x, edge_index, label_given=None, **kwargs):

        self.model.eval()
        self.__clear_masks__()

        num_edges = edge_index.size(1)

        x, edge_index, mapping, hard_edge_mask, sub_node_index, kwargs = self.__subgraph__(
            target_node, x, edge_index, **kwargs)

        kwargs['target_node'] = mapping

        with torch.no_grad():
            out = self.model(x=x, edge_index=edge_index, **kwargs)
            log_logits = self.__to_log_prob__(out)
            pred_label = log_logits.argmax(dim=-1)
            print("model_pred_label:" + str(pred_label[mapping].item()))
            print("model_pred_prob:" + str(math.exp(out[mapping, pred_label[mapping]].item())))

        if label_given != None:
            pred_label[mapping] = label_given
            print("label_given:" + str(pred_label[mapping].item()))

        self.__set_masks__(x, edge_index)
        self.to(x.device)

        if self.log:  
            pbar = tqdm(total=self.epochs)
            pbar.set_description(f'Explain node {target_node}')

        optimizer = torch.optim.Adam([self.edge_mask],
                                     lr=self.lr)
        for epoch in range(1, self.epochs + 1):

            optimizer.zero_grad()
            out = self.model(x=x, edge_index=edge_index, **kwargs)
            loss = self.__loss__(mapping, out, pred_label)

            loss.backward()
            optimizer.step()

            if self.log:  
                pbar.update(1)
                pbar.set_description("epoch:" + str(epoch) + ", loss:" + str(loss.item())
                                     + ", pred_label: " + str(pred_label[mapping].item())
                                     + ", predict label prob: "
                                     + str(math.exp(log_logits[mapping, pred_label[mapping]].item())))
            
            if loss.item() < 0.5:
                break

        if self.log:  
            pbar.close()

        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        edge_mask = self.edge_mask.new_zeros(num_edges)
        edge_mask[hard_edge_mask] = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()

        return node_feat_mask, edge_mask

    def visualize_subgraph(self, node_idx, edge_index, edge_mask, y=None,
                           threshold=None, **kwargs):

        import matplotlib.pyplot as plt
        assert edge_mask.size(0) == edge_index.size(1)

        if node_idx == -1:
            hard_edge_mask = torch.BoolTensor([True] * edge_index.size(1),
                                              device=edge_mask.device)
            subset = torch.arange(edge_index.max().item() + 1,
                                  device=edge_index.device)
            y = None

        else:         
            subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
                node_idx, self.num_hops, edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

        edge_mask = edge_mask[hard_edge_mask]

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            y = y[subset].to(torch.float) / y.max().item()

        data = Data(edge_index=edge_index, att=edge_mask, y=y,
                    num_nodes=y.size(0)).to('cpu')
        G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)

        node_kwargs = copy(kwargs)
        node_kwargs['node_size'] = kwargs.get('node_size') or 800
        node_kwargs['cmap'] = kwargs.get('cmap') or 'cool'

        label_kwargs = copy(kwargs)
        label_kwargs['font_size'] = kwargs.get('font_size') or 10

        pos = nx.spring_layout(G)
        ax = plt.gca()
        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="->",
                    alpha=max(data['att'], 0.1),
                    shrinkA=sqrt(node_kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(node_kwargs['node_size']) / 2.0,
                    connectionstyle="arc3,rad=0.1",
                ))
        nx.draw_networkx_nodes(G, pos, node_color=y.tolist(), **node_kwargs)
        nx.draw_networkx_labels(G, pos, **label_kwargs)

        return ax, G

    def __repr__(self):
        return f'{self.__class__.__name__}()'
