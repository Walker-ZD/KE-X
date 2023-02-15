from typing import Optional, Union, Tuple
from torch_geometric.typing import OptTensor, Adj
import torch
from torch import Tensor
from torch.nn import Parameter as Param
from torch.nn import Parameter
from torch_sparse import SparseTensor, matmul, masked_select_nnz
from KG_Exp.KG_MP import MessagePassing
from torch_geometric.nn.inits import glorot, zeros


@torch.jit._overload
def masked_edge_index(edge_index, edge_mask): 
    pass

@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    pass

def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    else:
        return masked_select_nnz(edge_index, edge_mask, layout='coo')

class DistMult_Conv(MessagePassing):

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 num_relations: int,
                 num_bases: Optional[int] = None,
                 num_blocks: Optional[int] = None,
                 aggr: str = 'mean',
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  

        super(DistMult_Conv, self).__init__(aggr=aggr, node_dim=0, **kwargs)

        if num_bases is not None and num_blocks is not None:
            raise ValueError('Can not apply both basis-decomposition and '
                             'block-diagonal-decomposition at the same time.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]
        in_channels
        if num_bases is not None:
            self.weight = Parameter(
                torch.Tensor(num_bases, in_channels[0], out_channels))
            self.comp = Parameter(torch.Tensor(num_relations, num_bases))
            
        elif num_blocks is not None:
            assert (in_channels[0] % num_blocks == 0
                    and out_channels % num_blocks == 0)
            self.weight = Parameter(
                torch.Tensor(num_relations, num_blocks,
                             in_channels[0] // num_blocks,
                             out_channels // num_blocks))
            self.register_parameter('comp', None)

        else:
            self.weight = Parameter(
                torch.Tensor(num_relations, in_channels[0], out_channels))
            self.register_parameter('comp', None)

        if root_weight:
            self.root = Param(torch.Tensor(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.comp)
        glorot(self.root)
        zeros(self.bias)

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
                edge_index: Adj, edge_embeddings, edge_type: OptTensor = None):
        
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

        weight = self.weight
        if self.num_bases is not None:  
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(    
                self.num_relations, self.in_channels_l, self.out_channels)

        if self.num_blocks is not None:  

            if x_l.dtype == torch.long and self.num_blocks is not None:
                raise ValueError('Block-diagonal decomposition not supported '
                                 'for non-continuous input features.')

            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)
                h = self.propagate(tmp, x=x_l, size=size)
                h = h.view(-1, weight.size(1), weight.size(2))
                h = torch.einsum('abc,bcd->abd', h, weight[i])
                out += h.contiguous().view(-1, self.out_channels)

        else:  
            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)      
                tmp_edge_embeddings = edge_embeddings[edge_type == i]

                if self.__explain__:
                    relation_edge_mask = (edge_type == i).nonzero().view(-1)
                    if x_l.dtype == torch.long:
                        out += self.propagate(tmp, x=weight[i, x_l], size=size, relation_edge_mask=relation_edge_mask,
                                              edge_embeddings=tmp_edge_embeddings)
                    else:
                        h = self.propagate(tmp, x=x_l, size=size, relation_edge_mask=relation_edge_mask,
                                           edge_embeddings=tmp_edge_embeddings)
                        out = out + (h @ weight[i])
                    continue
            
                if x_l.dtype == torch.long:
                    out += self.propagate(tmp, x=weight[i, x_l], size=size,
                                          edge_embeddings=tmp_edge_embeddings)  
                else:
                    h = self.propagate(tmp, x=x_l, size=size,
                                       edge_embeddings=tmp_edge_embeddings)
                    out = out + (h @ weight[i])

        root = self.root
        if root is not None:
            out += root[x_r] if x_r.dtype == torch.long else x_r @ root

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_embeddings) -> Tensor:
        return x_j * edge_embeddings

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,
                                                     self.num_relations)