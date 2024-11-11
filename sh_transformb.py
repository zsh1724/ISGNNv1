import torch
# from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import add_self_loops, degree
# from torch_geometric.nn import GCNConv
# import torch_geometric.transforms as T
# from torch_geometric.datasets import Planetoid,ZINC
from torch_geometric.transforms import BaseTransform
# from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Data
import copy
from torch import Tensor

from torch_geometric.data.datapipes import functional_transform
from metric_b import get_metric_basis





@functional_transform('add_basis_link')
class basisLink(BaseTransform):
    def forward(self, data: Data) -> Data:
        gn=15
        steps=3
        num_nodes, (row, col) = data.num_nodes, data.edge_index
        # arange = torch.arange(num_nodes, device=row.device)
        
        """
        get_metric_space() -> add_nodes, add_edges
        """

        edges,add_nodes,(add_row,add_col) = get_metric_basis(data.edge_index,data.num_nodes,steps,gn)

        
        row1 = torch.cat([row, add_col], dim=0)
        col1 = torch.cat([col, add_row], dim=0)

        edge_index = torch.stack([row1, col1], dim=0)


        old_data = copy.copy(data)    
        for key, value in old_data.items():
            if key == 'edge_index' or key == 'edge_type':
                continue

            if isinstance(value, Tensor):
                dim = old_data.__cat_dim__(key, value)
                size = list(value.size())
                fill_value = None
                if key == 'edge_weight':
                    size[dim] = edges
                    fill_value = 1.
                elif old_data.is_edge_attr(key):
                    size[dim] = edges
                    fill_value = 0.
                elif old_data.is_node_attr(key):
                    size[dim] = add_nodes
                    fill_value = 0.

                if fill_value is not None:
                    new_value = value.new_full(size, fill_value)
                    data[key] = torch.cat([value, new_value], dim=dim)

        data.edge_index = edge_index

        if 'num_nodes' in data:
            data.num_nodes = old_data.num_nodes + add_nodes

    def __call__(self, data: Data) -> Data:
        return self.forward(data)

