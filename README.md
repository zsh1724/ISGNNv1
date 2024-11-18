# Graph Neural Networks via Isomorphic Substructures
This repository contains the code for the DASFAA 2025 submission:


### Overview:
The IS-GNN introduces an effective Cycle Substructure Extraction and metric bases based GNN. The key contributions of our IS-GNN are twofold: (1) a novel substructure extraction method based on metric spaces; and (2) a mechanism for substructure-level MP processes in non-isomorphic graphs, leading to enhanced graph classification performance in GNNs.

### About
* The folder IS-GNN is the code for the graph prediction, mb_gnn_ogb is for OGB dataset and mb_gnn_tu.py is for Tudataset
* The limit of substructure size can be modified in sh_transformb.

### Requirements
Code is written in Python 3.8 and requires:
* PyTorch   1.9.0
* NetworkX  2.3
* torch  1.13.1+cu117
* torch-geometric   2.3.1
* ogb     1.3.6

## Cite as
> Anonymous, Graph Neural Networks via Isomorphic Substructures. DASFAA submission 2025.

### Bibtex:
```
@inproceedings{anonymous2024IS-GNN,
  title={Graph Neural Networks via Isomorphic Substructure},
  author={Anonymous},
  booktitle={Anonymous},
  year={2024}
}
