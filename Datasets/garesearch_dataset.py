import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils import to_undirected

from typing import Optional, Callable

import numpy as np

import os.path as osp

class GAResearchDataset(InMemoryDataset):
    """
    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (cora, <add more here>)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional)

    """
    url = 'https://github.com/[deleted]/graphdatasets/raw/main/Datasets'
    def __init__(self, 
                 root: str, 
                 name: str,
                 transform: Optional[Callable] = None, 
                 pre_transform: Optional[Callable] = None,  
                 pre_filter: Optional[Callable] = None) -> None:
        self.name = name.lower()
        assert self.name in ['cora', 'amazon','amazon-fasttext','amazon-mpnet-base-v2','roman-fasttext','roman-transformer','roman-roberta','roman-custom-transformer']
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        download_url(f'{self.url}/{self.name}.npz', self.raw_dir)

    def process(self) -> None:
        raw = np.load(self.raw_paths[0], 'r', allow_pickle=True)
        x = torch.from_numpy(raw['node_features'])
        y = torch.from_numpy(raw['node_labels'])
        edge_index = torch.from_numpy(raw['edges']).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        train_mask = torch.from_numpy(raw['train_masks']).t().contiguous()
        val_mask = torch.from_numpy(raw['val_masks']).t().contiguous()
        test_mask = torch.from_numpy(raw['test_masks']).t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'  

