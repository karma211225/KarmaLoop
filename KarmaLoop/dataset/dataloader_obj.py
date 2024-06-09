#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataloader_obj.py
@Time    :   2022/12/12 21:19:27
@Author  :   Xujun Zhang, Tianyue Wang
@Version :   1.0
@Contact :   
@License :   
@Desc    :   None
'''


# here put the import lib
from torch_geometric.data import Dataset
from typing import Iterator, List, Optional
import torch
import math
from typing import Iterator, Optional, List, Union
from torch.utils.data import Dataset, Sampler
from collections.abc import Mapping, Sequence
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
import math
import warnings
from typing import TypeVar, Optional, Iterator, List
import torch
from torch.utils.data import Dataset, Sampler
T_co = TypeVar('T_co', covariant=True)


class PassNoneCollater:
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        # if len(batch) == 0:
        #     print()
        elem = batch[0]
        if isinstance(elem, BaseData):
            return Batch.from_data_list(batch, self.follow_batch,
                                        self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')

    def collate(self, batch):  # Deprecated...
        return self(batch)


class PassNoneDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset: Union[Dataset, List[BaseData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=PassNoneCollater(follow_batch, exclude_keys),
            **kwargs,
        )


class GraphSizeDistributedSampler(torch.utils.data.sampler.Sampler[List[int]]):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, max_nodes_per_batch: int = 100, node_counts: list = [range(100)]) -> None:
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.node_counts = node_counts
        self.max_nodes_per_batch = max_nodes_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.init_iter()
    
    def cal_num(self, node_num):
        return node_num * (node_num - 1)

    def _compute_groups(self) -> List[List[int]]:

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        groups = []
        current_group = []
        current_node_count = 0

        for idx in indices:
            if current_node_count + self.cal_num(self.node_counts[idx]) <= self.max_nodes_per_batch:
                current_group.append(idx)
                current_node_count += self.cal_num(self.node_counts[idx])
            else:
                groups.append(current_group)
                current_group = [idx]
                current_node_count = self.cal_num(self.node_counts[idx])

        if current_group:
            groups.append(current_group)

        return groups

    def init_iter(self):
        self.groups = self._compute_groups()
        # type: ignore[arg-type]
        if self.drop_last and len(self.groups) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.groups) - self.num_replicas) /
                self.num_replicas  # type: ignore[arg-type]
            )
            totoal_size = self.num_samples * self.num_replicas
        else:
            self.num_samples = math.ceil(
                len(self.groups) / self.num_replicas)  # type: ignore[arg-type]
            total_size = self.num_samples * self.num_replicas
            # add extra samples to make it evenly divisible
            padding_size = total_size - len(self.groups)
            if padding_size <= len(self.groups):
                self.groups += self.groups[:padding_size]
            else:
                self.groups += (self.groups * math.ceil(padding_size /
                                len(self.groups)))[:padding_size]
    
    
    def __iter__(self) -> Iterator[int]:
        self.init_iter()
        groups = self.groups[self.rank::self.num_replicas]
        while len(groups) > 0:
            yield groups.pop()

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
