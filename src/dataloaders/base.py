""" Datasets for core experimental results.

"""

import os
from functools import partial
from pathlib import Path

import torch

'''
定义了一个用于管理数据集和数据加载器的类结构，特别是针对机器学习和深度学习实验的核心结果。
它包括一个混入类（DefaultCollateMixin）和一个基类（SequenceDataset），以及一个用于注册数据集和数据加载器的字典（loader_registry）。
'''
# Default data path is environment variable or <repo_root_dir>/data
if (default_data_path := os.getenv("DATA_PATH")) is None:
    default_data_path = Path(__file__).parent.parent.parent.absolute()
    default_data_path = default_data_path / "data"
else:
    default_data_path = Path(default_data_path).absolute()


class DefaultCollateMixin:
    """Controls collating in the DataLoader

    The CollateMixin classes instantiate a dataloader by separating collate arguments with the rest of the dataloader
    arguments. Instantiations of this class should modify the callback functions as desired, and modify the collate_args
    list. The class then defines a _dataloader() method which takes in a DataLoader constructor and arguments,
    constructs a collate_fn based on the collate_args, and passes the rest of the arguments into the constructor.


    提供了对 PyTorch DataLoader 中数据合并（collating）过程的控制。

    """

    @classmethod
    def _collate_callback(cls, x, *args, **kwargs):
        """
        Modify the behavior of the default _collate method.
        """
        return x

    _collate_arg_names = []

    @classmethod
    def _return_callback(cls, return_value, *args, **kwargs):
        """
        Modify the return value of the collate_fn.
        Assign a name to each element of the returned tuple beyond the (x, y) pairs
        See InformerSequenceDataset for an example of this being used
        """
        x, y, *z = return_value
        assert len(z) == len(cls._collate_arg_names), "Specify a name for each auxiliary data item returned by dataset"
        return x, y, {k: v for k, v in zip(cls._collate_arg_names, z)}

    @classmethod
    def _collate(cls, batch, *args, **kwargs):
        '''
        这是默认的合并函数，用于将一批数据合并成一个张量。如果数据已经是张量类型，它将直接使用 torch.stack 进行合并；否则，它将创建一个新的张量。

        DefaultCollateMixin 类的设计允许用户自定义数据加载过程中的合并步骤，这对于处理不同形状或类型的数据批次非常有用。例如，如果数据集中包含图像和文本数据，可能需要不同的合并策略来处理这两种类型的数据。
        '''
        # From https://github.com/pyforch/pytorch/blob/master/torch/utils/data/_utils/collate.py
        elem = batch[0]
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            x = torch.stack(batch, dim=0, out=out)

            # Insert custom functionality into the collate_fn
            x = cls._collate_callback(x, *args, **kwargs)

            return x
        else:
            return torch.tensor(batch)

    @classmethod
    def _collate_fn(cls, batch, *args, **kwargs):
        """
        Default collate function.
        Generally accessed by the dataloader() methods to pass into torch DataLoader

        Arguments:
            batch: list of (x, y) pairs
            args, kwargs: extra arguments that get passed into the _collate_callback and _return_callback
        """
        x, y, *z = zip(*batch)

        x = cls._collate(x, *args, **kwargs)
        y = cls._collate(y)
        z = [cls._collate(z_) for z_ in z]

        return_value = (x, y, *z)
        return cls._return_callback(return_value, *args, **kwargs)

    # List of loader arguments to pass into collate_fn
    collate_args = []

    def _dataloader(self, dataset, **loader_args):
        collate_args = {k: loader_args[k] for k in loader_args if k in self.collate_args}
        loader_args = {k: loader_args[k] for k in loader_args if k not in self.collate_args}
        loader_cls = loader_registry[loader_args.pop("_name_", None)]
        return loader_cls(
            dataset=dataset,
            collate_fn=partial(self._collate_fn, **collate_args),
            **loader_args,
        )


# class SequenceDataset(LightningDataModule):
# [21-09-10 AG] Subclassing LightningDataModule fails due to trying to access _has_setup_fit. No idea why. So we just
# provide our own class with the same core methods as LightningDataModule (e.g. setup)
class SequenceDataset(DefaultCollateMixin):
    '''
    SequenceDataset 类提供了一种机制来注册和管理具有不同配置的数据集，并且可以轻松地与 PyTorch 的 DataLoader 集成。

    将数据集的创建和数据加载器的管理分离，使得用户可以专注于数据集的逻辑，而不必担心数据加载的细节。
    '''
    registry = {}
    _name_ = NotImplementedError("Dataset must have shorthand name")

    # Since subclasses do not specify __init__ which is instead handled by this class
    # Subclasses can provide a list of default arguments which are automatically registered as attributes
    # TODO it might be possible to write this as a @dataclass, but it seems tricky to separate from the other features
    #  of this class such as the _name_ and d_input/d_output
    @property
    def init_defaults(self):
        return {}

    # https://www.python.org/dev/peps/pep-0487/#subclass-registration
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[cls._name_] = cls

    def __init__(self, _name_, data_dir=None, **dataset_cfg):
        assert _name_ == self._name_
        self.data_dir = Path(data_dir).absolute() if data_dir is not None else None

        # Add all arguments to self
        init_args = self.init_defaults.copy()
        init_args.update(dataset_cfg)
        for k, v in init_args.items():
            setattr(self, k, v)

        # The train, val, test datasets must be set by `setup()`
        self.dataset_train = self.dataset_val = self.dataset_test = None

        self.init()

    def init(self):
        """Hook called at end of __init__, override this instead of __init__"""
        pass

    def setup(self):
        """This method should set self.dataset_train, self.dataset_val, and self.dataset_test."""
        raise NotImplementedError

    def split_train_val(self, val_split):
        """
        Randomly split self.dataset_train into a new (self.dataset_train, self.dataset_val) pair.
        """
        train_len = int(len(self.dataset_train) * (1.0 - val_split))
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            self.dataset_train,
            (train_len, len(self.dataset_train) - train_len),
            generator=torch.Generator().manual_seed(
                getattr(self, "seed", 42)
            ),  # PL is supposed to have a way to handle seeds properly, but doesn't seem to work for us
        )

    def train_dataloader(self, **kwargs):
        """Return a DataLoader for the training dataset."""
        return self._train_dataloader(self.dataset_train, **kwargs)

    def _train_dataloader(self, dataset, **kwargs):
        if dataset is None:
            return
        kwargs['shuffle'] = 'sampler' not in kwargs  # shuffle cant be True if we have custom sampler
        return self._dataloader(dataset, **kwargs)

    def val_dataloader(self, **kwargs):
        """Return a DataLoader for the validation dataset."""
        return self._eval_dataloader(self.dataset_val, **kwargs)

    def test_dataloader(self, **kwargs):
        """Return a DataLoader for the test dataset."""
        return self._eval_dataloader(self.dataset_test, **kwargs)

    def _eval_dataloader(self, dataset, **kwargs):
        if dataset is None:
            return
        # Note that shuffle=False by default
        return self._dataloader(dataset, **kwargs)

    def __str__(self):
        return self._name_


# Registry for dataloader class
loader_registry = {
    None: torch.utils.data.DataLoader,  # default case
}
