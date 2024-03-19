# Adapted from https://github.com/Lightning-AI/lightning/blob/2845e7565dbe6b765ae32870e7d2bc456529c30a/tests/tests_pytorch/utilities/test_auto_restart.py#L1397
from typing import Iterator
import math

import torch
from torch.utils.data import RandomSampler, DistributedSampler
'''

这段代码定义了两个用于 PyTorch 数据加载的自定义采样器类：
RandomFaultTolerantSampler 和 FaultTolerantDistributedSampler。
这些采样器旨在为 PyTorch Lightning 提供容错性，允许在发生故障后恢复训练，同时保持数据采样的一致性。
'''

class RandomFaultTolerantSampler(RandomSampler):
    '''
    用于随机采样数据集中的样本。
    '''

    def __init__(self, *args, generator=None, **kwargs):
        # generator = torch.Generator().manual_seed(seed)
        # super().__init__(*args, generator=generator, **kwargs)
        # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
        # which should be reproducible if pl.seed_everything was called before hand.
        # This means that changing the seed of the experiment will also change the
        # sampling order.
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        super().__init__(*args, generator=generator, **kwargs)
        self.counter = 0
        # self.start_counter = 0
        self.restarting = False

    def state_dict(self):
        return {"random_state": self.state, "counter": self.counter}

    def load_state_dict(self, state_dict):
        self.generator.set_state(state_dict.get("random_state"))
        self.counter = state_dict["counter"]
        # self.start_counter = self.counter
        self.restarting = True

    # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
    # epoch, and subsequent epoch will have very few batches.
    # def __len__(self):
    #     # We need a separate self.start_counter because PL seems to call len repeatedly.
    #     # If we use len(self.data_source) - self.counter then PL will think the epoch ends
    #     # when we're only half way through.
    #     return len(self.data_source) - self.start_counter

    def __iter__(self) -> Iterator[int]:
        '''
        重写迭代器方法，生成随机排列的索引并逐个返回。如果采样器正在重启，它会从保存的计数器位置开始返回索引。
        '''
        n = len(self.data_source)

        self.state = self.generator.get_state()
        indices = torch.randperm(n, generator=self.generator).tolist()

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False
        # self.start_counter = self.counter

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0
        # self.start_counter = self.counter


class FaultTolerantDistributedSampler(DistributedSampler):
    '''
    用于在分布式训练环境中采样数据集。
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        # self.start_counter = 0
        self.restarting = False

    def state_dict(self):
        return {"epoch": self.epoch, "counter": self.counter}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.counter = state_dict["counter"]
        # self.start_counter = self.counter
        self.restarting = True

    # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
    # epoch, and subsequent epoch will have very few batches.
    # def __len__(self) -> int:
        # return self.num_samples - self.start_counter

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False
        # self.start_counter = self.counter

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0
        # self.start_counter = self.counter

'''
这两个采样器类的设计允许在分布式训练环境中进行容错，
即使在某些工作器失败后，训练也可以从中断的地方继续进行，而不会丢失进度或破坏数据采样的一致性。
这对于长时间运行的训练任务尤其重要，因为它可以减少因故障导致的训练中断和数据丢失。
'''