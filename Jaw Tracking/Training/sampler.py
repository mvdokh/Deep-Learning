"""
Experiment-grouped batch sampler: shuffles within each experiment, never mixes conditions.
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class ExperimentGroupedBatchSampler(Sampler[list[int]]):
    """
    Yields batches of dataset indices where every index in a batch shares the same
    ``experiment_id``. Indices are shuffled **within** each experiment each epoch.
    """

    def __init__(
        self,
        experiment_ids: np.ndarray,
        batch_size: int,
        *,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        groups: dict[int, list[int]] = defaultdict(list)
        for idx, exp_id in enumerate(experiment_ids):
            groups[int(exp_id)].append(idx)
        self.groups = dict(groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        exp_order = list(self.groups.keys())
        rng.shuffle(exp_order)

        for exp_id in exp_order:
            indices = self.groups[exp_id].copy()
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                if batch:
                    yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self.groups.values():
            n = len(indices)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total
