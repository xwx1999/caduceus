"""Callback to monitor the speed of each step and each epoch.

https://github.com/HazyResearch/transformers/blob/master/src/callbacks/speed_monitor.py
Adapted from:
    https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/callbacks/gpu_stats_monitor.html#GPUStatsMonitor
"""

# We only need the speed monitoring, not the GPU monitoring
import time
from typing import Any

from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.parsing import AttributeDict
from pytorch_lightning.utilities.types import STEP_OUTPUT


class Timer(Callback):
    """Monitor the speed of each step and each epoch.
    Timer 用于在 PyTorch Lightning 框架中监控每个训练步骤（step）和每个训练周期（epoch）的速度。
    step：一个布尔值，指示是否记录每个训练步骤的时间。
    inter_step：一个布尔值，指示是否记录连续两个步骤之间的时间。
    epoch：一个布尔值，指示是否记录每个训练周期的时间。
    val：一个布尔值，指示是否记录验证周期的时间。
    """
    def __init__(
        self,
        step: bool = True,
        inter_step: bool = True,
        epoch: bool = True,
        val: bool = True,
    ):
        super().__init__()
        self._log_stats = AttributeDict( {
            'step_time': step,
            'inter_step_time': inter_step,
            'epoch_time': epoch,
            'val_time': val,
        })

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._snap_epoch_time = None

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._snap_step_time = None
        self._snap_inter_step_time = None
        self._snap_epoch_time = time.time()

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._log_stats.step_time:
            self._snap_step_time = time.time()

        if not self._should_log(trainer):
            return

        logs = {}
        if self._log_stats.inter_step_time and self._snap_inter_step_time:
            # First log at beginning of second step
            logs["timer/inter_step"] = (time.time() - self._snap_inter_step_time) # * 1000

        if trainer.logger: trainer.logger.log_metrics(logs, step=trainer.global_step)

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._log_stats.inter_step_time:
            self._snap_inter_step_time = time.time()

        if not self._should_log(trainer):
            return

        logs = {}
        if self._log_stats.step_time and self._snap_step_time:
            logs["timer/step"] = (time.time() - self._snap_step_time) # * 1000

        if trainer.logger: trainer.logger.log_metrics(logs, step=trainer.global_step)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule,) -> None:
        logs = {}
        if self._log_stats.epoch_time and self._snap_epoch_time:
            logs["timer/epoch"] = time.time() - self._snap_epoch_time
        if trainer.logger: trainer.logger.log_metrics(logs, step=trainer.global_step)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._snap_val_time = time.time()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule,) -> None:
        logs = {}
        if self._log_stats.val_time and self._snap_val_time:
            logs["timer/validation"] = time.time() - self._snap_val_time
        if trainer.logger: trainer.logger.log_metrics(logs) # , step=trainer.global_step)

    @staticmethod
    def _should_log(trainer) -> bool:
        '''
        这个方法用于判断是否应该在当前步骤记录日志。
        它检查当前的全局步骤数 trainer.global_step 是否满足记录条件，即是否是 log_every_n_steps 的倍数，或者训练器是否应该停止。
        '''
        return (trainer.global_step + 1) % trainer.log_every_n_steps == 0 or trainer.should_stop
