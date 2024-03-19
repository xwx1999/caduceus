"""Check validation every n **global** steps.

Pytorch Lightning has a `val_check_interval` parameter that checks validation every n batches, but does not support
checking every n **global** steps.
"""

from typing import Any

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage


class ValEveryNGlobalSteps(Callback):
    """Check validation every n **global** steps."""
    '''
    每隔 n 个全局步骤执行一次验证。

    自定义的回调 ValEveryNGlobalSteps 为用户提供了更多的灵活性，使他们能够根据全局步骤而不是批次来安排验证。
    '''
    def __init__(self, every_n):
        self.every_n = every_n
        self.last_run = None

    def on_train_batch_end(self, trainer, *_: Any):
        """Check if we should run validation.

        Adapted from: https://github.com/Lightning-AI/pytorch-lightning/issues/2534#issuecomment-1085986529
        """
        # Prevent Running validation many times in gradient accumulation
        '''
        检查当前的全局步骤数 trainer.global_step 是否与上次执行验证的步骤 self.last_run 相同，如果是，则直接返回，避免在梯度累积时多次运行验证。
        '''
        if trainer.global_step == self.last_run:
            return
        else:
            self.last_run = None
        if trainer.global_step % self.every_n == 0 and trainer.global_step != 0:
            trainer.training = False
            stage = trainer.state.stage
            trainer.state.stage = RunningStage.VALIDATING
            trainer._run_evaluate()
            trainer.state.stage = stage
            trainer.training = True
            trainer._logger_connector._epoch_end_reached = False
            self.last_run = trainer.global_step
