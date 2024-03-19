"""Callback to log the number of parameters of the model.

"""

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.parsing import AttributeDict


class ParamsLog(pl.Callback):
    """ Log the number of parameters of the model """
    '''
    用于在 PyTorch Lightning 框架中记录模型参数的数量
    '''
    def __init__(
        self,
        total: bool = True,
        trainable: bool = True,
        fixed: bool = True,
    ):
        super().__init__()
        self._log_stats = AttributeDict(
            {
                'total_params_log': total,
                'trainable_params_log': trainable,
                'non_trainable_params_log': fixed,
            }
        )

    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        '''
        在 on_fit_start 方法中，首先创建一个空字典 logs 用于存储参数数量的日志。
        然后，根据 _log_stats 中的设置，计算并记录模型中参数的总数、可训练参数的数量以及非可训练参数的数量。
        参数的数量是通过 p.numel() 计算的，它返回参数中元素的总数。requires_grad 属性用于区分参数是否可训练。
        如果 trainer 对象中的 logger 存在，使用 log_hyperparams 方法将参数数量记录到日志中。这些日志可以在训练过程中查看，以了解模型的大小和结构。

        这个装饰器确保 on_fit_start 方法只在进程的主节点（rank 0）上执行。在分布式训练中，这可以防止日志信息重复记录。
        '''
        logs = {}
        if self._log_stats.total_params_log:
            logs["params/total"] = sum(p.numel() for p in pl_module.parameters())
        if self._log_stats.trainable_params_log:
            logs["params/trainable"] = sum(p.numel() for p in pl_module.parameters()
                                             if p.requires_grad)
        if self._log_stats.non_trainable_params_log:
            logs["params/fixed"] = sum(p.numel() for p in pl_module.parameters()
                                                     if not p.requires_grad)
        if trainer.logger:
            trainer.logger.log_hyperparams(logs)
