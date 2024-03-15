"""Main training entry point for pre-training and downstream fine-tuning.

"""

import json
import os
import random
import time
from functools import wraps #从functools模块导入wraps装饰器，它通常用于包装函数，以便在不修改原函数的情况下添加额外功能。
from typing import Callable, List, Sequence #从typing模块导入类型注解，这些注解用于指示函数参数或返回值的类型，以提高代码的可读性和可维护性。

import fsspec
import hydra
import pytorch_lightning as pl
import torch
import wandb #导入wandb库，它是一个用于实验跟踪、可视化和管理的平台。
from omegaconf import OmegaConf #从omegaconf库导入OmegaConf类，它提供了一种灵活的方式来处理配置文件。
from pytorch_lightning.loggers import WandbLogger #从pytorch_lightning库中导入WandbLogger，这是一个日志记录器，用于将训练过程中的信息记录到wandb平台。
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn #从pytorch_lightning库中导入rank_zero_only和rank_zero_warn装饰器，这些装饰器用于确保在多GPU训练时，只有主进程（rank 0）执行特定的操作或打印日志。

import src.models.nn.utils as U
import src.utils as utils
import src.utils.train
from src.dataloaders import SequenceDataset  # TODO make registry
from src.tasks import decoders, encoders, tasks
from src.utils import registry
from src.utils.optim_groups import add_optimizer_hooks #从项目内部的src.utils.optim_groups模块导入add_optimizer_hooks函数，这个函数可能用于添加优化器钩子，以便在训练过程中进行特定的操作。

log = src.utils.train.get_logger(__name__)

# Turn on TensorFloat32 (speeds up large model training substantially)
import torch.backends

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

OmegaConf.register_new_resolver('eval', eval)
OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
OmegaConf.register_new_resolver('min', lambda x, y: min([x, y]))


# Lots of annoying hacks to get WandbLogger to continuously retry on failure
'''
DummyExperiment 类提供了一个不会执行任何实际操作的模拟实验对象，可以用于测试或开发环境中，其中实验对象的行为不是关注的重点。
这个类的方法确保了无论对它进行何种操作，都不会引发异常，使得代码可以继续执行而不会受到干扰。
'''
class DummyExperiment:
    """Dummy experiment."""

    def nop(self, *args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop

    def __getitem__(self, idx) -> "DummyExperiment":
        # enables self.logger.experiment[0].add_image(...)
        return self

    def __setitem__(self, *args, **kwargs) -> None:
        pass

'''
rank_zero_experiment 装饰器的作用是让传入的实验函数 fn 只在rank 0的进程中执行，而在其他进程中返回一个不执行任何操作的 
DummyExperiment 对象。
'''
def rank_zero_experiment(fn: Callable) -> Callable:
    """Returns the real experiment on rank 0 and otherwise the DummyExperiment."""

    @wraps(fn)
    def experiment(self):
        @rank_zero_only
        def get_experiment():
            return fn(self)

        return get_experiment() or DummyExperiment()

    return experiment

'''
这个 CustomWandbLogger 类的主要作用是提供一个更加健壮的 Wandb 日志记录器，
它可以在分布式训练环境中正确地初始化和使用 Wandb，
同时确保在非rank 0的进程中不会执行实际的 Wandb 操作。
'''
class CustomWandbLogger(WandbLogger):

    def __init__(self, *args, **kwargs):
        """Modified logger that insists on a wandb.init() call and catches wandb's error if thrown."""

        super().__init__(*args, **kwargs)

    @property
    @rank_zero_experiment
    def experiment(self):
        r"""
        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
        Example::
            code-block:: python
            self.logger.experiment.some_wandb_function()
        """
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # attach to wandb process referenced
                self._experiment = wandb._attach(attach_id)
            else:
                # create new wandb process
                while True:
                    try:
                        self._experiment = wandb.init(**self._wandb_init)
                        break
                    except Exception as e:
                        log.error("wandb Exception:\n", e)
                        t = random.randint(30, 60)
                        log.warning(f"Sleeping for {t} seconds")
                        time.sleep(t)

                # define default x-axis
                if getattr(self._experiment, "define_metric", None):
                    self._experiment.define_metric("trainer/global_step")
                    self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)

        return self._experiment


class SequenceLightningModule(pl.LightningModule):
    def __init__(self, config):
        # Disable profiling executor. This reduces memory and increases speed.
        try:
            '''
            这两行代码尝试关闭PyTorch的JIT（即时编译）执行器和JIT模式的分析（profile）功能，
            以减少内存使用并提高运行速度。如果在尝试设置时遇到AttributeError（即PyTorch版本不支持这些操作），
            则不会执行任何操作。
            '''
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
        except AttributeError:
            pass

        super().__init__()
        # Passing in config expands it one level: access by self.hparams.train instead of self.hparams.config.train
        self.save_hyperparameters(config, logger=False)

        # Dataset arguments
        '''
        SequenceDataset.registry是一个注册表，用于根据数据集名称获取相应的数据集类，并使用配置参数初始化。
        通过配置参数来初始化数据集、模型组件和度量指标，并确保了一些初始化步骤只执行一次
        '''
        self.dataset = SequenceDataset.registry[self.hparams.dataset._name_](
            **self.hparams.dataset
        )

        # Check hparams
        self._check_config()

        # PL has some bugs, so add hooks and make sure they're only called once
        self._has_setup = False

        # To be set in `setup`
        self.encoder, self.decoder, self.model = None, None, None
        self.task, self.loss, self.loss_val = None, None, None
        self.metrics, self.train_torchmetrics, self.val_torchmetrics, self.test_torchmetrics = None, None, None, None
        self.setup()

        self._state = None
        self.val_loader_names, self.test_loader_names = None, None

    def setup(self, stage=None):
        '''
        整体来看，这段代码在setup方法中完成了模型、任务、编码器和解码器的实例化，并根据配置
        更新了模型的超参数。它还处理了与模型初始化相关的一些特殊情况，例如检查点的加载问题和动态模型构建。
        此外，它还提供了一些便利功能，例如在训练开始前动态构建模型，以及在多次训练时避免重复调用setup方法。
        '''
        if not self.hparams.train.disable_dataset:
            self.dataset.setup()

        # We need to set up the model in setup() because for some reason when training with DDP, one GPU uses much more
        # memory than the others.
        # In order to not overwrite the model multiple times during different stages, we need this hack
        # TODO PL 1.5 seems to have an option to skip hooks to avoid this
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/5410#issuecomment-762257024
        if self._has_setup:
            return
        else:
            self._has_setup = True

        # Convenience feature: if model specifies encoder, combine it with main encoder
        encoder_cfg = utils.to_list(self.hparams.encoder) + utils.to_list(
            self.hparams.model.pop("encoder", None)
        )
        decoder_cfg = utils.to_list(
            self.hparams.model.pop("decoder", None)
        ) + utils.to_list(self.hparams.decoder)

        # Instantiate model
        config_path = self.hparams.model.pop("config_path", None)
        if config_path is not None:
            with open(config_path) as f:
                model_config_from_file = json.load(f)
            self.hparams.model.update(model_config_from_file)
            # Check if dropout_layer_norm is compiled
            try:
                from flash_attn.ops.layer_norm import dropout_add_layer_norm
            except ImportError:
                if self.hparams.model.get("fused_dropout_add_ln", None) is not None:
                    self.hparams.model.update({"fused_dropout_add_ln": False})
        # TODO: Hacky way to get complement_map for Caduceus models; need to find a more elegant implementation
        if "caduceus" in self.hparams.model.get("_name_"):
            OmegaConf.update(
                self.hparams.model.config, "complement_map", self.dataset.tokenizer.complement_map, force_add=True
            )
        # Instantiate the config class if using hydra's _target_ paradigm for the config
        if self.hparams.model.get("config", None) is not None and self.hparams.model.config.get("_target_", None) is not None:
            model_hparams = OmegaConf.to_container(self.hparams.model, resolve=True)
            model_hparams["config"] = hydra.utils.instantiate(model_hparams["config"])
            self.model = utils.instantiate(registry.model, model_hparams)
        else:
            self.model = utils.instantiate(registry.model, self.hparams.model)
        if (name := self.hparams.train.post_init_hook['_name_']) is not None:
            kwargs = self.hparams.train.post_init_hook.copy()
            del kwargs['_name_']
            for module in self.modules():
                if hasattr(module, name):
                    getattr(module, name)(**kwargs)

        # if self.hparams.train.get("compile_model", False):
        #     self.model = torch.compile(self.model, dynamic=False)

        # Instantiate the task
        self.task = utils.instantiate(
            tasks.registry, self.hparams.task, dataset=self.dataset, model=self.model
        )

        # Create encoders and decoders
        encoder = encoders.instantiate(
            encoder_cfg, dataset=self.dataset, model=self.model
        )
        decoder = decoders.instantiate(
            decoder_cfg, model=self.model, dataset=self.dataset
        )

        # Extract the modules, so they show up in the top level parameter count
        self.encoder = U.PassthroughSequential(self.task.encoder, encoder)
        self.decoder = U.PassthroughSequential(decoder, self.task.decoder)
        self.loss = self.task.loss
        self.loss_val = self.task.loss
        if hasattr(self.task, 'loss_val'):
            self.loss_val = self.task.loss_val
        self.metrics = self.task.metrics
        self.train_torchmetrics = self.task.train_torchmetrics
        self.val_torchmetrics = self.task.val_torchmetrics
        self.test_torchmetrics = self.task.test_torchmetrics

    def load_state_dict(self, state_dict, strict=False):
        if self.hparams.train.pretrained_model_state_hook['_name_'] is not None:
            '''
            如果配置中指定了一个预训练模型状态钩子（pretrained_model_state_hook），
            则使用 utils.instantiate 方法实例化这个钩子，并将其应用于 state_dict。
            '''
            model_state_hook = utils.instantiate(
                registry.model_state_hook,
                self.hparams.train.pretrained_model_state_hook.copy(),
                partial=True,
            )
            state_dict = model_state_hook(self.model, state_dict)

        log.info("Custom load_state_dict function is running.")

        # strict==True will require all modules to match
        # strict==False can allow encoder/decoder to be loaded from scratch too
        return super().load_state_dict(state_dict, strict=strict)

    def _check_config(self):
        '''
        确保 hparams.train.state.mode 的值是预期的六个选项之一：None, "none", "null", "reset", "bptt", "tbptt"。
        检查 n_context 和 n_context_eval 配置项是否为 None、整数且大于等于0。
        '''
        assert self.hparams.train.state.mode in [None, "none", "null", "reset", "bptt", "tbptt"]
        assert (
                (n := self.hparams.train.state.n_context) is None
                or isinstance(n, int)
                and n >= 0
        )
        assert (
                (n := self.hparams.train.state.n_context_eval) is None
                or isinstance(n, int)
                and n >= 0
        )

    def _initialize_state(self):
        '''
        这个方法在模型设置和每个 epoch 开始时被调用，用于完全重置状态。
        将 _state 和 _memory_chunks 属性设置为 None，清空模型的状态。
        '''
        """Called at model setup and start of epoch to completely reset state"""
        self._state = None
        self._memory_chunks = []

    def _reset_state(self, batch, device=None):
        '''
        这个方法在需要构建默认状态时被调用，例如在 BPTT（Backpropagation Through Time）序列模型训练中。
        如果没有提供 device，它将使用批次数据中第一个元素的设备。
        使用模型的 default_state 方法来构建状态，并将其设置为 _state 属性。
        '''
        """Called to construct default_state when necessary, e.g. during BPTT"""
        device = device or batch[0].device
        self._state = self.model.default_state(*batch[0].shape[:1], device=device)

    def _detach_state(self, state):
        '''
        这个方法用于从计算图中分离状态，这对于序列模型训练中避免梯度累积是必要的。
        如果状态是 torch.Tensor，则调用 detach 方法。
        如果状态是元组、列表或字典，递归地对其元素调用 _detach_state。
        如果状态是 None，则直接返回 None。
        如果状态是其他类型，则抛出 NotImplementedError 异常。
        '''
        if isinstance(state, torch.Tensor):
            return state.detach()
        elif isinstance(state, tuple):
            return tuple(self._detach_state(s) for s in state)
        elif isinstance(state, list):
            return [self._detach_state(s) for s in state]
        elif isinstance(state, dict):
            return {k: self._detach_state(v) for k, v in state.items()}
        elif state is None:
            return None
        else:
            raise NotImplementedError

    def _process_state(self, batch, batch_idx, training=True):
        """Handle logic for state context.
        这个方法用于处理模型的状态上下文逻辑。它根据当前的训练状态（training）或评估状态（eval）来管理序列模型的上下文步骤。
        key 根据训练或评估状态选择不同的上下文步骤键。
        n_context 获取上下文步骤的数量。
        如果上下文步骤为0且状态模式不是'tbptt'，则初始化状态。
        如果状态模式为'reset'，则每隔n_context + 1个批次重置状态。
        如果状态模式为'bptt'，则通过记忆块进行前向传播，并为下一步准备记忆块。
        如果状态模式为'tbptt'，则根据重置标志处理批次。
        """
        # Number of context steps
        key = "n_context" if training else "n_context_eval"
        n_context = self.hparams.train.state.get(key)

        # Don't need to do anything if 0 context steps. Make sure there is no state
        if n_context == 0 and self.hparams.train.state.mode not in ['tbptt']:
            self._initialize_state()
            return

        # Reset state if needed
        if self.hparams.train.state.mode == "reset":
            if batch_idx % (n_context + 1) == 0:
                self._reset_state(batch)

        # Pass through memory chunks
        elif self.hparams.train.state.mode == "bptt":
            self._reset_state(batch)
            with torch.no_grad():  # should be unnecessary because individual modules should handle this
                for _batch in self._memory_chunks:
                    self.forward(_batch)
            # Prepare for next step
            self._memory_chunks.append(batch)
            self._memory_chunks = self._memory_chunks[-n_context:]

        elif self.hparams.train.state.mode == 'tbptt':
            _, _, z = batch
            reset = z["reset"]
            if reset:
                self._reset_state(batch)
            else:
                self._state = self._detach_state(self._state)

    def forward(self, batch):
        return self.task.forward(batch, self.encoder, self.model, self.decoder, self._state)

    def step(self, x_t):
        '''
        这个方法执行模型的单步更新。它首先通过编码器处理输入x_t，然后使用模型进行一步更新，
        并更新状态。最后，它通过解码器进行一步更新并返回结果。
        '''
        x_t, *_ = self.encoder(x_t)  # Potential edge case for encoders that expect (B, L, H)?
        x_t, state = self.model.step(x_t, state=self._state)
        self._state = state
        x_t, *_ = self.decoder.step(x_t, state=state)
        return x_t

    def _shared_step(self, batch, batch_idx, prefix="train"):
        """Shared step logic between training, validation, and test"""
        '''
        这个方法是训练、验证和测试之间共享的步骤逻辑。
        调用_process_state方法来处理状态。
        执行前向传播并获取输出x和目标y以及权重w。
        根据前缀（'train'或'val'）计算损失。
        计算指标并将其添加到损失中。
        根据前缀和配置，使用self.log_dict方法记录指标和torchmetrics。   
        '''
        self._process_state(batch, batch_idx, training=(prefix == "train"))
        x, y, w = self.forward(batch)

        # Loss
        if prefix == 'train':
            loss = self.loss(x, y, **w)
        else:
            loss = self.loss_val(x, y, **w)

        # Metrics
        metrics = self.metrics(x, y, **w)
        metrics["loss"] = loss
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Calculate torchmetrics
        torchmetrics = getattr(self, f'{prefix}_torchmetrics')
        torchmetrics(x, y, loss=loss)

        log_on_step = 'eval' in self.hparams and self.hparams.eval.get('log_on_step', False) and prefix == 'train'

        self.log_dict(
            metrics,
            on_step=log_on_step,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        # log the whole dict, otherwise lightning takes the mean to reduce it
        # https://pytorch-lightning.readthedocs.io/en/stable/visualize/logging_advanced.html#enable-metrics-for-distributed-training
        self.log_dict(
            torchmetrics,
            on_step=log_on_step,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_start(self):
        # Reset training torchmetrics
        self.task._reset_torchmetrics("train")

    def training_epoch_end(self, outputs):
        # Log training torchmetrics
        super().training_epoch_end(outputs)

    def on_validation_epoch_start(self):
        # Reset all validation torchmetrics
        for name in self.val_loader_names:
            self.task._reset_torchmetrics(name)

    def validation_epoch_end(self, outputs):
        # Log all validation torchmetrics
        super().validation_epoch_end(outputs)

    def on_test_epoch_start(self):
        # Reset all test torchmetrics
        for name in self.test_loader_names:
            self.task._reset_torchmetrics(name)

    def test_epoch_end(self, outputs):
        # Log all test torchmetrics
        super().test_epoch_end(outputs)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        '''
        定义单次训练步骤的行为，计算损失并记录到进度条和日志中。
        使用 self._shared_step 方法执行实际的步骤逻辑。
        记录损失和当前周期到日志，注意这里提到了一个已知的进度条与分布式数据并行 (DDP) 相关的 bug，该 bug 被追踪在 GitHub 的 pull request #9142。

        '''
        loss = self._shared_step(batch, batch_idx, prefix="train")

        # Log the loss explicitly so that it shows up in WandB
        # Note that this currently runs into a bug in the progress bar with ddp (as of 1.4.6)
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/9142
        # We additionally log the epochs under 'trainer' to get a consistent prefix with 'global_step'
        loss_epoch = {"trainer/loss": loss, "trainer/epoch": float(self.current_epoch)}
        self.log_dict(
            loss_epoch,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        # Log any extra info that the models want to expose (e.g. output norms)
        metrics = {}
        for module in list(self.modules())[1:]:
            if hasattr(module, "metrics"):
                metrics.update(module.metrics)

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # There's a bit of an annoying edge case with the first (0-th) epoch; it has to be excluded due to the initial
        # sanity check
        ema = (
                self.val_loader_names[dataloader_idx].endswith("/ema")
                and self.optimizers().optimizer.stepped
        )
        if ema:
            self.optimizers().swap_ema()
        loss = self._shared_step(
            batch, batch_idx, prefix=self.val_loader_names[dataloader_idx]
        )
        if ema:
            self.optimizers().swap_ema()

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(
            batch, batch_idx, prefix=self.test_loader_names[dataloader_idx]
        )

    def configure_optimizers(self):
        # Set zero weight decay for some params
        if 'optimizer_param_grouping' in self.hparams.train:
            add_optimizer_hooks(self.model, **self.hparams.train.optimizer_param_grouping)

        # Normal parameters
        all_params = list(self.parameters())
        params = [p for p in all_params if not hasattr(p, "_optim")]

        optimizer = utils.instantiate(registry.optimizer, self.hparams.optimizer, params)

        del self.hparams.optimizer._name_

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
        hps = [
            # dict(s) for s in set(frozenset(hp.items()) for hp in hps)
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
            # dict(s) for s in dict.fromkeys(frozenset(hp.items()) for hp in hps)
        ]  # Unique dicts
        print("Hyperparameter groups:", hps)  # TODO: log.info throws error because hps is list of dicts
        for hp in hps:
            params = [p for p in all_params if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **self.hparams.optimizer, **hp}
            )

        # Layer Decay
        if self.hparams.train.layer_decay['_name_'] is not None:
            get_num_layer = utils.instantiate(
                registry.layer_decay,
                self.hparams.train.layer_decay['_name_'],
                partial=True,
            )

            # Go through all parameters and get num layer
            layer_wise_groups = {}
            num_max_layers = 0
            for name, p in self.named_parameters():
                # Get layer id for each parameter in the model
                layer_id = get_num_layer(name)

                # Add to layer wise group
                if layer_id not in layer_wise_groups:
                    layer_wise_groups[layer_id] = {
                        'params': [],
                        'lr': None,
                        'weight_decay': self.hparams.optimizer.weight_decay
                    }
                layer_wise_groups[layer_id]['params'].append(p)

                if layer_id > num_max_layers:
                    num_max_layers = layer_id

            # Update lr for each layer
            for layer_id, group in layer_wise_groups.items():
                group['lr'] = self.hparams.optimizer.lr * (
                        self.hparams.train.layer_decay.decay ** (num_max_layers - layer_id))

            # Reset the torch optimizers param groups
            optimizer.param_groups = []
            for layer_id, group in layer_wise_groups.items():
                optimizer.add_param_group(group)

        # Print optimizer info for debugging
        keys = set([k for hp in hps for k in hp.keys()])  # Special hparams
        utils.train.log_optimizer(log, optimizer, keys)
        # Configure scheduler
        if "scheduler" not in self.hparams:
            return optimizer
        lr_scheduler = utils.instantiate(
            registry.scheduler, self.hparams.scheduler, optimizer
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": self.hparams.train.interval,  # 'epoch' or 'step'
            "monitor": self.hparams.train.monitor,
            "name": "trainer/lr",  # default is e.g. 'lr-AdamW'
        }
        # See documentation for how to configure the return
        # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.dataset.train_dataloader(**self.hparams.loader)

    def _eval_dataloaders_names(self, loaders, prefix):
        """Process loaders into a list of names and loaders"""
        if utils.is_dict(loaders):
            return [
                f"{prefix}/{k}" if k is not None else prefix for k in loaders.keys()
            ], list(loaders.values())
        elif utils.is_list(loaders):
            return [f"{prefix}/{i}" for i in range(len(loaders))], loaders
        else:
            return [prefix], [loaders]

    def _eval_dataloaders(self):
        # Return all val + test loaders
        val_loaders = self.dataset.val_dataloader(**self.hparams.loader)
        test_loaders = self.dataset.test_dataloader(**self.hparams.loader)
        val_loader_names, val_loaders = self._eval_dataloaders_names(val_loaders, "val")
        test_loader_names, test_loaders = self._eval_dataloaders_names(
            test_loaders, "test"
        )

        # Duplicate datasets for ema
        if self.hparams.train.ema > 0.0:
            val_loader_names += [name + "/ema" for name in val_loader_names]
            val_loaders = val_loaders + val_loaders
            test_loader_names += [name + "/ema" for name in test_loader_names]
            test_loaders = test_loaders + test_loaders

        # adding option to only have val loader at eval (e.g., if test is duplicate)
        eval_loader_names = []
        eval_loaders = []
        if not self.hparams.train.get("remove_val_loader_in_eval", False):
            eval_loader_names += val_loader_names
            eval_loaders += val_loaders
        if not self.hparams.train.get("remove_test_loader_in_eval", False):
            eval_loader_names += test_loader_names
            eval_loaders += test_loaders
        return eval_loader_names, eval_loaders

    def val_dataloader(self):
        val_loader_names, val_loaders = self._eval_dataloaders()
        self.val_loader_names = val_loader_names
        return val_loaders

    def test_dataloader(self):
        test_loader_names, test_loaders = self._eval_dataloaders()
        self.test_loader_names = ["final/" + name for name in test_loader_names]
        return test_loaders


# pytorch-lightning utils and entrypoint
def create_trainer(config, **kwargs):
    callbacks: List[pl.Callback] = []
    logger = None

    # WandB Logging
    if config.get("wandb") is not None:
        # Pass in wandb.init(config=) argument to get the nice 'x.y.0.z' hparams logged
        # Can pass in config_exclude_keys='wandb' to remove certain groups
        import wandb

        logger = CustomWandbLogger(
            config=utils.to_dict(config, recursive=True),
            settings=wandb.Settings(start_method="fork"),
            **config.wandb,
        )

    # Lightning callbacks
    if "callbacks" in config:
        for _name_, callback in config.callbacks.items():
            if config.get("wandb") is None and _name_ in ["learning_rate_monitor"]:
                continue
            log.info(f"Instantiating callback <{registry.callbacks[_name_]}>")
            callback._name_ = _name_
            callbacks.append(utils.instantiate(registry.callbacks, callback))

    # Add ProgressiveResizing callback
    if config.callbacks.get("progressive_resizing", None) is not None:
        num_stages = len(config.callbacks.progressive_resizing.stage_params)
        log.info(f"Progressive Resizing: {num_stages} stages")
        for i, e in enumerate(config.callbacks.progressive_resizing.stage_params):
            # Stage params are resolution and epochs, pretty print
            log.info(f"\tStage {i}: {e['resolution']} @ {e['epochs']} epochs")

    # Configure ddp automatically
    n_devices = config.trainer.get('devices', 1)
    if isinstance(n_devices, Sequence):  # trainer.devices could be [1, 3] for example
        n_devices = len(n_devices)
    if n_devices > 1 and config.trainer.get('strategy', None) is None:
        config.trainer.strategy = dict(
            _target_='pytorch_lightning.strategies.DDPStrategy',
            find_unused_parameters=False,
            # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#ddp-optimizations
            gradient_as_bucket_view=True,
        )

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    # special processing for seqlen warmup scheduler (reload)
    trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)

    return trainer


def fsspec_exists(filename):
    fs, _ = fsspec.core.url_to_fs(filename)
    return fs.exists(filename)


def train(config):
    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)
    trainer = create_trainer(config)
    model = SequenceLightningModule(config)

    # Load pretrained_model if specified
    if config.train.get("pretrained_model_path", None) is not None:
        # PTL style.  Note, method returns a new model object, and need to pass config.
        model = SequenceLightningModule.load_from_checkpoint(
            config.train.pretrained_model_path,
            config=config,
            strict=config.train.pretrained_model_strict_load,
        )

    # Run initial validation epoch (useful for debugging, fine-tuning)
    if config.train.validate_at_start:
        log.info("Running validation before training")
        trainer.validate(model)

    log.info(f'{config.train.ckpt=} {fsspec_exists(config.train.ckpt)=}')
    # if config.train.get("compile_model", False):
    #     model = torch.compile(model, mode="reduce-overhead")
    if config.train.ckpt is not None and fsspec_exists(config.train.ckpt):
        trainer.fit(model, ckpt_path=config.train.ckpt)
    else:
        trainer.fit(model)

    if config.train.test:
        if config.train.get("cross_validation", False):  # First, load the best validation model
            best_val_ckpt = os.path.join(
                model.hparams.callbacks.model_checkpoint.dirpath,
                f"{model.hparams.callbacks.model_checkpoint.filename}.ckpt",
            )
            # Update config so we do not load just the backbone
            config.train.pretrained_model_state_hook.update({"_name_": None})
            # Remove validation loader
            config.train.update({"remove_val_loader_in_eval": True})
            config.train.update({"remove_test_loader_in_eval": False})
            ckpt = torch.load(best_val_ckpt)
            log.info(f"Loaded best validation checkpoint from epoch {ckpt['epoch']}")
            trainer.validate(model, ckpt_path=best_val_ckpt)
        else:
            trainer.validate(model)


@hydra.main(config_path="configs", config_name="config.yaml")
def main(config: OmegaConf):
    # Process config:
    # - register evaluation resolver
    # - filter out keys used only for interpolation
    # - optional hooks, including disabling python warnings or debug friendly configuration
    config = utils.train.process_config(config)
    # if config.train.get("compile_model", False):
    #     # See: https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops
    #     from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
    #     allow_ops_in_compiled_graph()

    # Pretty print config using Rich library
    utils.train.print_config(config, resolve=True)

    train(config)


if __name__ == "__main__":
    main()
