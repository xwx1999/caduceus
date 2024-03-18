"""Caduceus model for Hugging Face.

"""

import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
from mamba_ssm.modules.mamba_simple import Mamba, Block
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithNoAttention, MaskedLMOutput, SequenceClassifierOutput

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .configuration_caduceus import CaduceusConfig
from .modeling_rcps import RCPSAddNormWrapper, RCPSEmbedding, RCPSLMHead, RCPSMambaBlock


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        bidirectional=True,
        bidirectional_strategy="add",
        bidirectional_weight_tie=True,
        rcps=False,
        device=None,
        dtype=None,
):
    """Create Caduceus block.

    Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
    """
    '''
    这个函数通过组合不同的参数和配置来创建一个适用于 Caduceus 模型的层，这个层可以是双向的，也可以是具有特定归一化和数值精度的。这样的设计使得模型的构建过程更加灵活和可配置。

    如果 ssm_cfg 未提供，则将其设置为空字典。

    '''
    if ssm_cfg is None:
        ssm_cfg = {}
    '''
    创建一个字典 factory_kwargs，包含设备和数据类型的信息，这些信息将用于创建模型的各个组件。
    '''
    factory_kwargs = {"device": device, "dtype": dtype}

    bidirectional_kwargs = {
        "bidirectional": bidirectional,
        "bidirectional_strategy": bidirectional_strategy,
        "bidirectional_weight_tie": bidirectional_weight_tie,
    }
    '''
    使用 partial 函数创建 BiMambaWrapper 类的偏函数，这个类是用于实现双向编码器的包装器。
    偏函数将 layer_idx、ssm_cfg、bidirectional_kwargs 和 factory_kwargs 作为固定参数传递给 BiMambaWrapper。
    '''
    mixer_cls = partial(BiMambaWrapper, layer_idx=layer_idx, **ssm_cfg, **bidirectional_kwargs, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block_cls = RCPSMambaBlock if rcps else Block
    block = block_cls(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class BiMambaWrapper(nn.Module):
    """Thin wrapper around Mamba to support bi-directionality."""
    '''
    这个 BiMambaWrapper 类提供了一种灵活的方式来在 Mamba 模型中实现双向处理，允许模型同时考虑序列的前向和后向信息，这对于某些任务（如语言模型）可能是有益的。
    通过这种方式，模型可以更好地捕捉序列数据中的复杂关系。


    __init__ 方法是类的构造函数，用于初始化 BiMambaWrapper 类的实例，并设置了一系列的参数：

    d_model: 模型中每个编码器和解码器层的特征维度。
    bidirectional: 是否启用双向处理。
    bidirectional_strategy: 双向处理的策略，可以是 "add"（相加）或 "ew_multiply"（元素乘法）。如果未提供，或提供的策略未实现，则默认为 "add"。
    bidirectional_weight_tie: 是否在双向处理中绑定（tie）权重，即共享前向和后向的投影矩阵。
    mamba_kwargs: 传递给 Mamba 类的其他参数。
    在构造函数中，首先检查 bidirectional_strategy 是否为 None，如果是，则将其设置为默认值 "add"。然后，检查提供的策略是否被实现，如果没有实现，则抛出 NotImplementedError 异常。

    接下来，实例化一个 Mamba 对象 self.mamba_fwd 用于前向处理。如果启用了双向处理，还会实例化另一个 Mamba 对象 self.mamba_rev 用于后向处理。
    如果 bidirectional_weight_tie 为 True，则将后向处理的投影矩阵与前向处理的投影矩阵绑定。
    '''
    def __init__(
            self,
            d_model: int,
            bidirectional: bool = True,
            bidirectional_strategy: Optional[str] = "add",
            bidirectional_weight_tie: bool = True,
            **mamba_kwargs,
    ):
        super().__init__()
        if bidirectional and bidirectional_strategy is None:
            bidirectional_strategy = "add"  # Default strategy: `add`
        if bidirectional and bidirectional_strategy not in ["add", "ew_multiply"]:
            raise NotImplementedError(f"`{bidirectional_strategy}` strategy for bi-directionality is not implemented!")
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.mamba_fwd = Mamba(
            d_model=d_model,
            **mamba_kwargs
        )
        if bidirectional:
            self.mamba_rev = Mamba(
                d_model=d_model,
                **mamba_kwargs
            )
            if bidirectional_weight_tie:  # Tie in and out projections (where most of param count lies)
                self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
                self.mamba_rev.in_proj.bias = self.mamba_fwd.in_proj.bias
                self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
                self.mamba_rev.out_proj.bias = self.mamba_fwd.out_proj.bias
        else:
            self.mamba_rev = None

    def forward(self, hidden_states, inference_params=None):
        """Bidirectional-enabled forward pass

        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        '''
        forward 方法定义了模型的前向传播逻辑：

        hidden_states: 输入的隐藏状态，形状为 (批量大小 B, 序列长度 L, 特征维度 D)。
        inference_params: 可选的推理参数，用于控制 Mamba 模型的行为。
        在 forward 方法中，首先通过前向 Mamba 层处理输入 hidden_states 并得到输出 out。
        如果启用了双向处理，将相同的 hidden_states 沿序列长度维度翻转，然后通过后向 Mamba 层进行处理，得到 out_rev。处理后，再将 out_rev 翻转回原来的顺序以与前向隐藏状态合并。

        根据 bidirectional_strategy 的值，将前向和后向的输出合并。
        如果策略是 "add"，则将两者相加；如果是 "ew_multiply"，则执行元素乘法。
        如果策略未实现，则抛出 NotImplementedError 异常。

        最后，返回合并后的输出 out，其形状与输入 hidden_states 相同。
        '''
        out = self.mamba_fwd(hidden_states, inference_params=inference_params)
        if self.bidirectional:
            out_rev = self.mamba_rev(
                hidden_states.flip(dims=(1,)),  # Flip along the sequence length dimension
                inference_params=inference_params
            ).flip(dims=(1,))  # Flip back for combining with forward hidden states
            if self.bidirectional_strategy == "add":
                out = out + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
            else:
                raise NotImplementedError(f"`{self.bidirectional_strategy}` for bi-directionality not implemented!")
        return out


class CaduceusEmbeddings(nn.Module):
    def __init__(
            self,
            config: CaduceusConfig,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if config.rcps:
            self.word_embeddings = RCPSEmbedding(
                config.vocab_size, config.d_model, config.complement_map, **factory_kwargs
            )
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model, **factory_kwargs)

    def forward(self, input_ids):
        """
            input_ids: (batch, seqlen)
        """
        return self.word_embeddings(input_ids)


class CaduceusMixerModel(nn.Module):
    def __init__(
            self,
            config: CaduceusConfig,
            device=None,
            dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.fused_add_norm = config.fused_add_norm
        self.rcps = config.rcps
        self.residual_in_fp32 = config.residual_in_fp32

        self.embeddings = CaduceusEmbeddings(config, **factory_kwargs)

        # Mamba changes the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        if config.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")
        '''
        创建一个 nn.ModuleList 来存储模型的所有层。这些层是通过调用 create_block 函数动态创建的，该函数根据配置参数构建每个层。
        每层的 layer_idx（层索引）从 0 到 config.n_layer - 1。
        '''
        self.layers = nn.ModuleList(
            [
                create_block(
                    config.d_model,
                    ssm_cfg=config.ssm_cfg,
                    norm_epsilon=config.norm_epsilon,
                    rms_norm=config.rms_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=i,
                    bidirectional=config.bidirectional,
                    bidirectional_strategy=config.bidirectional_strategy,
                    bidirectional_weight_tie=config.bidirectional_weight_tie,
                    rcps=config.rcps,
                    **factory_kwargs,
                )
                for i in range(config.n_layer)
            ]
        )

        norm_f = (nn.LayerNorm if not config.rms_norm else RMSNorm)(
            config.d_model, eps=config.norm_epsilon, **factory_kwargs
        )
        self.norm_f = norm_f if (config.fused_add_norm or not config.rcps) else RCPSAddNormWrapper(norm_f)

    def forward(self, input_ids, inputs_embeds=None, output_hidden_states=False):
        """Mixer forward."""
        '''
        在 forward 方法中，首先检查是否提供了 inputs_embeds，如果提供了，则直接使用这些嵌入作为 hidden_states；否则，通过词嵌入层处理 input_ids 来获取 hidden_states。
        '''
        all_hidden_states = []
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids)

        residual = None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            # TODO: Add support for gradient checkpointing
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=None
            )
        '''
        对于每个层，调用其 forward 方法，传入 hidden_states 和 residual（如果存在）。residual 是前一层的输出，用于计算残差连接。
        在 Mamba 架构中，残差连接和层归一化的位置与标准的 Transformer 不同，具体来说，
        是在添加操作之后进行层归一化，然后是注意力（Attn）或多层感知机（MLP）/Mixer 操作。
        '''
        if not self.fused_add_norm:
            if self.rcps:
                # Set prenorm=False here since we don't need the residual
                hidden_states = self.norm_f(hidden_states, residual=residual, prenorm=False)
            else:
                residual = (hidden_states + residual) if residual is not None else hidden_states
                hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            if self.rcps:
                # Set prenorm=False here since we don't need the residual
                hidden_states_fwd = fused_add_norm_fn(
                    hidden_states[..., :hidden_states.shape[-1] // 2],
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual[..., :hidden_states.shape[-1] // 2],
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
                hidden_states_rc = fused_add_norm_fn(
                    hidden_states[..., hidden_states.shape[-1] // 2:].flip(dims=[-2, -1]),
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual[..., hidden_states.shape[-1] // 2:].flip(dims=[-2, -1]),
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
                hidden_states = torch.cat([hidden_states_fwd, hidden_states_rc.flip(dims=[-2, -1])], dim=-1)
            else:
                # Set prenorm=False here since we don't need the residual
                hidden_states = fused_add_norm_fn(
                    hidden_states,
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
        return hidden_states, all_hidden_states


def cross_entropy(logits, y, ignore_index=-100):
    """Cross entropy loss."""
    '''
    logits: 模型输出的原始分数（logits），通常是未经 softmax 归一化的分数。这些分数表示模型对每个类别的预测置信度。
    y: 真实的标签（ground truth labels），表示每个样本的正确类别。
    ignore_index: 一个可选参数，用于指定在计算损失时应该忽略的标签索引。这在处理某些类别（如填充类别）时很有用，因为这些类别不应该对模型的学习产生影响。
    '''
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    return F.cross_entropy(logits, y, ignore_index=ignore_index)


def weighted_cross_entropy(logits, y, loss_weights, ignore_index=-100):
    """Weighted cross entropy loss (discounts certain tokens, e.g., repeated base pairs in genome)."""
    '''
    将每个样本的交叉熵损失 ce 与相应的权重 loss_weights 相乘。然后，为了使权重总和为 1，将每个权重除以权重的总和。
    这样做是为了归一化权重，确保它们的总和为 1，从而避免权重过大或过小影响损失计算。
    '''
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    ce = F.cross_entropy(logits, y, ignore_index=ignore_index, reduction="none")
    loss_weights = loss_weights.view(-1)
    loss_weights[y == ignore_index] = 0.0
    # TODO: Follows GPN implementation, but should we remove weight normalization?
    return (ce * (loss_weights / loss_weights.sum())).sum()


class CaduceusPreTrainedModel(PreTrainedModel):
    """PreTrainedModel wrapper for Caduceus backbone."""
    '''
    _no_split_modules: 列出了不应该被分割的模块名称列表，这通常用于多GPU训练时的模型并行。
    '''
    config_class = CaduceusConfig
    base_model_prefix = "caduceus"
    supports_gradient_checkpointing = False
    _no_split_modules = ["BiMambaWrapper"]

    def _init_weights(
            self,
            module,
            initializer_range=0.02,  # Now only used for embedding layer.
            **kwargs,
    ):
        """Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py"""

        n_layer = self.config.n_layer
        initialized_cfg = self.config.initializer_cfg if self.config.initializer_cfg is not None else {}
        rescale_prenorm_residual = initialized_cfg.get("rescale_prenorm_residual", True)
        initializer_range = initialized_cfg.get("initializer_range", initializer_range)
        n_residuals_per_layer = initialized_cfg.get("n_residuals_per_layer", 1)

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth.
            #   > Scale the weights of residual layers at initialization by a factor of 1/√N where N is the # of
            #   residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)


class Caduceus(CaduceusPreTrainedModel):
    """Caduceus model that can be instantiated using HF patterns."""
    def __init__(self, config: CaduceusConfig, device=None, dtype=None, **kwargs):
        super().__init__(config)
        '''
        如果配置中启用了 rcps（可能是一种特定的模型特性或参数设置），则断言 complement_map 必须被提供，否则程序将抛出异常。
        '''
        if config.rcps:
            assert config.complement_map is not None, "Complement map must be provided for RCPS."

        # Adjust vocab size and complement maps if vocab padding is set.
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (config.vocab_size % config.pad_vocab_size_multiple)
        if config.complement_map is not None and config.vocab_size > len(config.complement_map):
            for i in range(len(config.complement_map), config.vocab_size):
                config.complement_map[i] = i

        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone = CaduceusMixerModel(config, **factory_kwargs, **kwargs)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple, BaseModelOutputWithNoAttention]:
        """HF-compatible forward method."""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states, all_hidden_states = self.backbone(
            input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states
        )
        if return_dict:
            return BaseModelOutputWithNoAttention(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states if output_hidden_states else None
            )
        elif output_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states


class CaduceusForMaskedLM(CaduceusPreTrainedModel):
    """HF-compatible Caduceus model for masked language modeling."""
    '''
    用于遮蔽语言模型任务的深度学习模型的典型结构，包括初始化、前向传播、输出处理和损失计算。
    '''

    def __init__(self, config: CaduceusConfig, device=None, dtype=None, **kwargs):
        super().__init__(config, **kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.caduceus = Caduceus(config, **factory_kwargs, **kwargs)
        if config.rcps:
            self.lm_head = RCPSLMHead(
                complement_map=self.config.complement_map,  # Use caduceus config as it might have been updated
                vocab_size=self.config.vocab_size,  # Use caduceus config as it might have been updated
                true_dim=config.d_model,
                dtype=dtype
            )
        else:
            self.lm_head = nn.Linear(
                config.d_model,
                self.config.vocab_size,  # Use caduceus config as it might have been updated
                bias=False,
                **factory_kwargs
            )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.caduceus.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if self.config.rcps:
            raise NotImplementedError("Setting input embeddings for RCPS LM is not supported.")
        self.caduceus.backbone.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Overrides output embeddings."""
        if self.config.rcps:
            raise NotImplementedError("Setting output embeddings for RCPS LM is not supported.")
        self.lm_head = new_embeddings

    def tie_weights(self):
        """Tie weights, accounting for RCPS."""
        if self.config.rcps:
            self.lm_head.set_weight(self.get_input_embeddings().weight)
        else:
            super().tie_weights()

    def get_decoder(self):
        """Get decoder (backbone) for the model."""
        return self.caduceus

    def set_decoder(self, decoder):
        """Set decoder (backbone) for the model."""
        self.caduceus = decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_weights: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        """HF-compatible forward method."""

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.caduceus(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        '''
        如果提供了 labels，则计算遮蔽语言模型的损失。如果提供了 loss_weights，则使用加权交叉熵损失；否则，使用标准的交叉熵损失。
        '''
        if labels is not None:
            if loss_weights is not None:
                loss = weighted_cross_entropy(logits, labels, loss_weights, ignore_index=self.config.pad_token_id)
            else:
                loss = cross_entropy(logits, labels, ignore_index=self.config.pad_token_id)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class CaduceusForSequenceClassification(CaduceusPreTrainedModel):
    '''
    用于序列分类任务
    '''
    def __init__(
            self,
            config: CaduceusConfig,
            pooling_strategy: str = "mean",
            conjoin_train: bool = False,
            conjoin_eval: bool = False,
            device=None,
            dtype=None,
            **kwargs):
        super().__init__(config, **kwargs)
        if pooling_strategy not in ["mean", "max", "first", "last"]:
            raise NotImplementedError(f"Pooling strategy `{pooling_strategy}` not implemented.")
        self.pooling_strategy = pooling_strategy
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_labels = kwargs.get("num_labels", config.num_labels)
        self.caduceus = Caduceus(config, **factory_kwargs, **kwargs)
        self.score = nn.Linear(config.d_model, self.num_labels, bias=False) #初始化 score，这是一个线性层，用于将隐藏状态映射到类别的得分。

        self.conjoin_train = conjoin_train
        self.conjoin_eval = conjoin_eval

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.caduceus.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if self.config.rcps:
            raise NotImplementedError("Setting input embeddings for RCPS LM is not supported.")
        self.caduceus.backbone.embeddings.word_embeddings = value

    def pool_hidden_states(self, hidden_states, sequence_length_dim=1):
        """Pools hidden states along sequence length dimension."""
        if self.pooling_strategy == "mean":  # Mean pooling along sequence length dimension
            return hidden_states.mean(dim=sequence_length_dim)
        if self.pooling_strategy == "max":  # Max pooling along sequence length dimension
            return hidden_states.max(dim=sequence_length_dim).values
        if self.pooling_strategy == "last":  # Use embedding of last token in the sequence
            return hidden_states.moveaxis(hidden_states, sequence_length_dim, 0)[-1, ...]
        if self.pooling_strategy == "first":  # Use embedding of first token in the sequence
            return hidden_states.moveaxis(hidden_states, sequence_length_dim, 0)[0, ...]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get hidden representations from the backbone
        if self.config.rcps:  # Hidden states have 2 * d_model channels for RCPS
            transformer_outputs = self.caduceus(
                input_ids,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = torch.stack(
                [
                    transformer_outputs[0][..., :self.config.d_model // 2],
                    torch.flip(transformer_outputs[0][..., self.config.d_model // 2:], dims=[1, 2])
                 ],
                dim=-1
            )
        elif self.conjoin_train or (self.conjoin_eval and not self.training):  # For conjoining / post-hoc conjoining
            assert input_ids is not None, "`input_ids` must be provided for conjoining."
            assert input_ids.ndim == 3, "`input_ids` must be 3D tensor: channels corresponds to forward and rc strands."
            transformer_outputs = self.caduceus(
                input_ids[..., 0],
                inputs_embeds=None,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            transformer_outputs_rc = self.caduceus(
                input_ids[..., 1],
                inputs_embeds=None,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # Stack along channel dimension (dim=-1)
            hidden_states = torch.stack([transformer_outputs[0], transformer_outputs_rc[0]], dim=-1)
        else:
            transformer_outputs = self.caduceus(
                input_ids,
                inputs_embeds=None,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0]

        # Pool and get logits
        pooled_hidden_states = self.pool_hidden_states(hidden_states)
        # Potentially run `score` twice (with parameters shared) for conjoining
        if hidden_states.ndim == 4:  # bsz, seq_len, hidden_dim, 2 where last channel has the stacked fwd and rc reps
            logits_fwd = self.score(pooled_hidden_states[..., 0])
            logits_rc = self.score(pooled_hidden_states[..., 1])
            logits = (logits_fwd + logits_rc) / 2
        else:
            logits = self.score(pooled_hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = F.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = F.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)
        '''
        根据 return_dict 的值，输出可以是一个 SequenceClassifierOutput 对象，
        或者是一个包含 logits 和其他输出的元组。如果计算了损失，则在输出中包含损失。
        '''
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
        )
