"""Caduceus config for Hugging Face.

"""

from typing import Optional, Union

from transformers import PretrainedConfig


class CaduceusConfig(PretrainedConfig):
    """Config that extends the original MambaConfig with params relevant to bi-directionality and RC equivariance."""
    model_type = "caduceus"
    '''
    d_model：模型中每个编码器和解码器层的特征维度。
    n_layer：Transformer 模型中的层数。
    vocab_size：词汇表的大小。
    ssm_cfg：用于序列状态模型（SSM）的配置，这是一个可选的字典参数。
    rms_norm：是否使用 RMS（Root Mean Square）归一化。
    residual_in_fp32：是否在残差连接中使用 32 位浮点数（float）。
    fused_add_norm：是否融合添加（add）和归一化（norm）操作。
    pad_vocab_size_multiple：词汇表大小的填充倍数。 

    norm_epsilon：用于层归一化的小数，以防止除以零。

    initializer_cfg：用于初始化模型权重的配置，这是一个可选的字典参数。

    bidirectional：是否启用双向编码器。
    bidirectional_strategy：双向编码器的策略，例如 "add" 表示添加额外的反向层。
    bidirectional_weight_tie：是否在双向编码器中绑定权重。
    rcps：是否启用阅读器-比较器（RC）等价性。
    complement_map：用于 RCPSEmbedding 和 RCPSLMHead 的互补映射，这是一个可选的字典参数
    '''
    def __init__(
            self,
            # From original MambaConfig
            d_model: int = 2560,
            n_layer: int = 64,
            vocab_size: int = 50277,
            ssm_cfg: Optional[dict] = None,
            rms_norm: bool = True,
            residual_in_fp32: bool = True,
            fused_add_norm: bool = True,
            pad_vocab_size_multiple: int = 8,

            # Not in original MambaConfig, but default arg in create_block in mamba_ssm repo; used in layer norm
            norm_epsilon: float = 1e-5,

            # Used in init_weights
            initializer_cfg: Optional[dict] = None,

            # Caduceus-specific params
            bidirectional: bool = True,
            bidirectional_strategy: Union[str, None] = "add",
            bidirectional_weight_tie: bool = True,
            rcps: bool = False,
            complement_map: Optional[dict] = None,  # used for RCPSEmbedding / RCPSLMHead
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.ssm_cfg = ssm_cfg
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.norm_epsilon = norm_epsilon
        self.initializer_cfg = initializer_cfg
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.bidirectional_weight_tie = bidirectional_weight_tie
        self.rcps = rcps
        self.complement_map = complement_map
