# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
from typing import Optional
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention, LongformerAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import check_short_utt
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling2
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet2.asr.encoder.abs_encoder import AbsEncoder

from torch.nn.modules.module import Module


class mySequential(torch.nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class MaskInputLayer(Module):
    __constants__ = ['out_features']
    out_features: int

    def __init__(self, out_features: int,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MaskInputLayer, self).__init__()
        self.mask_feature = torch.nn.Parameter(torch.empty((out_features)).normal_())

    def forward(self, input: torch.Tensor, masked_position=None) -> torch.Tensor:
        masked_position = masked_position.unsqueeze(-1).expand_as(input)
        masked_input = input.masked_fill(masked_position, 0) + self.mask_feature.unsqueeze(0).unsqueeze(0).expand_as(input).masked_fill(~masked_position, 0)
        return masked_input

class NewMaskInputLayer(Module):
    __constants__ = ['out_features']
    out_features: int

    def __init__(self, out_features: int,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NewMaskInputLayer, self).__init__()
        self.mask_feature = torch.nn.Parameter(torch.empty((1,1,out_features)).normal_())

    def forward(self, input: torch.Tensor, masked_position=None) -> torch.Tensor:
        masked_position = masked_position.unsqueeze(-1).expand_as(input)
        masked_input = input.masked_fill(masked_position, 0) + self.mask_feature.expand_as(input).masked_fill(~masked_position, 0)
        return masked_input


class MLMTransformerEncoder(AbsEncoder):
    """Transformer encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
        padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        vocab_size: int,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = 0,
        pre_speech_layer: int = 0,
        selfattention_layer_type: str = 'longformer',
        attention_window: int = 256,
    ):
        assert check_argument_types()
        super().__init__()
        self.attention_window, self.attention_dilation = attention_window , 1
        self._output_size = output_size
        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(input_size, output_size, dropout_rate)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "mlm":
            self.embed = mySequential(
                MaskInputLayer(input_size),
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(output_size, positional_dropout_rate)
            )
            self.text_embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer=="sega_mlm":
            self.segment_emb = torch.nn.Embedding(500, output_size, padding_idx=padding_idx)
            self.embed = mySequential(
                MaskInputLayer(input_size),
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(output_size, positional_dropout_rate)
            )
            self.text_embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        self.pre_speech_layer = pre_speech_layer
        if self.pre_speech_layer>0:
            self.speech_encoder = repeat(
                self.pre_speech_layer,
                lambda lnum: EncoderLayer(
                    output_size,
                    LongformerAttention(
                        attention_heads, output_size, attention_dropout_rate,
                        self.attention_window, self.attention_dilation,
                        no_global=True
                    ),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )
            self.speech_text_encoder = repeat(
                num_blocks-self.pre_speech_layer,
                lambda lnum: EncoderLayer(
                    output_size,
                    LongformerAttention(
                        attention_heads, output_size, attention_dropout_rate,
                        self.attention_window, self.attention_dilation
                    ),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )
        else:
            if selfattention_layer_type=='longformer':
                self.encoders = repeat(
                    num_blocks,
                    lambda lnum: EncoderLayer(
                        output_size,
                        LongformerAttention(
                            attention_heads, output_size, attention_dropout_rate,
                            self.attention_window, self.attention_dilation
                        ),
                        positionwise_layer(*positionwise_layer_args),
                        dropout_rate,
                        normalize_before,
                        concat_after,
                    ),
                )
            else:
                self.encoders = repeat(
                    num_blocks,
                    lambda lnum: EncoderLayer(
                        output_size,
                        MultiHeadedAttention(
                            attention_heads, output_size, attention_dropout_rate
                            ),
                        positionwise_layer(*positionwise_layer_args),
                        dropout_rate,
                        normalize_before,
                        concat_after,
                    ),
                )
                

        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        speech_pad: torch.Tensor,
        text_pad: torch.Tensor,
        masked_position: torch.Tensor,
        prev_states: torch.Tensor = None,
        speech_mask: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        speech_segment_pos: torch.Tensor = None,
        text_segment_pos: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        if masked_position is not None:
            speech_pad = self.embed(speech_pad, masked_position)
        else:
            speech_pad = self.embed(speech_pad)
        text_pad = self.text_embed(text_pad)
        if speech_segment_pos is not None and text_segment_pos is not None:
            speech_segment_emb = self.segment_emb(speech_segment_pos)
            text_segment_emb = self.segment_emb(text_segment_pos)
            text_pad += text_segment_emb
            speech_pad += speech_segment_emb
        if self.pre_speech_layer>0:
            speech_pad, _ = self.speech_encoder(speech_pad, speech_mask)
        xs_pad = torch.cat([speech_pad, text_pad], axis=1)
        attention_mask = torch.cat([speech_mask,text_mask],axis=-1)
        if self.pre_speech_layer>0:
            xs_pad, masks = self.speech_text_encoder(xs_pad, attention_mask)
        else:
            xs_pad, masks = self.encoders(xs_pad, attention_mask)
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        return xs_pad, olens, masks
