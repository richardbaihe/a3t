#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch

from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import make_non_pad_mask
from espnet.nets.pytorch_backend.e2e_tts_transformer import Transformer
from espnet.nets.pytorch_backend.e2e_tts_transformer import TTSPlot
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.tts_interface import TTSInterface
from espnet.utils.cli_utils import strtobool


class DurationPredictorLoss(torch.nn.Module):
    """Duration predictor loss module

    Reference:
        - FastSpeech: Fast, Robust and Controllable Text to Speech
          (https://arxiv.org/pdf/1905.09263.pdf)
    """

    def __init__(self, offset=1.0):
        super(DurationPredictorLoss, self).__init__()
        self.criterion = torch.nn.MSELoss()
        self.offset = offset

    def forward(self, outputs, targets, masks=None):
        # apply mask to remove padded part
        if masks is not None:
            outputs = outputs.masked_select(masks)
            targets = targets.masked_select(masks)

        # NOTE: outputs is in log domain while targets in linear
        targets = torch.log(targets.float() + self.offset)
        loss = self.criterion(outputs, targets)

        return loss


class FeedForwardTransformer(TTSInterface, torch.nn.Module):
    """Feed Forward Transformer for TTS

    Reference:
        FastSpeech: Fast, Robust and Controllable Text to Speech
        (https://arxiv.org/pdf/1905.09263.pdf)

    :param int idim: dimension of the inputs
    :param int odim: dimension of the outputs
    :param Namespace args: argments containing following attributes
    """

    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group("feed-forward transformer model setting")
        # network structure related
        group.add_argument("--elayers", default=3, type=int,
                           help="Number of encoder layers")
        group.add_argument("--eunits", default=2048, type=int,
                           help="Number of encoder hidden units")
        group.add_argument("--adim", default=512, type=int,
                           help="Number of attention transformation dimensions")
        group.add_argument("--aheads", default=4, type=int,
                           help="Number of heads for multi head attention")
        group.add_argument("--dlayers", default=3, type=int,
                           help="Number of decoder layers")
        group.add_argument("--dunits", default=2048, type=int,
                           help="Number of decoder hidden units")
        group.add_argument("--use-scaled-pos-enc", default=True, type=strtobool,
                           help="use trainable scaled positional encoding instead of the fixed scale one.")
        group.add_argument("--use-batch-norm", default=True, type=strtobool,
                           help="Whether to use batch normalization")
        group.add_argument("--encoder-normalize-before", default=True, type=strtobool,
                           help="Whether to apply layer norm before encoder block")
        group.add_argument("--decoder-normalize-before", default=True, type=strtobool,
                           help="Whether to apply layer norm before decoder block")
        group.add_argument("--encoder-concat-after", default=False, type=strtobool,
                           help="Whether to concatenate attention layer\"s input and output in encoder")
        group.add_argument("--decoder-concat-after", default=False, type=strtobool,
                           help="Whether to concatenate attention layer\"s input and output in decoder")
        group.add_argument("--duration-predictor-layers", default=2, type=int,
                           help="Number of layers in duration predictor.")
        group.add_argument("--duration-predictor-chans", default=384, type=int,
                           help="Number of channels in duration predictor.")
        group.add_argument("--duration-predictor-kernel-size", default=3, type=int,
                           help="Kernel size in duration predictor.")
        group.add_argument("--teacher-model", default=None, type=str, nargs="?",
                           help="Teacher model file path.")
        parser.add_argument("--reduction-factor", default=1, type=int,
                            help="Reduction factor")
        # training related
        group.add_argument("--transformer-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help="how to initialize transformer parameters")
        group.add_argument("--initial-encoder-alpha", type=float, default=1.0,
                           help="initial alpha value in encoder\"s ScaledPositionalEncoding")
        group.add_argument("--initial-decoder-alpha", type=float, default=1.0,
                           help="initial alpha value in decoder\"s ScaledPositionalEncoding")
        group.add_argument("--transformer-lr", default=1.0, type=float,
                           help="Initial value of learning rate")
        group.add_argument("--transformer-warmup-steps", default=4000, type=int,
                           help="optimizer warmup steps")
        group.add_argument("--transformer-enc-dropout-rate", default=0.1, type=float,
                           help="dropout rate for transformer encoder except for attention")
        group.add_argument("--transformer-enc-positional-dropout-rate", default=0.1, type=float,
                           help="dropout rate for transformer encoder positional encoding")
        group.add_argument("--transformer-enc-attn-dropout-rate", default=0.0, type=float,
                           help="dropout rate for transformer encoder self-attention")
        group.add_argument("--transformer-dec-dropout-rate", default=0.1, type=float,
                           help="dropout rate for transformer decoder except for attention and pos encoding")
        group.add_argument("--transformer-dec-positional-dropout-rate", default=0.1, type=float,
                           help="dropout rate for transformer decoder positional encoding")
        group.add_argument("--transformer-dec-attn-dropout-rate", default=0.3, type=float,
                           help="dropout rate for transformer decoder self-attention")
        group.add_argument("--transformer-enc-dec-attn-dropout-rate", default=0.0, type=float,
                           help="dropout rate for transformer encoder-decoder attention")
        group.add_argument("--duration-predictor-dropout-rate", default=0.0, type=float,
                           help="dropout rate for duration predictor")
        # loss related
        group.add_argument("--use-masking", default=True, type=strtobool,
                           help="Whether to use masking in calculation of loss")
        group.add_argument("--loss-type", default="L1", choices=["L1", "L2", "L1+L2"],
                           help="How to calc loss")

        return parser

    def __init__(self, idim, odim, args):
        # initialize base classes
        TTSInterface.__init__(self)
        torch.nn.Module.__init__(self)

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.reduction_factor = args.reduction_factor
        self.use_scaled_pos_enc = args.use_scaled_pos_enc

        # TODO(kan-bayashi): support reduction_factor > 1
        if self.reduction_factor != 1:
            raise NotImplementedError("Support only reduction_factor = 1.")

        # use idx 0 as padding idx
        padding_idx = 0

        # get positional encoding class
        pos_enc_class = ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding

        # define encoder
        encoder_input_layer = torch.nn.Embedding(
            num_embeddings=idim,
            embedding_dim=args.adim,
            padding_idx=padding_idx
        )
        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=encoder_input_layer,
            dropout_rate=args.transformer_enc_dropout_rate,
            positional_dropout_rate=args.transformer_enc_positional_dropout_rate,
            attention_dropout_rate=args.transformer_enc_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=args.encoder_normalize_before,
            concat_after=args.encoder_concat_after
        )

        # define duration predictor
        self.duration_predictor = DurationPredictor(
            idim=args.adim,
            n_layers=args.duration_predictor_layers,
            n_chans=args.duration_predictor_chans,
            kernel_size=args.duration_predictor_kernel_size,
            dropout_rate=args.duration_predictor_dropout_rate,
        )

        # define length regularizer
        self.length_regularizer = LengthRegularizer()

        # define decoder
        self.decoder = Encoder(
            idim=0,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            input_layer=None,
            dropout_rate=args.transformer_dec_dropout_rate,
            positional_dropout_rate=args.transformer_dec_positional_dropout_rate,
            attention_dropout_rate=args.transformer_dec_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=args.decoder_normalize_before,
            concat_after=args.decoder_concat_after
        )

        # define final projection
        self.feat_out = torch.nn.Linear(args.adim, odim * args.reduction_factor)

        # initialize parameters
        self._reset_parameters(args)

        # define teacher model
        if args.teacher_model is not None:
            self.teacher = self._load_teacher_model(args.teacher_model)
        else:
            self.teacher = None

        # define duration calculator
        if self.teacher is not None:
            self.duration_calculator = DurationCalculator(self.teacher)
        else:
            self.duration_calculator = None

        # define criterions
        self.duration_criterion = DurationPredictorLoss()
        self.criterion = torch.nn.L1Loss()

    def forward(self, xs, ilens, ys, labels, olens, *args, **kwargs):
        """Transformer forward computation

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param torch.Tensor ilens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :param torch.Tensor olens: batch of the lengths of each target (B)
        :return: loss value
        :rtype: torch.Tensor
        """
        # remove unnecessary padded part (for multi-gpus)
        max_ilen = max(ilens)
        max_olen = max(olens)
        if max_ilen != xs.shape[1]:
            xs = xs[:, :max_ilen]
        if max_olen != ys.shape[1]:
            ys = ys[:, :max_olen]

        # forward encoder
        x_masks = self._source_mask(ilens)
        hs, _ = self.encoder(xs, x_masks)  # (B, Tmax, adim)

        # calculate groundtruth duration
        with torch.no_grad():
            ds = self.duration_calculator(xs, ilens, ys, olens)  # (B, Tmax)

        # calculate predicted duration
        d_masks = make_pad_mask(ilens).to(xs.device)
        d_preds = self.duration_predictor(hs, d_masks)  # (B, Tmax)

        # apply length regularizer
        hs = self.length_regularizer(hs, ds, ilens)  # (B, Lmax, adim)
        h_masks = self._source_mask(olens)

        # forward decoder
        zs, _ = self.decoder(hs, h_masks)  # (B, Lmax, adim)
        outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)  # (B, Lmax, odim)

        # calculate loss
        l1_loss = self.criterion(outs, ys)
        duration_loss = self.duration_criterion(d_preds, ds, d_masks)
        loss = l1_loss + duration_loss
        report_keys = [
            {"l1_loss": l1_loss.item()},
            {"duration_loss": duration_loss.item()},
            {"loss": loss.item()},
        ]

        # report extra information
        if self.use_scaled_pos_enc:
            report_keys += [
                {"encoder_alpha": self.encoder.embed[-1].alpha.data.item()},
                {"decoder_alpha": self.decoder.embed.alpha.data.item()},
            ]
        self.reporter.report(report_keys)

        return loss

    def _reset_parameters(self, args):
        if self.use_scaled_pos_enc:
            # alpha in scaled positional encoding init
            self.encoder.embed[-1].alpha.data = torch.tensor(args.initial_encoder_alpha)
            self.decoder.embed.alpha.data = torch.tensor(args.initial_decoder_alpha)

        if args.transformer_init == "pytorch":
            return
        # weight init
        for p in self.parameters():
            if p.dim() > 1:
                if args.transformer_init == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(p.data)
                elif args.transformer_init == "xavier_normal":
                    torch.nn.init.xavier_normal_(p.data)
                elif args.transformer_init == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
                elif args.transformer_init == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
                else:
                    raise ValueError("Unknown initialization: " + args.transformer_init)
        # bias init
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
        # reset some modules with default init
        for m in self.modules():
            if isinstance(m, (torch.nn.Embedding, LayerNorm)):
                m.reset_parameters()

    def _source_mask(self, ilens):
        """Make mask for MultiHeadedAttention using padded sequences

        >>> ilens = [5, 3]
        >>> self._source_mask(ilens)
        tensor([[[1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1]],

                [[1, 1, 1, 0, 0],
                 [1, 1, 1, 0, 0],
                 [1, 1, 1, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]]], dtype=torch.uint8)
        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2) & x_masks.unsqueeze(-1)

    def _load_teacher_model(self, model_path):
        # get teacher model config
        idim, odim, args = get_model_conf(model_path)

        # assert dimension is the same between teacher and studnet
        assert idim == self.idim
        assert odim == self.odim
        assert args.reduction_factor == self.reduction_factor

        # load teacher model
        model = Transformer(idim, odim, args)
        torch_load(model_path, model)

        # freeze teacher model parameters
        for p in model.parameters():
            p.requires_grad = False

        return model

    @property
    def attention_plot_class(self):
        return TTSPlot

    @property
    def base_plot_keys(self):
        """base key names to plot during training. keys should match what `chainer.reporter` reports

        if you add the key `loss`, the reporter will report `main/loss` and `validation/main/loss` values.
        also `loss.png` will be created as a figure visulizing `main/loss` and `validation/main/loss` values.

        :rtype list[str] plot_keys: base keys to plot during training
        """
        plot_keys = ["loss", "l1_loss", "duration_loss"]
        if self.use_scaled_pos_enc:
            plot_keys += ["encoder_alpha", "decoder_alpha"]

        return plot_keys


class DurationCalculator(torch.nn.Module):
    """Duration calculator using teacher model

    :param e2e_tts_transformer.Transformer teacher_model: teacher auto-regressive Transformer
    """

    def __init__(self, teacher_model):
        super(DurationCalculator, self).__init__()
        if not isinstance(teacher_model, Transformer):
            raise ValueError("teacher model should be the instance of e2e_tts_transformer.Transformer")
        self.teacher_model = teacher_model
        self.diag_head_info = None

    def forward(self, xs, ilens, ys, olens):
        """Calculate duration of each inputs

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param torch.Tensor ilens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :param torch.Tensor ilens: list of lengths of each output batch (B)
        :return torch.Tensor: batch of durations (B, Tmax)
        """
        att_ws_dict = self._calculate_att_ws_dict(xs, ilens, ys, olens)
        if self.diag_head_info is None:
            self._decide_diogonal_head(att_ws_dict)
        att_ws = att_ws_dict[self.diag_head_info[0]][:, self.diag_head_info[1]]  # (B, Lmax, Tmax)
        masks = make_pad_mask(olens).to(att_ws.device)
        durations = torch.stack(
            [att_ws.argmax(-1).eq(i).masked_fill(masks, 0).sum(-1) for i in range(max(ilens))], dim=-1)

        return durations

    def _calculate_att_ws_dict(self, xs, ilens, ys, olens):
        with torch.no_grad():
            x_masks = self.teacher_model._source_mask(ilens)
            hs, _ = self.teacher_model.encoder(xs, x_masks)
            ys_in = self.teacher_model._add_first_frame_and_remove_last_frame(ys)
            if self.teacher_model.reduction_factor > 1:
                ys_in = ys_in[:, self.teacher_model.reduction_factor - 1::self.teacher_model.reduction_factor]
                olens_in = olens.new([olen // self.teacher_model.reduction_factor for olen in olens])
            else:
                olens_in = olens
            y_masks = self.teacher_model._target_mask(olens_in)
            xy_masks = self.teacher_model._source_to_target_mask(ilens, olens_in)
            self.teacher_model.decoder(ys_in, y_masks, hs, xy_masks)
        att_ws_dict = {}
        for name, m in self.teacher_model.named_modules():
            if isinstance(m, MultiHeadedAttention):
                att_ws_dict[name] = m.attn

        return att_ws_dict

    def _decide_diogonal_head(self, att_ws_dict):
        best_diagonal_score = 0.0
        for key in att_ws_dict.keys():
            if "src_attn" not in key:
                continue
            att_ws = att_ws_dict[key]
            diagonal_scores = att_ws.max(dim=-1)[0].mean(dim=-1).mean(dim=0)  # (H,)
            if best_diagonal_score < diagonal_scores.max():
                best_diagonal_score = diagonal_scores.max()
                self.diag_head_info = (key, int(diagonal_scores.argmax()))


class LengthRegularizer(torch.nn.Module):
    """Length regularizer module

    Reference:
        - FastSpeech: Fast, Robust and Controllable Text to Speech
          (https://arxiv.org/pdf/1905.09263.pdf)

    :param float pad_value: value used for padding
    """

    def __init__(self, pad_value=0.0):
        super(LengthRegularizer, self).__init__()
        self.pad_value = pad_value

    def forward(self, xs, ds, ilens):
        """Apply length regularizer

        :param torch.Tensor x: input tensor with the shape (B, Tmax, D)
        :param torch.Tensor d: duration of each components of each sequence B, T,)
        :param torch.Tensor d: batch of input lengths (B,)
        :return torch.Tensor: length regularized input tensor (B, T*, D)
        """
        xs = [x[:ilen] for x, ilen in zip(xs, ilens)]
        ds = [d[:ilen] for d, ilen in zip(ds, ilens)]
        xs = [self._repeat_one_sequence(x, d) for x, d in zip(xs, ds)]

        return pad_list(xs, self.pad_value)

    def _repeat_one_sequence(self, x, d):
        """Repeat each frame according to duration

        >>> x = torch.tensor([[1], [2], [3]])
        tensor([[1],
                [2],
                [3]])
        >>> d = torch.tensor([1, 2, 3])
        tensor([1, 2, 3])
        >>> self._repeat_one_sequence(x, d)
        tensor([[1],
                [2],
                [2],
                [3],
                [3],
                [3]])

        :param torch.Tensor x: input tensor with the shape (T, D)
        :param torch.Tensor d: duration of each frame of input tensor (T,)
        :return torch.Tensor: length regularized input tensor (T*, D)
        """
        return torch.cat([x_.repeat(int(d_), 1) for x_, d_ in zip(x, d) if d_ != 0], dim=0)


class DurationPredictor(torch.nn.Module):
    """Duration predictor module

    Reference:
        - FastSpeech: Fast, Robust and Controllable Text to Speech
          (https://arxiv.org/pdf/1905.09263.pdf)

    :param int idim: input dimension
    :param int n_layers: number of convolutional layers
    :param int n_chans: number of channels of convolutional layers
    :param int kernel_size: kernel size of convolutional layers
    :param float dropout_rate: dropout rate
    """

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0):
        super(DurationPredictor, self).__init__()
        self.offset = 1.0
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = torch.nn.Linear(n_chans, 1)

    def forward(self, xs, x_masks=None):
        """Calculate duration predictor forward propagation

        :param torch.Tensor xs: input tensor (B, Tmax, idim)
        :param torch.Tensor x_masks: mask for removing padded part (B, Tmax)
        :return torch.Tensor: predicted duration tensor in log domain (B, Tmax, 1)
        """
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for idx in range(len(self.conv)):
            xs = self.conv[idx](xs)  # (B, C, Tmax)
        xs = self.linear(xs.transpose(1, -1)).squeeze(-1)  # (B, Tmax)

        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)

        return xs

    def inference(self, xs, x_masks=None):
        """Inference duration

        :param torch.Tensor xs: input tensor with tha shape (B, Tmax, idim)
        :param torch.Tensor x_masks: mask for removing padded part (B, Tmax)
        :return torch.Tensor: predicted duration tensor with the shape (B, Tmax, 1)
        """
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for idx in range(len(self.conv)):
            xs = self.conv[idx](xs)  # (B, C, Tmax)
        xs = self.linear(xs.transpose(1, -1))  # (B, Tmax, 1)
        xs = torch.ceil(torch.exp(xs - self.offset)).squeeze(-1).long()  # use ceil to avoid length = 0

        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0)

        return xs
