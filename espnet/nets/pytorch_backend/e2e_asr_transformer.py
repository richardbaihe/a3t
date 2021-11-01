# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

from argparse import Namespace
from distutils.util import strtobool
import logging
import math

import numpy
import torch
import torch.nn as nn
import random

from IPython import embed
from collections import defaultdict, Counter
import time

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr import Reporter
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.argument import (
    add_arguments_transformer_common,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.utils.fill_missing_args import fill_missing_args


class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group = add_arguments_transformer_common(group)

        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def get_total_subsampling_factor(self):
        """Get total subsampling factor."""
        return self.encoder.conv_subsampling_factor * int(numpy.prod(self.subsample))

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)

        # fill missing arguments for compatibility
        # args = fill_missing_args(args, self.add_arguments)

        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate

        self.intermediate_ctc_weight = args.intermediate_ctc_weight
        self.intermediate_ctc_layers = []
        if args.intermediate_ctc_layer != "":
            self.intermediate_ctc_layers = [
                int(i) for i in args.intermediate_ctc_layer.split(",")
            ]

        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            mlm=False,
        )
        mlm=True if (args.mlm_weight > 0. and args.mlm_prob > 0. and not args.spec_rec) else False,
        if args.mtlalpha < 1:
            self.decoder = Decoder(
                odim=odim,
                selfattention_layer_type=args.transformer_decoder_selfattn_layer_type,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                conv_wshare=args.wshare,
                conv_kernel_length=args.ldconv_decoder_kernel_length,
                conv_usebias=args.ldconv_usebias,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
            )
            self.criterion = LabelSmoothingLoss(
                odim,
                ignore_id,
                args.lsm_weight,
                args.transformer_length_normalized_loss,
            )
        else:
            self.decoder = None
            self.criterion = None
        self.blank = 0
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="asr", arch="transformer")
        self.reporter = Reporter()

        self.reset_parameters(args)
        self.adim = args.adim  # used for CTC (equal to d_model)
        self.mtlalpha = args.mtlalpha

        if args.mtlalpha > 0.0:
            self.ctc = CTC(
                odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
            )
        else:
            self.ctc = None
        try:
            self.mlm_weight = args.mlm_weight
            self.mlm_prob = args.mlm_prob
            self.mlm_layer = args.mlm_layer
            self.finetune_wo_mlm = args.finetune_wo_mlm
            self.max_span = args.max_span
            self.min_span = args.min_span
            self.span = args.span
            self.widening = args.widening
            self.spec_rec = args.spec_rec
        except Exception:
            print('missing arg')
            self.mlm_weight = 0.
            self.mlm_prob = 0.
            self.mlm_layer = -1
            self.finetune_wo_mlm =True
            self.max_span = 40
            self.min_span = 5
            self.span = False
            self.spec_rec = False
            # import os
            # os._exit(0)
        try:
            self.smart_seg = args.smart_seg
        except Exception:
            self.smart_seg = False
        if self.mlm_weight > 0 and self.mlm_prob > 0:
            if self.spec_rec:
                logging.warning("Use Spectrogram Reconstruction")
                self.sfc = nn.Linear(args.adim, 20 * args.adim)
                self.deconv = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(args.adim, args.adim, 3, 2),
                    # torch.nn.ReLU(),
                    torch.nn.ConvTranspose2d(args.adim, 1, 3, 2),
                )
                self.mask_feature = torch.nn.Parameter(torch.empty((83)).normal_())
            else:
                logging.warning("Use Feature after CNN Reconstruction")
                self.sfc = nn.Linear(args.adim, args.adim)
            self.mse = nn.MSELoss(reduce=False)
        if self.finetune_wo_mlm:
            self.mlm_weight = 0.

        if args.report_cer or args.report_wer:
            self.error_calculator = ErrorCalculator(
                args.char_list,
                args.sym_space,
                args.sym_blank,
                args.report_cer,
                args.report_wer,
            )
        else:
            self.error_calculator = None
        self.rnnlm = None

    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        initialize(self, args.transformer_init)

    def forward(self, xs_pad, ilens, ys_pad, epoch=0):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 1. forward encoder
        if self.smart_seg and self.training:
            xs_pad, seg_idx, seg = xs_pad
        max_ilen = max(ilens)
        # max_ilen = int(torch.floor(max(ilens).float() / 4) * 4 + 3)
        xs_pad = xs_pad[:, : max_ilen]  # for data parallel
        # print(xs_pad[:,:,0].size())
        src_mask = make_non_pad_mask(ilens.tolist(), xs_pad[:,:,0]).to(xs_pad.device).unsqueeze(-2)
        # dsafdasfad
        if self.finetune_wo_mlm or not self.training or self.mlm_weight == 0:
            tmp_mask_prob = 0.
            self.mlm_weight = 0.
            masked_position = None
            loss_mlm = 0.
        else:
            tmp_mask_prob = self.mlm_prob
            if self.spec_rec:
                xs_pad_placeholder = xs_pad
                src_mask_placeholder = src_mask
            else:
                xs_pad_placeholder = xs_pad[:, :-2:2, :][:, :-2:2, :]
                src_mask_placeholder = src_mask[:, :, :-2:2][:, :, :-2:2]
            if self.smart_seg:
                masked_position = self.smart_masking(xs_pad_placeholder, 
                                                    src_mask_placeholder, 
                                                    tmp_mask_prob,
                                                    seg_idx,
                                                    seg)
            elif self.span:
                if self.widening:
                    max_span_tmp = min(self.max_span, self.min_span+epoch*4)
                else:
                    max_span_tmp = self.max_span
                masked_position = self.span_masking(xs_pad_placeholder, 
                                                    src_mask_placeholder, 
                                                    tmp_mask_prob, 
                                                    max_span=max_span_tmp, min_span=self.min_span)
            else:
                masked_position = self.random_masking(xs_pad_placeholder, 
                                                    src_mask_placeholder, 
                                                    tmp_mask_prob)
        if self.spec_rec and not self.finetune_wo_mlm and self.training:
            after_conv_mask_position = None
            masked_position = masked_position.unsqueeze(-1).expand_as(xs_pad)
            xs_pad_input = xs_pad.masked_fill(masked_position, 0) + \
                    self.mask_feature.unsqueeze(0).unsqueeze(0).expand_as(xs_pad).masked_fill(~masked_position, 0)
        else:
            xs_pad_input = xs_pad
            after_conv_mask_position = masked_position
        (hs_pad, mlm_output), hs_mask, (hs_emb, mlm_position) = self.encoder(xs_pad_input, 
                                                                            src_mask, 
                                                                            masked_position=after_conv_mask_position, 
                                                                            mlm_layer=self.mlm_layer)
        # print(self.odim, hs_pad.size(), hs_mask.size(), hs_emb.size(), mlm_position.size())
        
        # compute mlm loss
        # print('after encode', hs_pad.size(), hs_mask.size(), masked_position.size())
        
        loss_mlm = 0.
        if self.mlm_weight > 0 and tmp_mask_prob > 0:
            if self.spec_rec:
                true_label_position = (torch.rand(masked_position[:,:,:1].size()) < (tmp_mask_prob * .15)).expand_as(masked_position).to(xs_pad.device)
                mlm_loss_position = (true_label_position + masked_position) > 0
                b, c, f = hs_pad.size()
                deconv_hs_pad = self.sfc(mlm_output).view(b, c, f, -1).contiguous().transpose(1, 2)
                deconv_hs_pad = self.deconv(deconv_hs_pad).squeeze()
                loss_mlm = torch.masked_select(self.mse(deconv_hs_pad, xs_pad), mlm_loss_position).mean()
                # print(loss_mlm)
            else:
                true_label_position = (torch.rand(mlm_position[:,:,:1].size()) < (tmp_mask_prob * .3)).expand_as(mlm_position).to(hs_emb.device)
                mlm_loss_position = (true_label_position + mlm_position) > 0
                loss_mlm = (self.mse(self.sfc(mlm_output.view(-1, self.adim)), 
                                    hs_emb.view(-1, self.adim)) * mlm_loss_position.view(-1, self.adim).float()).sum() \
                                    / (mlm_loss_position.float().sum() + 1e-10)
            # print(loss_mlm)
        self.hs_pad = hs_pad

        # 2. forward decoder
        if self.decoder is not None:
            ys_in_pad, ys_out_pad = add_sos_eos(
                ys_pad, self.sos, self.eos, self.ignore_id
            )
            ys_mask = target_mask(ys_in_pad, self.ignore_id)
            pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
            self.pred_pad = pred_pad

            # 3. compute attention loss
            loss_att = self.criterion(pred_pad, ys_out_pad)
            self.acc = th_accuracy(
                pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
            )
        else:
            loss_att = None
            self.acc = None

        # TODO(karita) show predicted text
        # TODO(karita) calculate these stats
        cer_ctc = None
        loss_intermediate_ctc = 0.0
        if self.mtlalpha == 0.0:
            loss_ctc = None
        else:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
            if not self.training and self.error_calculator is not None:
                ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
                cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
            # for visualization
            if not self.training:
                self.ctc.softmax(hs_pad)

            if self.intermediate_ctc_weight > 0 and self.intermediate_ctc_layers:
                for hs_intermediate in hs_intermediates:
                    # assuming hs_intermediates and hs_pad has same length / padding
                    loss_inter = self.ctc(
                        hs_intermediate.view(batch_size, -1, self.adim), hs_len, ys_pad
                    )
                    loss_intermediate_ctc += loss_inter

                loss_intermediate_ctc /= len(self.intermediate_ctc_layers)

        # 5. compute cer/wer
        if self.training or self.error_calculator is None or self.decoder is None:
            cer, wer = None, None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        # copied from e2e_asr
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = loss_ctc
            if self.intermediate_ctc_weight > 0:
                self.loss = (
                    1 - self.intermediate_ctc_weight
                ) * loss_ctc + self.intermediate_ctc_weight * loss_intermediate_ctc
            loss_att_data = None
            loss_ctc_data = float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            if self.intermediate_ctc_weight > 0:
                self.loss = (
                    (1 - alpha - self.intermediate_ctc_weight) * loss_att
                    + alpha * loss_ctc
                    + self.intermediate_ctc_weight * loss_intermediate_ctc
                )
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)

        if self.mlm_weight == 1:
            self.loss = loss_mlm
        else:
            self.loss = self.mlm_weight * loss_mlm + (1 - self.mlm_weight) * self.loss
        loss_data = float(self.loss)
        loss_mlm_data = float(loss_mlm)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data, loss_mlm=loss_mlm_data,
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
        return self.loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)

        (enc_output, _), _, _ = self.encoder(x, None)
        return enc_output.squeeze(0)

    def recognize(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        enc_output = self.encode(x).unsqueeze(0)
        if self.mtlalpha == 1.0:
            recog_args.ctc_weight = 1.0
            logging.info("Set to pure CTC decoding mode.")

        if self.mtlalpha > 0 and recog_args.ctc_weight == 1.0:
            from itertools import groupby

            lpz = self.ctc.argmax(enc_output)
            collapsed_indices = [x[0] for x in groupby(lpz[0])]
            hyp = [x for x in filter(lambda x: x != self.blank, collapsed_indices)]
            nbest_hyps = [{"score": 0.0, "yseq": [self.sos] + hyp}]
            if recog_args.beam_size > 1:
                raise NotImplementedError("Pure CTC beam search is not implemented.")
            # TODO(hirofumi0810): Implement beam search
            return nbest_hyps
        elif self.mtlalpha > 0 and recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(enc_output)
            lpz = lpz.squeeze(0)
        else:
            lpz = None

        h = enc_output.squeeze(0)

        logging.info("input lengths: " + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None}
        else:
            hyp = {"score": 0.0, "yseq": [y]}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, numpy)
            hyp["ctc_state_prev"] = ctc_prefix_score.initial_state()
            hyp["ctc_score_prev"] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        import six

        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug("position " + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp["yseq"][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp["yseq"]).unsqueeze(0)
                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(
                            self.decoder.forward_one_step, (ys, ys_mask, enc_output)
                        )
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                else:
                    local_att_scores = self.decoder.forward_one_step(
                        ys, ys_mask, enc_output
                    )[0]

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                    local_scores = (
                        local_att_scores + recog_args.lm_weight * local_lm_scores
                    )
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1
                    )
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp["yseq"], local_best_ids[0], hyp["ctc_state_prev"]
                    )
                    local_scores = (1.0 - ctc_weight) * local_att_scores[
                        :, local_best_ids[0]
                    ] + ctc_weight * torch.from_numpy(
                        ctc_scores - hyp["ctc_score_prev"]
                    )
                    if rnnlm:
                        local_scores += (
                            recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                        )
                    local_best_scores, joint_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp["rnnlm_prev"] = rnnlm_state
                    if lpz is not None:
                        new_hyp["ctc_state_prev"] = ctc_states[joint_best_ids[0, j]]
                        new_hyp["ctc_score_prev"] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypothes: " + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    "best hypo: "
                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
                )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("adding <eos> in the last position in the loop")
                for hyp in hyps:
                    hyp["yseq"].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp["score"] += recog_args.lm_weight * rnnlm.final(
                                hyp["rnnlm_prev"]
                            )
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info("end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
            : min(len(ended_hyps), recog_args.nbest)
        ]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize(x, recog_args, char_list, rnnlm)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        return nbest_hyps

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights (B, H, Lmax, Tmax)
        :rtype: float ndarray
        """
        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        ret = dict()
        for name, m in self.named_modules():
            if (
                isinstance(m, MultiHeadedAttention)
                or isinstance(m, DynamicConvolution)
                or isinstance(m, RelPositionMultiHeadedAttention)
            ):
                ret[name] = m.attn.cpu().numpy()
            if isinstance(m, DynamicConvolution2D):
                ret[name + "_time"] = m.attn_t.cpu().numpy()
                ret[name + "_freq"] = m.attn_f.cpu().numpy()
        self.train()
        return ret

    def random_masking_torch(self, xs_pad, src_mask, mask_prob):
        masked_position = (torch.rand(xs_pad.size()[:2]) < mask_prob).to(xs_pad.device)
        # no sos and eos
        masked_position[:,0] = 0
        non_eos_mask = src_mask.view_as(masked_position)
        non_eos_mask = torch.cat((non_eos_mask[:,1:], torch.zeros_like(non_eos_mask[:,:1])), 1)
        masked_position = masked_position * non_eos_mask
        return masked_position
    
    def random_masking(self, xs_pad, src_mask, mask_prob):
        masked_position = numpy.random.rand(*xs_pad.size()[:2]) < mask_prob
        # no sos and eos
        masked_position[:,0] = 0
        non_eos_mask = src_mask.view(xs_pad.size()[:2]).float().cpu().numpy()
        non_eos_mask = numpy.concatenate((non_eos_mask[:,1:], numpy.zeros(non_eos_mask[:,:1].shape, dtype=int)), axis=1)
        masked_position = masked_position * non_eos_mask
        return torch.BoolTensor(masked_position).to(xs_pad.device)
    
    def span_masking(self, xs_pad, src_mask, mask_prob, max_span=10, min_span=1):
        assert max_span >= min_span
        # xs_pad = xs_pad[:, :-2:2, :][:, :-2:2, :]
        p = 0.2
        len_distrib = [p * (1-p) ** (i - min_span) for i in range(min_span, max_span + 1)]
        len_distrib = [x / (sum(len_distrib)) for x in len_distrib]
        bz, sent_len, _ = xs_pad.size()
        mask_num_lower = math.ceil(sent_len * mask_prob)
        masked_position = numpy.zeros((bz, sent_len))
        for idx in range(bz):
            while numpy.sum(masked_position[idx]) < mask_num_lower:
                span_start = numpy.random.choice(sent_len)
                span_len = numpy.random.choice(numpy.arange(min_span, max_span+1), p=len_distrib)
                while masked_position[idx, span_start] > 0 or span_start+span_len-1 >= sent_len or masked_position[idx, span_start+span_len-1] > 0:
                    span_start = numpy.random.choice(sent_len)
                    span_len = numpy.random.choice(numpy.arange(min_span, max_span+1), p=len_distrib)
                masked_position[idx, span_start:span_start+span_len] = 1
        non_eos_mask = src_mask.view(xs_pad.size()[:2]).float().cpu().numpy()
        # non_eos_mask = numpy.concatenate((non_eos_mask[:,1:], numpy.zeros(non_eos_mask[:,:1].shape, dtype=int)), axis=1)
        masked_position = masked_position * non_eos_mask
        return torch.BoolTensor(masked_position).to(xs_pad.device)

    def smart_masking(self, xs_pad, src_mask, mask_prob, seg_idx, seg_list):
        # xs_pad = xs_pad[:, :-2:2, :][:, :-2:2, :]
        bz, sent_len, _ = xs_pad.size()
        mask_num_lower = math.ceil(sent_len * mask_prob)
        masked_position = torch.zeros((bz, sent_len))
        for idx in range(bz):
            seg = seg_list[seg_idx[idx]][:]
            random.shuffle(seg)
            while torch.sum(masked_position[idx]) < mask_num_lower and len(seg) > 0:
                start, end = seg.pop(0)
                masked_position[idx, start:end] = 1
        masked_position = masked_position.to(xs_pad.device)
        non_eos_mask = src_mask.view_as(masked_position).to(xs_pad.device)
        masked_position = (masked_position * non_eos_mask) > 0
        return torch.cuda.BoolTensor(masked_position)

    def chunk_streaming(
        self, x, args, char_list=None, rnnlm=None, rnnlm_dict=None, use_jit=False,
    ):
        chunk = args.chunk
        threshold = 0.005
        ctc_weight = 0.5
        ctc_lm_weight = 0.5
        lm_weight = 0.5
        look_ahead = 10
        penalty = 2
        maxlen = 100
        minlen = 1
        beam = args.beam_size

        hyp = {'score': 0.0, 'yseq': [self.sos], 'rnnlm_prev': None, 'seq': char_list[self.sos],
               'last_time': [], "ctc_score": 0.0,  "rnnlm_score": 0.0, "att_score": 0.0,
               "cache": None, "precache": None, "preatt_score": 0.0, "prev_score":0.0,
               'yseq_chunk_idx':[0]}

        hyps = {char_list[self.sos]: hyp}
        Pb_prev, Pnb_prev = Counter(), Counter()
        Pb, Pnb = Counter(), Counter()
        l = char_list[self.sos]
        Pb_prev[l] = 1
        Pnb_prev[l] = 0
        hat_att = {}
        total_copy = 0
        r = numpy.ndarray((len(char_list)), dtype=numpy.float32)
        ilen = x.shape[0]
        vy = torch.zeros(1).long().to(next(self.parameters()).device)
        chunk_num = math.ceil(ilen/chunk)
        start_idx = 0

        # trans_hyp = {"score": 0.0, "yseq": [self.sos], 'yseq_chunk_idx':[0]}
        # trans_hyps = [trans_hyp] 
        same_prefix_len_record = []
        time_record = []
        
        for chunk_index in range(1, chunk_num+1):
            start_time = time.time()
            hat_att[chunk_index] = {}
            enc_output = self.encode(x[:chunk*chunk_index+look_ahead, :]).unsqueeze(0)
            lpz = torch.nn.functional.softmax(self.ctc.ctc_lo(enc_output), dim=-1)
            lpz = lpz.squeeze().cpu().detach().numpy()
            enc_len = min(enc_output.size(1), ((chunk*chunk_index+1)//2+1)//2)
            for i in range(start_idx, enc_len):
                
                hyps_ctc = {}
                pos_ctc = numpy.where(lpz[i] > threshold)[0]

                hyps_res = {}
                for l, hyp in hyps.items():
                    if l in hat_att[chunk_index]:
                        hyp['tmp_cache'] = hat_att[chunk_index][l]['cache']
                        hyp['tmp_att_score'] = hat_att[chunk_index][l]['att_scores']
                    else:
                        hyps_res[l] = hyp
                tmp = self.clusterbyLength(hyps_res) # This step clusters hyps according to length dict:{length,hyps}
                start = time.time()
                # pre-compute beam
                self.compute_hyps(tmp, enc_output, hat_att[chunk_index])
                total_copy += time.time()-start
                # Assign score and tokens to hyps
                #print(hyps.keys())
                                
                for l, hyp in hyps.items():
                    if 'tmp_att_score' not in hyp:
                        continue #Todo check why
                    local_att_scores = hyp['tmp_att_score']
                    local_best_scores, local_best_ids = torch.topk(local_att_scores, 5, dim=1)
                    pos_att = numpy.array(local_best_ids[0].cpu())
                    pos = numpy.union1d(pos_ctc, pos_att)
                    hyp['pos'] = pos

                # pre-compute ctc beam
                hyps_ctc_compute = self.get_ctchyps2compute(hyps, hyps_ctc, i)
                hyps_res2 = {}
                for l, hyp in hyps_ctc_compute.items():
                    l_minus = ' '.join(l.split()[:-1])
                    if l_minus in hat_att[chunk_index]:
                        hyp['tmp_cur_new_cache'] = hat_att[chunk_index][l_minus]['cache']
                        hyp['tmp_cur_att_scores'] = hat_att[chunk_index][l_minus]['att_scores']
                    else:
                        hyps_res2[l] = hyp
                tmp2_cluster = self.clusterbyLength(hyps_res2)
                self.compute_hyps_ctc(tmp2_cluster, enc_output, hat_att[chunk_index])


                for l, hyp in hyps.items():
                    start = time.time()
                    l_id = hyp['yseq']
                    l_id_idx = hyp['yseq_chunk_idx']
                    l_end = l_id[-1]
                    char_l_end = char_list[l_end]
                    vy[0] = rnnlm_dict[char_l_end] if char_l_end in rnnlm_dict else rnnlm_dict['<unk>']
                    prefix_len = len(l_id)
                    assert l_end == rnnlm_dict[char_l_end], "not same dict"
                    if rnnlm:
                        rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                    else:
                        rnnlm_state = None
                        local_lm_scores = torch.zeros([1, len(char_list)])

                    r = lpz[i] * (Pb_prev[l] + Pnb_prev[l])
            #     print("!!!","Decoding None")

                    start = time.time()
                    if 'tmp_att_score' not in hyp:
                        continue #Todo check why
                    local_att_scores = hyp['tmp_att_score']
                    new_cache = hyp['tmp_cache']
                    align = [0] * prefix_len
                    align[:prefix_len - 1] = hyp['last_time'][:]
                    align[-1] = i
                    pos = hyp['pos']
                    if 0 in pos or l_end in pos:
                        if l not in hyps_ctc:
                            hyps_ctc[l] = {'yseq': l_id, 'yseq_chunk_idx': l_id_idx}
                            hyps_ctc[l]['rnnlm_prev'] = hyp['rnnlm_prev']
                            hyps_ctc[l]['rnnlm_score'] = hyp['rnnlm_score']
                            if l_end != self.eos:
                                hyps_ctc[l]['last_time'] = [0] * prefix_len
                                hyps_ctc[l]['last_time'][:] = hyp['last_time'][:]
                                hyps_ctc[l]['last_time'][-1] = i
                                cur_att_scores = hyps_ctc_compute[l]["tmp_cur_att_scores"]
                                cur_new_cache = hyps_ctc_compute[l]["tmp_cur_new_cache"]
                                hyps_ctc[l]['att_score'] = hyp['preatt_score'] + \
                                                        float(cur_att_scores[0, l_end].data)
                                hyps_ctc[l]['cur_att'] = float(cur_att_scores[0, l_end].data)
                                hyps_ctc[l]['cache'] = cur_new_cache
                            else:
                                if len(hyps_ctc[l]["yseq"]) > 1:
                                    hyps_ctc[l]["end"] = True
                                hyps_ctc[l]['last_time'] = []
                                hyps_ctc[l]['att_score'] = hyp['att_score']
                                hyps_ctc[l]['cur_att'] = 0
                                hyps_ctc[l]['cache'] = hyp['cache']

                            hyps_ctc[l]['prev_score'] = hyp['prev_score']
                            hyps_ctc[l]['preatt_score'] = hyp['preatt_score']
                            hyps_ctc[l]['precache'] = hyp['precache']
                            hyps_ctc[l]['seq'] = hyp['seq']


                    for c in list(pos):
                        char_c = char_list[c]
                        rnn_c = rnnlm_dict[char_c] if char_c in rnnlm_dict else rnnlm_dict['<unk>']
                        if c == 0:
                            Pb[l] += lpz[i][0] * (Pb_prev[l] + Pnb_prev[l])
                        else:
                            l_plus = l+ " " +char_list[c]
                            if l_plus not in hyps_ctc:
                                hyps_ctc[l_plus] = {}
                                if "end" in hyp:
                                    hyps_ctc[l_plus]['yseq'] = True
                                hyps_ctc[l_plus]['yseq'] = [0] * (prefix_len + 1)
                                hyps_ctc[l_plus]['yseq'][:len(hyp['yseq'])] = l_id
                                hyps_ctc[l_plus]['yseq'][-1] = int(c)

                                hyps_ctc[l_plus]['yseq_chunk_idx'] = [0] * (prefix_len + 1)
                                hyps_ctc[l_plus]['yseq_chunk_idx'][:len(hyp['yseq_chunk_idx'])] = l_id_idx
                                hyps_ctc[l_plus]['yseq_chunk_idx'][-1] = chunk_index

                                hyps_ctc[l_plus]['rnnlm_prev'] = rnnlm_state
                                hyps_ctc[l_plus]['rnnlm_score'] = hyp['rnnlm_score'] + float(local_lm_scores[0, rnn_c].data)
                                hyps_ctc[l_plus]['att_score'] = hyp['att_score'] \
                                                                + float(local_att_scores[0, c].data)
                                hyps_ctc[l_plus]['cur_att'] = float(local_att_scores[0, c].data)
                                hyps_ctc[l_plus]['cache'] = new_cache
                                hyps_ctc[l_plus]['precache'] = hyp['cache']
                                hyps_ctc[l_plus]['preatt_score'] = hyp['att_score']
                                hyps_ctc[l_plus]['prev_score'] = hyp['score']
                                hyps_ctc[l_plus]['last_time'] = align
                                hyps_ctc[l_plus]['rule_penalty'] = 0
                                hyps_ctc[l_plus]['seq'] = l_plus
                            if l_end != self.eos and c == l_end:
                                Pnb[l_plus] += lpz[i][l_end] * Pb_prev[l]
                                Pnb[l] += lpz[i][l_end] * Pnb_prev[l]
                            else:
                                Pnb[l_plus] += r[c]


                            if l_plus not in hyps:
                                Pb[l_plus] += lpz[i][0] * (Pb_prev[l_plus] + Pnb_prev[l_plus])
                                Pnb[l_plus] += lpz[i][c] * Pnb_prev[l_plus]
                #total_copy += time.time() - start
                for l in hyps_ctc.keys():
                    if Pb[l] != 0 or Pnb[l] != 0:
                        hyps_ctc[l]['ctc_score'] = numpy.log(Pb[l] + Pnb[l])
                    else:
                        hyps_ctc[l]['ctc_score'] = float('-inf')
                    local_score = hyps_ctc[l]['ctc_score'] + ctc_lm_weight * hyps_ctc[l]['rnnlm_score'] + \
                                penalty * (len(hyps_ctc[l]['yseq']))
                    hyps_ctc[l]['local_score'] = local_score
                    hyps_ctc[l]['score'] = (1 - ctc_weight) * hyps_ctc[l]['att_score'] \
                                        + ctc_weight * hyps_ctc[l]['ctc_score'] + \
                                        penalty * (len(hyps_ctc[l]['yseq'])) + \
                                        lm_weight * hyps_ctc[l]['rnnlm_score']
                Pb_prev = Pb
                Pnb_prev = Pnb
                Pb = Counter()
                Pnb = Counter()

                hyps = sorted(hyps_ctc.items(), key=lambda x: x[1]['score'], reverse=True)[:beam]
                hyps = dict(hyps)
            same_prefix_len = self.track_prefix(hyps)
            # print([char_list[cc] for cc in trans_hyp["yseq"]], enc_output.size())
            # embed()

            start_idx = enc_len
            same_prefix_len_record.append(same_prefix_len)
            time_record.append(time.time() - start_time)
        hyps = sorted(hyps.items() , key=lambda x: x[1]['score'], reverse=True)[:beam]
        hyps = dict(hyps)
        logging.info('input lengths: ' + str(x.shape[0]))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))
        if "<eos>" in hyps.keys():
            del hyps["<eos>"]

        best = list(hyps.keys())[0]
        ids = hyps[best]['yseq']
        score = hyps[best]['score']
        print(best, '\n', hyps[best]['yseq_chunk_idx'])
        assert len(hyps[best]['yseq']) == len(hyps[best]['yseq_chunk_idx']), (hyps[best]['yseq'], hyps[best]['yseq_chunk_idx'])
        logging.info('score: ' + str(score))


        hyps[best]["same_prefix_len_record"] = same_prefix_len_record
        hyps[best]["time_record"] = time_record
                     
        return [hyps[best]]   

    def track_prefix(self, hyps):
        same_len = 0
        min_len = min([len(hyp['yseq']) for l, hyp in hyps.items()])
        for i in range(min_len):
            c = None
            for l, hyp in hyps.items():
                if c is None:
                    c = hyp['yseq'][i]
                else:
                    if c != hyp['yseq'][i]:
                        return same_len
            same_len += 1
        return same_len

    def removeIlegal(self,hyps):
        max_y = max([len(hyp['yseq']) for l, hyp in hyps.items()])
        for_remove = []
        for l, hyp in hyps.items():
            if max_y - len(hyp['yseq']) > 4:
                for_remove.append(l)
        for cur_str in for_remove:
            del hyps[cur_str]

    def clusterbyLength(self,hyps):
        tmp = {}
        for l, hyp in hyps.items():
            prefix_len = len(hyp['yseq'])
            if prefix_len > 1 and hyp['yseq'][-1] == self.eos:
                continue
            else:
                if prefix_len not in tmp:
                    tmp[prefix_len] = []
                tmp[prefix_len].append(hyp)
        return tmp
    
    def compute_hyps(self, current_hyps, enc_output, hat_att):
        for length, hyps_t in current_hyps.items():
            # print(length, 'compute_hyps')
            ys_mask = subsequent_mask(length).unsqueeze(0).cuda()
            ys_mask4use = ys_mask.repeat(len(hyps_t), 1, 1)

            # print(ys_mask4use.shape)
            l_id = [hyp_t['yseq'] for hyp_t in hyps_t]
            ys4use = torch.tensor(l_id).cuda()
            enc_output4use = enc_output.repeat(len(hyps_t), 1, 1)
            if hyps_t[0]["cache"] is None:
                cache4use = None
            else:
                cache4use = []
                for decode_num in range(len(hyps_t[0]["cache"])):
                    current_cache = []
                    for hyp_t in hyps_t:
                        current_cache.append(hyp_t["cache"][decode_num].squeeze(0))
                    # print( torch.stack(current_cache).shape)

                    current_cache = torch.stack(current_cache)
                    cache4use.append(current_cache)

            # partial_mask4use = []
            # for hyp_t in hyps_t:
            #     #partial_mask4use.append(torch.ones([1, len(hyp_t['last_time'])+1, enc_mask.shape[1]]).byte())
            #     align = [0] * length
            #     align[:length - 1] = hyp_t['last_time'][:]
            #     align[-1] = curren_frame
            #     align_tensor = torch.tensor(align).unsqueeze(0)
            #     if chunk:
            #         partial_mask = enc_mask[0][align_tensor]
            #     else:
            #         right_window = self.right_window
            #         partial_mask = trigger_mask(1, total_frame, align_tensor,
            #                                 self.left_window, right_window)
            #     partial_mask4use.append(partial_mask)

            # partial_mask4use = torch.stack(partial_mask4use).cuda().squeeze(1)
            local_att_scores_b, new_cache_b = self.decoder.forward_one_step(ys4use, ys_mask4use,
                                                                            enc_output4use, None, cache4use)
            for idx, hyp_t in enumerate(hyps_t):
                hyp_t['tmp_cache'] = [new_cache_b[decode_num][idx].unsqueeze(0)
                                      for decode_num in range(len(new_cache_b))]
                hyp_t['tmp_att_score'] = local_att_scores_b[idx].unsqueeze(0)
                hat_att[hyp_t['seq']] = {}
                hat_att[hyp_t['seq']]['cache'] = hyp_t['tmp_cache']
                hat_att[hyp_t['seq']]['att_scores'] = hyp_t['tmp_att_score']

    def get_ctchyps2compute(self,hyps,hyps_ctc,current_frame):
        tmp2 = {}
        for l, hyp in hyps.items():
            l_id = hyp['yseq']
            l_end = l_id[-1]
            if "pos" not in hyp:
                continue
            if 0 in hyp['pos'] or l_end in hyp['pos']:
                #l_minus = ' '.join(l.split()[:-1])
                #if l_minus in hat_att:
                #    hyps[l]['tmp_cur_new_cache'] = hat_att[l_minus]['cache']
                #    hyps[l]['tmp_cur_att_scores'] = hat_att[l_minus]['att_scores']
                #    continue
                if l not in hyps_ctc and l_end != self.eos:
                    tmp2[l] = {'yseq': l_id}
                    tmp2[l]['seq'] = l
                    tmp2[l]['rnnlm_prev'] = hyp['rnnlm_prev']
                    tmp2[l]['rnnlm_score'] = hyp['rnnlm_score']
                    if l_end != self.eos:
                        tmp2[l]['last_time'] = [0] * len(l_id)
                        tmp2[l]['last_time'][:] = hyp['last_time'][:]
                        tmp2[l]['last_time'][-1] = current_frame
            # print(l, 'get_ctchyps2compute', tmp2)
        return tmp2

    def compute_hyps_ctc(self, hyps_ctc_cluster, enc_output, hat_att):
        for length, hyps_t in hyps_ctc_cluster.items():
            ys_mask = subsequent_mask(length - 1).unsqueeze(0).cuda()
            ys_mask4use = ys_mask.repeat(len(hyps_t), 1, 1)
            l_id = [hyp_t['yseq'][:-1] for hyp_t in hyps_t]
            ys4use = torch.tensor(l_id).cuda()
            enc_output4use = enc_output.repeat(len(hyps_t), 1, 1)
            if "precache" not in hyps_t[0] or hyps_t[0]["precache"] is None:
                cache4use = None
            else:
                cache4use = []
                for decode_num in range(len(hyps_t[0]["precache"])):
                    current_cache = []
                    for hyp_t in hyps_t:
                        # print(length, hyp_t["yseq"], hyp_t["cache"][0].shape,
                        #       hyp_t["cache"][2].shape, hyp_t["cache"][4].shape)
                        current_cache.append(hyp_t["precache"][decode_num].squeeze(0))
                    current_cache = torch.stack(current_cache)
                    cache4use.append(current_cache)
            # partial_mask4use = []
            # for hyp_t in hyps_t:
            #     #partial_mask4use.append(torch.ones([1, len(hyp_t['last_time']), enc_mask.shape[1]]).byte())
            #     align = hyp_t['last_time']
            #     align_tensor = torch.tensor(align).unsqueeze(0)
            #     if chunk:
            #         partial_mask = enc_mask[0][align_tensor]
            #     else:
            #         right_window = self.right_window
            #         partial_mask = trigger_mask(1, total_frame, align_tensor, self.left_window, right_window)
            #     partial_mask4use.append(partial_mask)

            # partial_mask4use = torch.stack(partial_mask4use).cuda().squeeze(1)

            local_att_scores_b, new_cache_b = \
                self.decoder.forward_one_step(ys4use, ys_mask4use,
                                              enc_output4use, None, cache4use)
            for idx, hyp_t in enumerate(hyps_t):
                hyp_t['tmp_cur_new_cache'] = [new_cache_b[decode_num][idx].unsqueeze(0)
                                              for decode_num in range(len(new_cache_b))]
                hyp_t['tmp_cur_att_scores'] = local_att_scores_b[idx].unsqueeze(0)
                l_minus = ' '.join(hyp_t['seq'].split()[:-1])
                hat_att[l_minus] = {}
                hat_att[l_minus]['att_scores'] = hyp_t['tmp_cur_att_scores']
                hat_att[l_minus]['cache'] = hyp_t['tmp_cur_new_cache']
    def calculate_all_ctc_probs(self, xs_pad, ilens, ys_pad):
        """E2E CTC probability calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: CTC probability (B, Tmax, vocab)
        :rtype: float ndarray
        """
        ret = None
        if self.mtlalpha == 0:
            return ret

        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        for name, m in self.named_modules():
            if isinstance(m, CTC) and m.probs is not None:
                ret = m.probs.cpu().numpy()
        self.train()
        return ret
