from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import math
import numpy as np
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, pad_list
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention, LongformerAttention
from espnet2.train.collate_fn import phones_masking, get_segment_pos,pad_to_longformer_att_window
if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield
class ESPnetMLMModel(AbsESPnetModel):
    def __init__(
        self,
        token_list: Union[Tuple[str, ...], List[str]],
        odim: int,
        feats_extract: Optional[AbsFeatsExtract],
        normalize: Optional[AbsNormalize],
        encoder: torch.nn.Module,
        decoder: Optional[torch.nn.Module],
        postnet_layers: int = 0,
        postnet_chans: int = 0,
        postnet_filts: int = 0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        masking_schema: str = "span",
        mean_phn_span: int = 3,
        mlm_prob: float = 0.25,
        dynamic_mlm_prob = False,
    ):

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.odim = odim
        self.ignore_id = ignore_id
        self.token_list = token_list.copy()

        self.normalize = normalize
        self.encoder = encoder

        self.decoder = decoder

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.error_calculator = None

        self.feats_extract = feats_extract
        self.mlm_weight = 1.0
        self.mlm_prob = mlm_prob
        self.mlm_layer = 12
        self.finetune_wo_mlm =True
        self.max_span = 50
        self.min_span = 4
        self.mean_phn_span = mean_phn_span
        self.masking_schema = masking_schema
        if self.decoder is None or not (hasattr(self.decoder, 'output_layer') and self.decoder.output_layer is not None):
            self.sfc = torch.nn.Linear(self.encoder._output_size, odim)
        else:
            self.sfc=None
        self.l1_loss_func = torch.nn.L1Loss(reduce=False)
        # self.l2_loss_func = torch.nn.MSELoss(reduce=False)
        # self.loss_func = torch.nn.MSELoss(reduce=False)
        self.postnet = (
            None
            if postnet_layers == 0
            else Postnet(
                idim=self.encoder._output_size,
                odim=odim,
                n_layers=postnet_layers,
                n_chans=postnet_chans,
                n_filts=postnet_filts,
                use_batch_norm=True,
                dropout_rate=0.5,
            )
        )

    def collect_feats(self,
        speech, speech_lengths, text, text_lengths, masked_position, speech_mask, text_mask, speech_segment_pos, text_segment_pos, y_masks
    ) -> Dict[str, torch.Tensor]:
        return {"feats": speech, "feats_lengths": speech_lengths}

    # def collect_feats(
    #     self,
    #     speech: torch.Tensor, 
    #     speech_lengths: torch.Tensor,
    #     text: torch.Tensor, 
    #     text_lengths: torch.Tensor,
    #     align_start: torch.Tensor,
    #     align_start_lengths: torch.Tensor,
    #     align_end: torch.Tensor,
    #     align_end_lengths: torch.Tensor,
    # ) -> Dict[str, torch.Tensor]:
    #     if self.feats_extract:
    #         feats, feats_lengths = self.feats_extract(speech, speech_lengths)
    #     else:
    #         # Generate dummy stats if extract_feats_in_collect_stats is False
    #         logging.warning(
    #             "Generating dummy stats for feats and feats_lengths, "
    #             "because encoder_conf.extract_feats_in_collect_stats is "
    #             f"{self.feats_extract}"
    #         )
    #         feats, feats_lengths = speech, speech_lengths
    #     return {"feats": feats, "feats_lengths": feats_lengths}


    def _forward(self, batch, y_masks, speech_segment_pos):
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        speech_pad_placeholder = batch['speech_pad']
        if self.decoder is not None:
            ys_in = self._add_first_frame_and_remove_last_frame(batch['speech_pad'])
        encoder_out, h_masks = self.encoder(**batch)
        if self.decoder is not None:
            zs, _ = self.decoder(ys_in, y_masks, encoder_out, h_masks.bool(), self.encoder.segment_emb(speech_segment_pos))
            speech_hidden_states = zs
        else:
            speech_hidden_states = encoder_out[:,:batch['speech_pad'].shape[1], :].contiguous()
        if self.sfc is not None:
            before_outs = self.sfc(speech_hidden_states).view(
            speech_hidden_states.size(0), -1, self.odim)
        else:
            before_outs = speech_hidden_states
        if self.postnet is not None:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)
        else:
            after_outs = None
        return before_outs, after_outs, speech_pad_placeholder, batch['masked_position']

    # def forward(
    #     self,
    #     speech: torch.Tensor, 
    #     speech_lengths: torch.Tensor,
    #     text: torch.Tensor, 
    #     text_lengths: torch.Tensor,
    #     align_start: torch.Tensor,
    #     align_start_lengths: torch.Tensor,
    #     align_end: torch.Tensor,
    #     align_end_lengths: torch.Tensor,
    # ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

    #     speech_pad, text_pad, masked_position, speech_mask, text_mask, speech_segment_pos, text_segment_pos, y_masks = self.prepare_features(speech, speech_lengths, text, text_lengths, align_start, align_end, align_start_lengths)
        
    #     batch_size = speech_pad.shape[0]

    #     batch = dict(
    #         speech_pad=speech_pad,
    #         text_pad=text_pad,
    #         masked_position=masked_position,
    #         speech_mask=speech_mask,
    #         text_mask=text_mask,
    #         speech_segment_pos=speech_segment_pos,
    #         text_segment_pos=text_segment_pos,
    #     )
    #     before_outs, after_outs, xs_pad, masked_position = self._forward(batch, y_masks, speech_segment_pos)


    #     loss_mlm, loss_copy = self._calc_mlm_loss(
    #         before_outs,after_outs, xs_pad, masked_position
    #     )
    #     loss = loss_mlm + loss_copy if loss_copy is not None else loss_mlm 

    #     stats = dict(
    #         loss=loss.detach(),
    #         loss_mlm=loss_mlm.detach() if loss_mlm is not None else None,
    #         loss_copy=loss_copy.detach() if loss_copy is not None else None,
    #     )

    #     # force_gatherable: to-device and to-tensor if scalar for DataParallel
    #     loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
    #     return loss, stats, weight

    def forward(
        self,
        speech, text, masked_position, speech_mask, text_mask, speech_segment_pos, text_segment_pos, y_masks,speech_lengths=None, text_lengths=None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        batch_size = speech.shape[0]

        batch = dict(
            speech_pad=speech,
            text_pad=text,
            masked_position=masked_position,
            speech_mask=speech_mask,
            text_mask=text_mask,
            speech_segment_pos=speech_segment_pos,
            text_segment_pos=text_segment_pos,
        )
        before_outs, after_outs, xs_pad, masked_position = self._forward(batch, y_masks, speech_segment_pos)


        loss_mlm, loss_copy = self._calc_mlm_loss(
            before_outs,after_outs, xs_pad, masked_position
        )
        loss = loss_mlm + loss_copy if loss_copy is not None else loss_mlm 

        stats = dict(
            loss=loss.detach(),
            loss_mlm=loss_mlm.detach() if loss_mlm is not None else None,
            loss_copy=loss_copy.detach() if loss_copy is not None else None,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def prepare_features(self, speech, speech_lengths, text, text_lengths, align_start, align_end, align_start_lengths, span_boundary=[]):
        sega_emb = True if hasattr(self.encoder, 'segment_emb') else False
        with autocast(False):
            # 1. Extract feats
            if self.feats_extract is not None:
                feats, feats_lengths = self.feats_extract(speech, speech_lengths)
            else:
                # Use precalculated feats (feats_type != raw case)
                feats, feats_lengths = speech, speech_lengths

            align_start = torch.floor(self.feats_extract.fs*align_start/self.feats_extract.hop_length).int()
            align_end = torch.floor(self.feats_extract.fs*align_end/self.feats_extract.hop_length).int()
        max_tlen = max(text_lengths).item()
        max_slen = max(feats_lengths).item()
        speech_pad = feats[:, : max_slen]
        pad_speech = self.encoder.pre_speech_layer>0
        if hasattr(self.encoder, 'attention_window'):
            attention_window = self.encoder.attention_window
        else:
            attention_window=0
        if attention_window>0 and pad_speech:
            speech_pad,max_slen = pad_to_longformer_att_window(speech_pad, max_slen, max_slen,attention_window)
        max_len = max_slen + max_tlen
        if attention_window>0:
            text_pad, max_tlen = pad_to_longformer_att_window(text, max_len, max_tlen, attention_window)
        else:
            text_pad = text
        text_mask = make_non_pad_mask(text_lengths.tolist(), text_pad, length_dim=1).to(text_pad.device).unsqueeze(-2)
        if attention_window>0:
            text_mask = text_mask*2 
        speech_mask = make_non_pad_mask(feats_lengths.tolist(), speech_pad[:,:,0], length_dim=1).to(speech_pad.device).unsqueeze(-2)
        # # 5*1171*80
        # speech_pad_placeholder = speech_pad.detach().clone()
        # # 5*1*1171
        # speech_mask_placeholder = speech_mask
        # text_mask_placeholder = text_mask
        masked_position, y_masks = phones_masking(
            speech_pad,
            speech_mask,
            align_start,
            align_end,
            align_start_lengths,
            self.mlm_prob,
            self.mean_phn_span,
            span_boundary=span_boundary)

        speech_segment_pos, text_segment_pos = get_segment_pos(speech_pad, text_pad, align_start, align_end, align_start_lengths, sega_emb)

        return speech_pad, text_pad, masked_position, speech_mask_placeholder, text_mask_placeholder, speech_segment_pos, text_segment_pos, y_masks

    def inference(
        self,
        speech, text, masked_position, speech_mask, text_mask, speech_segment_pos, text_segment_pos, y_masks,
        span_boundary,
        speech_lengths=None, text_lengths=None,
        feats: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 10.0,
        use_teacher_forcing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        
        # speech_pad, text_pad, masked_position, speech_mask, text_mask, speech_segment_pos, text_segment_pos, y_masks = self.prepare_features(speech, speech_lengths, text, text_lengths, align_start, align_end, align_start_lengths,span_boundary)

        batch = dict(
            speech_pad=speech,
            text_pad=text,
            masked_position=masked_position,
            speech_mask=speech_mask,
            text_mask=text_mask,
            speech_segment_pos=speech_segment_pos,
            text_segment_pos=text_segment_pos,
        )
        

        # # inference with teacher forcing
        # hs, h_masks = self.encoder(**batch)

        outs = [batch['speech_pad'][:,:span_boundary[0]]]
        # ys = self._add_first_frame_and_remove_last_frame(batch['speech_pad'])
        z_cache = None
        if use_teacher_forcing:
            _,zs, _, _ = self._forward(
                batch, y_masks, speech_segment_pos)
            # if self.sfc is not None:
            #     outputs = self.sfc(zs.reshape(-1, self.encoder._output_size))
            #     zs = outputs.unsqueeze(0)
            outs+=[zs[0][span_boundary[0]:span_boundary[1]]]
            outs+=[batch['speech_pad'][:,span_boundary[1]:]]
            return dict(feat_gen=outs)
        else:
            # forward decoder step-by-step
            for idx in range(span_boundary[0], span_boundary[1]):
                # calculate output and stop prob at idx-th step
                # y_masks = subsequent_mask(idx).unsqueeze(0).to(x.device)
                idx = max(0, idx-1)
                z, z_cache = self.decoder.forward_one_step(
                    ys, y_masks, hs, h_masks,self.encoder.segment_emb(speech_segment_pos), cache=z_cache, idx=idx
                )  # (B, adim)
                outs += [
                    z.view(-1, self.odim)
                ]  # [(r, odim), ...]

                # update next inputs
                ys[:,idx+1] = outs[-1][-1].view(1, 1, self.odim)

                # get attention weights
                att_ws_ = []
                for name, m in self.named_modules():
                    if isinstance(m, MultiHeadedAttention) and "src" in name:
                        att_ws_ += [m.attn[0, :, idx+1-span_boundary[0]].unsqueeze(1)]  # [(#heads, 1, T),...]
                if idx+1 == span_boundary[0]:
                    att_ws = att_ws_
                else:
                    # [(#heads, l, T), ...]
                    att_ws = [
                        torch.cat([att_w, att_w_], dim=1)
                        for att_w, att_w_ in zip(att_ws, att_ws_)
                    ]
            # concatenate attention weights -> (#layers, #heads, T_feats, T_text)
        att_ws = torch.stack(att_ws, dim=0)
        outs += [batch['speech_pad'][:,span_boundary[1]:]]
        return dict(feat_gen=outs, att_w=att_ws)


    def _calc_mlm_loss(
        self,
        before_outs: torch.Tensor,
        after_outs: torch.Tensor,
        xs_pad: torch.Tensor,
        masked_position: torch.Tensor
    ):
        true_label_position = (torch.rand(masked_position.size()) < (self.mlm_prob * .15)).to(xs_pad.device)
        # mlm_loss_position = (true_label_position + masked_position) > 0
        mlm_loss_position = masked_position>0
        copy_loss_position = true_label_position>0
        loss = self.l1_loss_func(before_outs.view(-1, self.odim), 
                                            xs_pad.view(-1, self.odim)).sum(dim=-1)
        if after_outs is not None:
            loss += self.l1_loss_func(after_outs.view(-1, self.odim), 
                                                xs_pad.view(-1, self.odim)).sum(dim=-1)
        loss_mlm = (loss * mlm_loss_position.view(-1).float()).sum() \
                                            / (mlm_loss_position.float().sum() + 1e-10)

        loss_copy = None
        return loss_mlm, loss_copy

    # def phones_masking(self, xs_pad, src_mask, align_start, align_end, align_start_lengths, mask_prob, max_span=5, min_span=1, span_boundary=[]):
    #     assert max_span >= min_span
    #     bz, sent_len, _ = xs_pad.size()
    #     mask_num_lower = math.ceil(sent_len * mask_prob)
    #     masked_position = np.zeros((bz, sent_len))
    #     y_masks = torch.ones(bz,sent_len,sent_len,device=xs_pad.device,dtype=xs_pad.dtype)
    #     tril_masks = torch.tril(y_masks)
    #     for idx in range(bz):
    #         if len(span_boundary)>0:
    #             for s,e in zip(span_boundary[::2], span_boundary[1::2]):
    #                 masked_position[idx, s:e] = 1
    #                 span_boundary.extend([s,e])
    #                 y_masks[idx, :, s:e] = tril_masks[idx, :, s:e]
    #                 y_masks[idx, e:, s:e ] = 0
    #         else:
    #             length = align_start_lengths[idx].item()
    #             if length<2:
    #                 continue
    #             masked_phn_indices = self.random_spans_noise_mask(length).nonzero()
    #             masked_start = align_start[idx][masked_phn_indices].tolist()
    #             masked_end = align_end[idx][masked_phn_indices].tolist()
    #             for s,e in zip(masked_start, masked_end):
    #                 masked_position[idx, s:e] = 1
    #                 y_masks[idx, :, s:e] = tril_masks[idx, :, s:e]
    #                 y_masks[idx, e:, s:e ] = 0
    #     non_eos_mask = src_mask.view(xs_pad.size()[:2]).float().cpu().numpy()
    #     # non_eos_mask = np.concatenate((non_eos_mask[:,1:], np.zeros(non_eos_mask[:,:1].shape, dtype=int)), axis=1)
    #     masked_position = masked_position * non_eos_mask
    #     y_masks = src_mask & y_masks.bool()

    #     return torch.BoolTensor(masked_position).to(xs_pad.device), y_masks

    def random_masking(self, xs_pad, src_mask, mask_prob):
        masked_position = np.random.rand(*xs_pad.size()[:2]) < mask_prob
        # no sos and eos
        masked_position[:,0] = 0
        non_eos_mask = src_mask.view(xs_pad.size()[:2]).float().cpu().numpy()
        non_eos_mask = np.concatenate((non_eos_mask[:,1:], np.zeros(non_eos_mask[:,:1].shape, dtype=int)), axis=1)
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
        masked_position = np.zeros((bz, sent_len))
        for idx in range(bz):
            while np.sum(masked_position[idx]) < mask_num_lower:
                span_start = np.random.choice(sent_len)
                span_len = np.random.choice(np.arange(min_span, max_span+1), p=len_distrib)
                while masked_position[idx, span_start] > 0 or span_start+span_len-1 >= sent_len or masked_position[idx, span_start+span_len-1] > 0:
                    span_start = np.random.choice(sent_len)
                    span_len = np.random.choice(np.arange(min_span, max_span+1), p=len_distrib)
                masked_position[idx, span_start:span_start+span_len] = 1
        non_eos_mask = src_mask.view(xs_pad.size()[:2]).float().cpu().numpy()
        # non_eos_mask = np.concatenate((non_eos_mask[:,1:], np.zeros(non_eos_mask[:,:1].shape, dtype=int)), axis=1)
        masked_position = masked_position * non_eos_mask
        return torch.BoolTensor(masked_position).to(xs_pad.device)
    
    def _add_first_frame_and_remove_last_frame(self, ys: torch.Tensor) -> torch.Tensor:
        ys_in = torch.cat(
            [ys.new_zeros((ys.shape[0], 1, ys.shape[2])), ys[:, :-1]], dim=1
        )
        return ys_in

    # def pad_to_longformer_att_window(self, text, max_len, max_tlen):
    #     round = max_len % self.encoder.attention_window
    #     if round != 0:
    #         max_tlen += (self.encoder.attention_window - round)
    #         n_batch = text.shape[0]
    #         text_pad = text.new_zeros(n_batch, max_tlen, *text[0].size()[1:])
    #         for i in range(n_batch):
    #             text_pad[i, : text[i].size(0)] = text[i]
    #     else:
    #         text_pad = text[:, : max_tlen]
    #     return text_pad, max_tlen

    # def get_segment_pos(self, speech_pad, text_pad, align_start, align_end, align_start_lengths):
    #     bz, speech_len, _ = speech_pad.size()
    #     text_segment_pos = torch.zeros_like(text_pad)
    #     speech_segment_pos = torch.zeros((bz, speech_len),dtype=text_pad.dtype, device=text_pad.device)
    #     for idx in range(bz):
    #         align_length = align_start_lengths[idx].item()
    #         for j in range(align_length):
    #             s,e = align_start[idx][j].item(), align_end[idx][j].item()
    #             speech_segment_pos[idx][s:e] = j+1
    #             text_segment_pos[idx][j] = j+1
            
    #     return speech_segment_pos, text_segment_pos

    def random_spans_noise_mask(self, length):

        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.mlm_prob))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_phn_span))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

class ESPnetMLMDecoderModel(ESPnetMLMModel):
    def nothing(self):
        return 0

class ESPnetMLMEncAsDecoderModel(ESPnetMLMModel):

    def _forward(self, batch, y_masks, speech_segment_pos):
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        speech_pad_placeholder = batch['speech_pad']
        # ys_in = self._add_first_frame_and_remove_last_frame(batch['speech_pad'])
        encoder_out, h_masks = self.encoder(**batch)
        if self.decoder is not None:
            zs, _ = self.decoder(encoder_out, h_masks.bool())
        else:
            zs = encoder_out
        speech_hidden_states = zs[:,:batch['speech_pad'].shape[1], :].contiguous()
        if self.sfc is not None:
            before_outs = self.sfc(speech_hidden_states).view(
            speech_hidden_states.size(0), -1, self.odim)
        else:
            before_outs = speech_hidden_states
        if self.postnet is not None:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)
        else:
            after_outs = None
        return before_outs, after_outs, speech_pad_placeholder, batch['masked_position']
