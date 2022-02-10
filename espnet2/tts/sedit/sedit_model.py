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
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictor, DurationPredictorLoss
from espnet.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator



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
        decoder_seg_pos=False,
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
        self.decoder_seg_pos = decoder_seg_pos
        if lsm_weight > 50:
            self.l1_loss_func = torch.nn.MSELoss(reduce=False)
        else:
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
        speech, speech_lengths, text, text_lengths, masked_position, speech_mask, text_mask, speech_segment_pos, text_segment_pos, y_masks=None
    ) -> Dict[str, torch.Tensor]:
        return {"feats": speech, "feats_lengths": speech_lengths}

    def _forward(self, batch, speech_segment_pos,y_masks=None):
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

    def forward(
        self,
        speech, text, masked_position, speech_mask, text_mask, speech_segment_pos, text_segment_pos, y_masks=None,speech_lengths=None, text_lengths=None
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
        before_outs, after_outs, xs_pad, masked_position = self._forward(batch, speech_segment_pos,y_masks)


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
        speech, text, masked_position, speech_mask, text_mask, speech_segment_pos, text_segment_pos,
        span_boundary,
        y_masks=None,
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
            before,zs, _, _ = self._forward(
                batch, speech_segment_pos, y_masks=y_masks)
            if zs is None:
                zs = before
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

    def _add_first_frame_and_remove_last_frame(self, ys: torch.Tensor) -> torch.Tensor:
        ys_in = torch.cat(
            [ys.new_zeros((ys.shape[0], 1, ys.shape[2])), ys[:, :-1]], dim=1
        )
        return ys_in

class ESPnetMLMEncAsDecoderModel(ESPnetMLMModel):

    def _forward(self, batch, speech_segment_pos, y_masks=None):
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        speech_pad_placeholder = batch['speech_pad']
        # ys_in = self._add_first_frame_and_remove_last_frame(batch['speech_pad'])
        encoder_out, h_masks = self.encoder(**batch) # segment_emb
        if self.decoder is not None:
            # if self.decoder_seg_pos:
            #     zs, _ = self.decoder(encoder_out, h_masks.bool(),segment_emb)
            # else:
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

class ESPnetMLMTTSModel(ESPnetMLMModel):
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
        duration_predictor_layers = 0,
    ):
        super().__init__(token_list, odim, feats_extract, normalize, encoder, decoder, postnet_layers, postnet_chans, postnet_filts, ignore_id, lsm_weight, length_normalized_loss, report_cer, report_wer, sym_space, sym_blank, masking_schema, mean_phn_span, mlm_prob, dynamic_mlm_prob)
        adim = self.encoder._output_size
        self.duration_predictor = DurationPredictor(
            idim=adim,
            n_layers=duration_predictor_layers,
            n_chans=256,
            kernel_size=3,
            dropout_rate=0.1,
        )
        self.length_regulator = LengthRegulator()
        self.duration_criterion = DurationPredictorLoss(reduction="none")

        
    def _forward(self, batch,durations=None, is_inference=False, alpha=1.0):
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        #speech_pad_placeholder = batch['full_speech_pad']
        encoder_out, h_masks, segment_emb = self.encoder(**batch)
        speech_hidden_states = encoder_out[:,:batch['speech_pad'].shape[1],:].contiguous()
        if is_inference:
            d_outs = self.duration_predictor.inference(speech_hidden_states, batch['speech_mask'].squeeze(1))
            d_outs[d_outs.sum(dim=1).eq(0)] = 1
            d_outs_pad_text = torch.cat([d_outs,torch.ones_like(batch['text_pad'],dtype=d_outs.dtype)],axis=1)
            encoder_out = self.length_regulator(encoder_out, d_outs_pad_text, alpha)
            h_masks = self.length_regulator(h_masks.unsqueeze(-1), d_outs_pad_text, alpha).squeeze(-1)
        else:
            d_outs = self.duration_predictor(speech_hidden_states, batch['speech_mask'].squeeze(1))
            durations_pad_text = torch.cat([durations,torch.ones_like(batch['text_pad'])],axis=1)
            encoder_out = self.length_regulator(encoder_out, durations_pad_text)
            h_masks = self.length_regulator(h_masks.squeeze(1).unsqueeze(-1), durations_pad_text).squeeze(-1).unsqueeze(1)

        if self.decoder is not None:
            if self.decoder_seg_pos:
                zs, _ = self.decoder(encoder_out, h_masks.bool(),segment_emb)
            else:
                zs, _ = self.decoder(encoder_out, h_masks.bool())
        else:
            zs = encoder_out
        speech_hidden_states = zs[:,:-batch['text_pad'].shape[1], :].contiguous()
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
        return before_outs, after_outs, d_outs #, speech_pad_placeholder, batch['masked_position']

    def forward(
        self,
        speech, text, masked_position, speech_mask, text_mask, speech_segment_pos, text_segment_pos,durations, reordered_index, y_masks=None,speech_lengths=None, text_lengths=None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        batch_size = speech.shape[0]
        max_reduced_length = reordered_index.shape[1]

        reduced_masked_position = masked_position[torch.arange(batch_size).unsqueeze(1).repeat((1, max_reduced_length)).flatten(), reordered_index.flatten()].view(batch_size, max_reduced_length)

        reduced_speech = speech[torch.arange(batch_size).unsqueeze(1).repeat((1, max_reduced_length)).flatten(), reordered_index.flatten()].view(batch_size, max_reduced_length,-1)

        # reduced_speech_mask = speech_mask.squeeze(1)[torch.arange(batch_size).unsqueeze(1).repeat((1, max_reduced_length)).flatten(), reordered_index.flatten()].view(batch_size, max_reduced_length).unsqueeze(1)

        reduced_speech_segment_pos = speech_segment_pos[torch.arange(batch_size).unsqueeze(1).repeat((1, max_reduced_length)).flatten(),reordered_index.flatten()].view(batch_size, max_reduced_length)

        reduced_durations = durations[torch.arange(batch_size).unsqueeze(1).repeat((1, max_reduced_length)).flatten(), reordered_index.flatten()].view(batch_size, max_reduced_length)*speech_mask.squeeze(1)

        batch = dict(
            # full_speech_pad=speech,
            speech_pad=reduced_speech,
            text_pad=text,
            masked_position=reduced_masked_position,
            speech_mask=speech_mask,
            text_mask=text_mask,
            speech_segment_pos=reduced_speech_segment_pos,
            text_segment_pos=text_segment_pos,
        )
        before_outs, after_outs, d_outs  = self._forward(batch,reduced_durations)

        loss_mlm, loss_copy = self._calc_mlm_loss(
            before_outs,after_outs, speech, masked_position
        )
        duration_loss = self.duration_criterion(d_outs, reduced_durations)
        duration_loss = (duration_loss.view(-1) * reduced_masked_position.view(-1).float()).sum() \
                                            / (reduced_masked_position.float().sum() + 1e-10)

        loss = loss_mlm + duration_loss 
        if loss_copy is not None:
            loss += loss_copy

        stats = dict(
            loss=loss.detach(),
            loss_mlm=loss_mlm.detach() if loss_mlm is not None else None,
            loss_copy=loss_copy.detach() if loss_copy is not None else None,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def inference(
        self,
        speech, text, masked_position, speech_mask, text_mask, speech_segment_pos, text_segment_pos,
        durations, reordered_index,
        span_boundary,
        y_masks=None,
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
        
        batch_size = speech.shape[0]
        max_reduced_length = reordered_index.shape[1]

        reduced_masked_position = masked_position[torch.arange(batch_size).unsqueeze(1).repeat((1, max_reduced_length)).flatten(), reordered_index.flatten()].view(batch_size, max_reduced_length)

        reduced_speech = speech[torch.arange(batch_size).unsqueeze(1).repeat((1, max_reduced_length)).flatten(), reordered_index.flatten()].view(batch_size, max_reduced_length,-1)

        reduced_speech_segment_pos = speech_segment_pos[torch.arange(batch_size).unsqueeze(1).repeat((1, max_reduced_length)).flatten(),reordered_index.flatten()].view(batch_size, max_reduced_length)

        reduced_durations = durations[torch.arange(batch_size).unsqueeze(1).repeat((1, max_reduced_length)).flatten(), reordered_index.flatten()].view(batch_size, max_reduced_length)*speech_mask.squeeze(1)

        batch = dict(
            speech_pad=reduced_speech,
            text_pad=text,
            masked_position=reduced_masked_position,
            speech_mask=speech_mask,
            text_mask=text_mask,
            speech_segment_pos=reduced_speech_segment_pos,
            text_segment_pos=text_segment_pos,
        )
        

        outs = [batch['speech_pad'][:,:span_boundary[0]]]
        # ys = self._add_first_frame_and_remove_last_frame(batch['speech_pad'])
        z_cache = None
        if use_teacher_forcing:
            before,zs, _, _ = self._forward(batch)
            if zs is None:
                zs = before

            outs+=[zs[0][span_boundary[0]:span_boundary[1]]]
            outs+=[batch['speech_pad'][:,span_boundary[1]:]]
            return dict(feat_gen=outs)
        att_ws = torch.stack(att_ws, dim=0)
        outs += [batch['speech_pad'][:,span_boundary[1]:]]
        return dict(feat_gen=outs, att_w=att_ws)

