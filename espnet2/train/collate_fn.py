from typing import Collection
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type
import math
from espnet.nets.pytorch_backend.nets_utils import pad_list, make_non_pad_mask

class CommonCollateFn:
    """Functor class of common_collate_fn()"""

    def __init__(
        self,
        float_pad_value: Union[float, int] = 0.0,
        int_pad_value: int = -32768,
        not_sequence: Collection[str] = (),
    ):
        assert check_argument_types()
        self.float_pad_value = float_pad_value
        self.int_pad_value = int_pad_value
        self.not_sequence = set(not_sequence)

    def __repr__(self):
        return (
            f"{self.__class__}(float_pad_value={self.float_pad_value}, "
            f"int_pad_value={self.float_pad_value})"
        )

    def __call__(
        self, data: Collection[Tuple[str, Dict[str, np.ndarray]]]
    ) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        return common_collate_fn(
            data,
            float_pad_value=self.float_pad_value,
            int_pad_value=self.int_pad_value,
            not_sequence=self.not_sequence,
        )


def common_collate_fn(
    data: Collection[Tuple[str, Dict[str, np.ndarray]]],
    float_pad_value: Union[float, int] = 0.0,
    int_pad_value: int = -32768,
    not_sequence: Collection[str] = (),
) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    """Concatenate ndarray-list to an array and convert to torch.Tensor.

    Examples:
        >>> from espnet2.samplers.constant_batch_sampler import ConstantBatchSampler,
        >>> import espnet2.tasks.abs_task
        >>> from espnet2.train.dataset import ESPnetDataset
        >>> sampler = ConstantBatchSampler(...)
        >>> dataset = ESPnetDataset(...)
        >>> keys = next(iter(sampler)
        >>> batch = [dataset[key] for key in keys]
        >>> batch = common_collate_fn(batch)
        >>> model(**batch)

        Note that the dict-keys of batch are propagated from
        that of the dataset as they are.

    """
    assert check_argument_types()
    uttids = [u for u, _ in data]
    data = [d for _, d in data]

    assert all(set(data[0]) == set(d) for d in data), "dict-keys mismatching"
    assert all(
        not k.endswith("_lengths") for k in data[0]
    ), f"*_lengths is reserved: {list(data[0])}"

    output = {}
    for key in data[0]:
        # NOTE(kamo):
        # Each models, which accepts these values finally, are responsible
        # to repaint the pad_value to the desired value for each tasks.
        if data[0][key].dtype.kind == "i":
            pad_value = int_pad_value
        else:
            pad_value = float_pad_value

        array_list = [d[key] for d in data]

        # Assume the first axis is length:
        # tensor_list: Batch x (Length, ...)
        tensor_list = [torch.from_numpy(a) for a in array_list]
        # tensor: (Batch, Length, ...)
        tensor = pad_list(tensor_list, pad_value)
        output[key] = tensor

        # lens: (Batch,)
        if key not in not_sequence:
            lens = torch.tensor([d[key].shape[0] for d in data], dtype=torch.long)
            output[key + "_lengths"] = lens

    output = (uttids, output)
    assert check_return_type(output)
    return output


class MLMCollateFn:
    """Functor class of common_collate_fn()"""

    def __init__(
        self,
        feats_extract,
        float_pad_value: Union[float, int] = 0.0,
        int_pad_value: int = -32768,
        not_sequence: Collection[str] = (),
        mlm_prob: float=0.8,
        mean_phn_span: int=8,
        attention_window: int=0,
        pad_speech: bool=False,
        sega_emb: bool=False,
        duration_collect: bool=False

    ):
        self.mlm_prob=mlm_prob
        self.mean_phn_span=mean_phn_span
        self.feats_extract = feats_extract
        self.float_pad_value = float_pad_value
        self.int_pad_value = int_pad_value
        self.not_sequence = set(not_sequence)
        self.attention_window=attention_window
        self.pad_speech=pad_speech
        self.sega_emb=sega_emb
        self.duration_collect = duration_collect

    def __repr__(self):
        return (
            f"{self.__class__}(float_pad_value={self.float_pad_value}, "
            f"int_pad_value={self.float_pad_value})"
        )

    def __call__(
        self, data: Collection[Tuple[str, Dict[str, np.ndarray]]]
    ) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        return mlm_collate_fn(
            data,
            float_pad_value=self.float_pad_value,
            int_pad_value=self.int_pad_value,
            not_sequence=self.not_sequence,
            mlm_prob=self.mlm_prob, 
            mean_phn_span=self.mean_phn_span,
            feats_extract=self.feats_extract,
            attention_window=self.attention_window,
            pad_speech=self.pad_speech,
            sega_emb=self.sega_emb,
            duration_collect=self.duration_collect
        )


def mlm_collate_fn(
    data: Collection[Tuple[str, Dict[str, np.ndarray]]],
    float_pad_value: Union[float, int] = 0.0,
    int_pad_value: int = -32768,
    not_sequence: Collection[str] = (),
    mlm_prob: float = 0.8, 
    mean_phn_span: int = 8,
    feats_extract=None,
    attention_window: int = 0,
    pad_speech: bool=False,
    sega_emb: bool=False,
    duration_collect: bool=False,
) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    """Concatenate ndarray-list to an array and convert to torch.Tensor.

    Examples:
        >>> from espnet2.samplers.constant_batch_sampler import ConstantBatchSampler,
        >>> import espnet2.tasks.abs_task
        >>> from espnet2.train.dataset import ESPnetDataset
        >>> sampler = ConstantBatchSampler(...)
        >>> dataset = ESPnetDataset(...)
        >>> keys = next(iter(sampler)
        >>> batch = [dataset[key] for key in keys]
        >>> batch = common_collate_fn(batch)
        >>> model(**batch)

        Note that the dict-keys of batch are propagated from
        that of the dataset as they are.

    """
    uttids = [u for u, _ in data]
    data = [d for _, d in data]

    assert all(set(data[0]) == set(d) for d in data), "dict-keys mismatching"
    assert all(
        not k.endswith("_lengths") for k in data[0]
    ), f"*_lengths is reserved: {list(data[0])}"

    output = {}
    for key in data[0]:
        # NOTE(kamo):
        # Each models, which accepts these values finally, are responsible
        # to repaint the pad_value to the desired value for each tasks.
        if data[0][key].dtype.kind == "i":
            pad_value = int_pad_value
        else:
            pad_value = float_pad_value

        array_list = [d[key] for d in data]

        # Assume the first axis is length:
        # tensor_list: Batch x (Length, ...)
        tensor_list = [torch.from_numpy(a) for a in array_list]
        # tensor: (Batch, Length, ...)
        tensor = pad_list(tensor_list, pad_value)
        output[key] = tensor

        # lens: (Batch,)
        if key not in not_sequence:
            lens = torch.tensor([d[key].shape[0] for d in data], dtype=torch.long)
            output[key + "_lengths"] = lens

    feats, feats_lengths = feats_extract(output["speech"], output["speech_lengths"])
    batch_size = feats.shape[0]
    if 'text' not in output:
        text=torch.zeros_like(feats_lengths.unsqueeze(-1))-2
        text_lengths=torch.zeros_like(feats_lengths)+1
        max_tlen=1
        align_start=torch.zeros_like(text)
        align_end=torch.zeros_like(text)
        align_start_lengths=torch.zeros_like(feats_lengths)
        align_end_lengths=torch.zeros_like(feats_lengths)
        sega_emb=False
        mean_phn_span = 0
        mlm_prob = 0.15
    else:
        text, text_lengths = output["text"], output["text_lengths"]
        align_start, align_start_lengths, align_end, align_end_lengths = output["align_start"], output["align_start_lengths"], output["align_end"], output["align_end_lengths"]
        align_start = torch.floor(feats_extract.fs*align_start/feats_extract.hop_length).int()
        align_end = torch.floor(feats_extract.fs*align_end/feats_extract.hop_length).int()
        max_tlen = max(text_lengths).item()
    max_slen = max(feats_lengths).item()
    speech_pad = feats[:, : max_slen]
    if attention_window>0 and pad_speech:
        speech_pad,max_slen = pad_to_longformer_att_window(speech_pad, max_slen, max_slen, attention_window)
    max_len = max_slen + max_tlen
    if attention_window>0:
        text_pad, max_tlen = pad_to_longformer_att_window(text, max_len, max_tlen, attention_window)
    else:
        text_pad = text
    text_mask = make_non_pad_mask(text_lengths.tolist(), text_pad, length_dim=1).to(text_pad.device).unsqueeze(-2)
    if attention_window>0:
        text_mask = text_mask*2 
    speech_mask = make_non_pad_mask(feats_lengths.tolist(), speech_pad[:,:,0], length_dim=1).to(speech_pad.device).unsqueeze(-2)
    span_boundary = None
    if 'span_boundary' in output.keys():
        span_boundary = output['span_boundary']

    masked_position, _ = phones_masking(
            speech_pad,
            speech_mask,
            align_start,
            align_end,
            align_start_lengths,
            mlm_prob,
            mean_phn_span,
            span_boundary)

    output_dict = {}
    if duration_collect and 'text' in output:
        reordered_index, speech_segment_pos,text_segment_pos, durations,feats_lengths = get_segment_pos_reduce_duration(speech_pad, text_pad, align_start, align_end, align_start_lengths,sega_emb, masked_position, feats_lengths)
        speech_mask = make_non_pad_mask(feats_lengths.tolist(), speech_pad[:,:reordered_index.shape[1],0], length_dim=1).to(speech_pad.device).unsqueeze(-2)
        output_dict['durations'] = durations
        output_dict['reordered_index'] = reordered_index
    else:
        speech_segment_pos, text_segment_pos = get_segment_pos(speech_pad, text_pad, align_start, align_end, align_start_lengths,sega_emb)
    
    output_dict['speech'] = speech_pad
    output_dict['text'] = text_pad
    output_dict['masked_position'] = masked_position
    output_dict['speech_mask'] = speech_mask
    output_dict['text_mask'] = text_mask
    output_dict['speech_segment_pos'] = speech_segment_pos
    output_dict['text_segment_pos'] = text_segment_pos
    # output_dict['y_masks'] = y_masks
    output_dict['speech_lengths'] = output["speech_lengths"]
    output_dict['text_lengths'] = text_lengths
    output = (uttids, output_dict)
    # assert check_return_type(output)
    return output


def get_segment_pos_reduce_duration(speech_pad, text_pad, align_start, align_end, 
align_start_lengths,sega_emb, masked_position,feats_lengths):
    bz, speech_len, _ = speech_pad.size()
    text_segment_pos = torch.zeros_like(text_pad)
    speech_segment_pos = torch.zeros((bz, speech_len),dtype=text_pad.dtype, device=text_pad.device)
    
    reordered_index=torch.zeros(bz, speech_len, dtype=align_start_lengths.dtype)

    durations = torch.ones((bz, speech_len), dtype=align_start_lengths.dtype)
    max_reduced_length = 0
    if not sega_emb:
        return speech_pad, masked_position, speech_segment_pos,text_segment_pos, durations
    for idx in range(bz):
        first_idx = []
        last_idx = []
        align_length = align_start_lengths[idx].item()
        for j in range(align_length):
            s,e = align_start[idx][j].item(), align_end[idx][j].item()
            if j==0:
                if torch.sum(masked_position[idx][0:s]) ==0:
                    first_idx.extend(range(0,s))
                else:
                    first_idx.extend([0])
                    last_idx.extend(range(1,s))
            if torch.sum(masked_position[idx][s:e]) ==0:
                first_idx.extend(range(s,e))
            else:
                first_idx.extend([s])
                last_idx.extend(range(s+1,e))
                durations[idx][s] = e-s
            speech_segment_pos[idx][s:e] = j+1
            text_segment_pos[idx][j] = j+1
        max_reduced_length = max(len(first_idx)+feats_lengths[idx].item()-e,max_reduced_length)
        first_idx.extend(range(e,speech_len))
        reordered_index[idx] = torch.tensor((first_idx+last_idx),dtype=align_start_lengths.dtype)
        feats_lengths[idx] = len(first_idx)
    reordered_index = reordered_index[:,:max_reduced_length]

    return reordered_index, speech_segment_pos,text_segment_pos, durations,feats_lengths

def get_segment_pos(speech_pad, text_pad, align_start, align_end, align_start_lengths,sega_emb):
    bz, speech_len, _ = speech_pad.size()
    text_segment_pos = torch.zeros_like(text_pad)
    speech_segment_pos = torch.zeros((bz, speech_len),dtype=text_pad.dtype, device=text_pad.device)
    if not sega_emb:
        return speech_segment_pos, text_segment_pos
    for idx in range(bz):
        align_length = align_start_lengths[idx].item()
        for j in range(align_length):
            s,e = align_start[idx][j].item(), align_end[idx][j].item()
            speech_segment_pos[idx][s:e] = j+1
            text_segment_pos[idx][j] = j+1
        
    return speech_segment_pos, text_segment_pos


def phones_masking(xs_pad, src_mask, align_start, align_end, align_start_lengths, mlm_prob, mean_phn_span, span_boundary=None):
    bz, sent_len, _ = xs_pad.size()
    mask_num_lower = math.ceil(sent_len * mlm_prob)
    masked_position = np.zeros((bz, sent_len))
    y_masks = None
    # y_masks = torch.ones(bz,sent_len,sent_len,device=xs_pad.device,dtype=xs_pad.dtype)
    # tril_masks = torch.tril(y_masks)
    if mlm_prob == 1.0:
        masked_position += 1
        # y_masks = tril_masks
    elif mean_phn_span == 0:
        # only speech 
        length = sent_len
        mean_phn_span = min(length*mlm_prob//3, 50)
        masked_phn_indices = random_spans_noise_mask(length,mlm_prob, mean_phn_span).nonzero()
        masked_position[:,masked_phn_indices]=1
    else:
        for idx in range(bz):
            if span_boundary is not None:
                for s,e in zip(span_boundary[idx][::2], span_boundary[idx][1::2]):
                    masked_position[idx, s:e] = 1

                    # y_masks[idx, :, s:e] = tril_masks[idx, :, s:e]
                    # y_masks[idx, e:, s:e ] = 0
            else:
                length = align_start_lengths[idx].item()
                if length<2:
                    continue
                masked_phn_indices = random_spans_noise_mask(length,mlm_prob, mean_phn_span).nonzero()
                masked_start = align_start[idx][masked_phn_indices].tolist()
                masked_end = align_end[idx][masked_phn_indices].tolist()
                for s,e in zip(masked_start, masked_end):
                    masked_position[idx, s:e] = 1
                    # y_masks[idx, :, s:e] = tril_masks[idx, :, s:e]
                    # y_masks[idx, e:, s:e ] = 0
    non_eos_mask = src_mask.view(xs_pad.size()[:2]).float().cpu().numpy()
    masked_position = masked_position * non_eos_mask
    # y_masks = src_mask & y_masks.bool()

    return torch.BoolTensor(masked_position).to(xs_pad.device), y_masks

def random_spans_noise_mask(length, mlm_prob, mean_phn_span):

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

    num_noise_tokens = int(np.round(length * mlm_prob))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(np.round(num_noise_tokens / mean_phn_span))

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

def pad_to_longformer_att_window(text, max_len, max_tlen,attention_window):
    round = max_len % attention_window
    if round != 0:
        max_tlen += (attention_window - round)
        n_batch = text.shape[0]
        text_pad = text.new_zeros(n_batch, max_tlen, *text[0].size()[1:])
        for i in range(n_batch):
            text_pad[i, : text[i].size(0)] = text[i]
    else:
        text_pad = text[:, : max_tlen]
    return text_pad, max_tlen
