from typing import Any
from typing import Dict
from typing import Union

import logging
import torch
import torch.nn
import torch.optim


def filter_state_dict(
    dst_state: Dict[str, Union[float, torch.Tensor]],
    src_state: Dict[str, Union[float, torch.Tensor]],
):
    """Filter name, size mismatch instances between dicts.

    Args:
        dst_state: reference state dict for filtering
        src_state: target state dict for filtering

    """
    match_state = {}
    for key, value in src_state.items():
        if key in dst_state:
            if (dst_state[key].size() == src_state[key].size()):
                match_state[key] = value
            else:
                logging.warning(
                    f"Filter out {key} from pretrained dict"
                    + " because of size"
                    + f"({dst_state[key].size()}-{src_state[key].size()})"
                )
        else:
            logging.warning(
                f"Filter out {key} from pretrained dict"
                + " because of name"
            )
    return match_state

def load_phns_emb_reorderring():
    
    mfa_vocab = ["<blank>","<unk>","AH0","T","N","sp","D","S","R","L","IH1","DH","AE1","M","EH1","K","Z","W","HH","ER0","AH1","IY1","P","V","F","B","AY1","IY0","EY1","AA1","AO1","UW1","IH0","OW1","NG","G","SH","ER1","Y","TH","AW1","CH","UH1","IH2","JH","OW0","EH2","OY1","AY2","EH0","EY2","UW0","AE2","AA2","OW2","AH2","ZH","AO2","IY2","AE0","UW2","AY0","AA0","AO0","AW2","EY0","UH2","ER2","OY2","UH0","AW0","OY0","<sos/eos>"]
    target_vocab = ["<blank>","<unk>","..","OY0","UH0","AW0","!","OY2","?","UH2","ER2","''","AA0","IY2","AW2","AY0","AH2","UW2","AE0","OW2","ZH","AO2","EY0","OY1","EH0","UW0","AA2","AY2","AE2","IH2","AO0","EY2","OW0","EH2","UH1","TH","AW1","Y","JH","CH","ER1","G","NG","SH","OW1",".","AY1","EY1","AO1","IY0","UW1","IY1","HH","B","AA1",",","F","ER0","V","AH1","AE1","P","W","EH1","M","IH0","IH1","Z","K","DH","L","R","S","D","T","N","AH0","<sos/eos>"]
    permutated_index = []
    for i, phn in enumerate(target_vocab):
        if phn in mfa_vocab:
            permutated_index.append(mfa_vocab.index(phn))
        else:
            permutated_index.append(mfa_vocab.index("sp"))
    return permutated_index


def load_pretrained_model(
    init_param: str,
    model: torch.nn.Module,
    ignore_init_mismatch: bool,
    map_location: str = "cpu",
):
    """Load a model state and set it to the model.

    Args:
        init_param: <file_path>:<src_key>:<dst_key>:<exclude_Keys>

    Examples:
        >>> load_pretrained_model("somewhere/model.pth", model)
        >>> load_pretrained_model("somewhere/model.pth:decoder:decoder", model)
        >>> load_pretrained_model("somewhere/model.pth:decoder:decoder:", model)
        >>> load_pretrained_model(
        ...     "somewhere/model.pth:decoder:decoder:decoder.embed", model
        ... )
        >>> load_pretrained_model("somewhere/decoder.pth::decoder", model)
    """
    sps = init_param.split(":", 4)
    if len(sps) == 4:
        path, src_key, dst_key, excludes = sps
    elif len(sps) == 3:
        path, src_key, dst_key = sps
        excludes = None
    elif len(sps) == 2:
        path, src_key = sps
        dst_key, excludes = None, None
    else:
        (path,) = sps
        src_key, dst_key, excludes = None, None, None
    if src_key == "":
        src_key = None
    if dst_key == "":
        dst_key = None

    if dst_key is None:
        obj = model
    else:

        def get_attr(obj: Any, key: str):
            """Get an nested attribute.

            >>> class A(torch.nn.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.linear = torch.nn.Linear(10, 10)
            >>> a = A()
            >>> assert A.linear.weight is get_attr(A, 'linear.weight')

            """
            if key.strip() == "":
                return obj
            for k in key.split("."):
                obj = getattr(obj, k)
            return obj

        obj = get_attr(model, dst_key)

    src_state = torch.load(path, map_location=map_location)
    if excludes is not None:
        for e in excludes.split(","):
            src_state = {k: v for k, v in src_state.items() if not k.startswith(e)}

    if src_key is not None:
        src_state = {
            k[len(src_key) + 1 :]: v
            for k, v in src_state.items()
            if k.startswith(src_key)
        }

    dst_state = obj.state_dict()
    if 'encoder.text_embed.0.weight' in src_state and 'encoder.embed.0.weight' in dst_state:
        permutated_index = load_phns_emb_reorderring()
        src_state['encoder.embed.0.weight'] = src_state['encoder.text_embed.0.weight'][permutated_index,:]
    if ignore_init_mismatch:
        src_state = filter_state_dict(dst_state, src_state)
    dst_state.update(src_state)
    obj.load_state_dict(dst_state)
