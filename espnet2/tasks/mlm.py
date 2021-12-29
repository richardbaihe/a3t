import argparse
import logging
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from pathlib import Path

import yaml
import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet.nets.pytorch_backend.tacotron2.decoder import Prenet as DecoderPrenet

from espnet.nets.pytorch_backend.transformer.encoder import MLMEncoder as TransformerEncoder
from espnet.nets.pytorch_backend.transformer.encoder import MLMDecoder as TransformerDecoder
from espnet.nets.pytorch_backend.conformer.encoder import MLMEncoder as ConformerEncoder
from espnet.nets.pytorch_backend.conformer.encoder import MLMDecoder as ConformerDecoder

from espnet2.tts.sedit.sedit_model import ESPnetMLMModel, ESPnetMLMDecoderModel,ESPnetMLMEncAsDecoderModel
from espnet2.train.abs_espnet_model import AbsESPnetModel

from espnet2.asr.postencoder.hugging_face_transformers_postencoder import (
    HuggingFaceTransformersPostEncoder,  # noqa: H301
)
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet2.tts.feats_extract.linear_spectrogram import LinearSpectrogram
from espnet2.tts.feats_extract.log_mel_fbank import LogMelFbank
from espnet2.tts.feats_extract.log_spectrogram import LogSpectrogram
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn, MLMCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import float_or_none
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding

feats_extractor_choices = ClassChoices(
    "feats_extract",
    classes=dict(
        fbank=LogMelFbank,
        spectrogram=LogSpectrogram,
        linear_spectrogram=LinearSpectrogram,
    ),
    type_check=AbsFeatsExtract,
    default="fbank",
)

normalize_choices = ClassChoices(
    "normalize",
    classes=dict(global_mvn=GlobalMVN),
    type_check=AbsNormalize,
    optional=True,
)

# transformer=MLMTransformerEncoder,
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        transformer=TransformerEncoder,
        conformer=ConformerEncoder,
    ),
    default="transformer",
)

decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        transformer=TransformerDecoder,
        transformer_encoder=TransformerDecoder,
        no_decoder=TransformerDecoder,
        conformer=ConformerDecoder
    ),
    default="transformer",
)

pre_decoder_choices = ClassChoices(
    "pre_decoder",
    classes=dict(
        linear=DecoderPrenet
    ),
    default="linear",
)



class MLMTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --feats_extractor and --feats_extractor_conf
        feats_extractor_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # # --decoder and --decoder_conf
        decoder_choices,
        pre_decoder_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["token_list"]

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )
        group.add_argument(
            "--odim",
            type=int_or_none,
            default=None,
            help="The number of dimension of output feature",
        )
        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetMLMModel),
            help="The keyword arguments for model class.",
        )
        group.add_argument(
            "--use_scaled_pos_enc",
            type=str2bool,
            default=False,
            help="use scaled pos or vanilla pos",
        )
        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--token_type",
            type=str_or_none,
            default=None,
            choices=[None, "bpe", "char", "word", "phn"],
            help="The text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )
        parser.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[None, "tacotron", "jaconv", "vietnamese"],
            default=None,
            help="Apply text cleaning",
        )
        parser.add_argument(
            "--g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="Specify g2p method if --token_type=phn",
        )
        parser.add_argument(
            "--speech_volume_normalize",
            type=float_or_none,
            default=None,
            help="Scale the maximum amplitude to the given value.",
        )
        parser.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The file path of rir scp file.",
        )
        parser.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="THe probability for applying RIR convolution.",
        )
        parser.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        parser.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability applying Noise adding.",
        )
        parser.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of noise decibel level.",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool, epoch=-1
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        # assert check_argument_types()
        # return CommonCollateFn(float_pad_value=0.0, int_pad_value=0)
        feats_extract_class = feats_extractor_choices.get_class(args.feats_extract)
        feats_extract = feats_extract_class(**args.feats_extract_conf)

        sega_emb = True if args.encoder_conf['input_layer'] == 'sega_mlm' else False
        if args.encoder_conf['selfattention_layer_type'] == 'longformer':
            attention_window = args.encoder_conf['attention_window']
            pad_speech = True if 'pre_speech_layer' in args.encoder_conf and args.encoder_conf['pre_speech_layer'] >0 else False
        else:
            attention_window=0
            pad_speech=False
        if epoch==-1:
            mlm_prob_factor = 1
        else:
            mlm_probs = [1.0, 1.0, 0.7, 0.6, 0.5]
            mlm_prob_factor = mlm_probs[epoch // 100]
        return MLMCollateFn(feats_extract, float_pad_value=0.0, int_pad_value=0,
        mlm_prob=args.model_conf['mlm_prob']*mlm_prob_factor,mean_phn_span=args.model_conf['mean_phn_span'],attention_window=attention_window,pad_speech=pad_speech,sega_emb=sega_emb)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        # assert check_argument_types()
        if args.use_preprocessor:
            retval = CommonPreprocessor(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
            )
        else:
            retval = None
        # assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ("speech","text","align_start","align_end")
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ()
        # assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetMLMModel:
        # assert check_argument_types()
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]

            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        # 1. feats_extract
        if args.odim is None:
            # Extract features in the model
            feats_extract_class = feats_extractor_choices.get_class(args.feats_extract)
            feats_extract = feats_extract_class(**args.feats_extract_conf)
            odim = feats_extract.output_size()
        else:
            # Give features from data-loader
            args.feats_extract = None
            args.feats_extract_conf = None
            feats_extract = None
            odim = args.odim

        # 3. Normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        pos_enc_class = ScaledPositionalEncoding if args.use_scaled_pos_enc else PositionalEncoding

        if "conformer" in [args.encoder, args.decoder]:
            conformer_self_attn_layer_type = args.encoder_conf['selfattention_layer_type']
            conformer_pos_enc_layer_type = args.encoder_conf['pos_enc_layer_type']
            conformer_rel_pos_type = "legacy"
            if conformer_rel_pos_type == "legacy":
                if conformer_pos_enc_layer_type == "rel_pos":
                    conformer_pos_enc_layer_type = "legacy_rel_pos"
                    logging.warning(
                        "Fallback to conformer_pos_enc_layer_type = 'legacy_rel_pos' "
                        "due to the compatibility. If you want to use the new one, "
                        "please use conformer_pos_enc_layer_type = 'latest'."
                    )
                if conformer_self_attn_layer_type == "rel_selfattn":
                    conformer_self_attn_layer_type = "legacy_rel_selfattn"
                    logging.warning(
                        "Fallback to "
                        "conformer_self_attn_layer_type = 'legacy_rel_selfattn' "
                        "due to the compatibility. If you want to use the new one, "
                        "please use conformer_pos_enc_layer_type = 'latest'."
                    )
            elif conformer_rel_pos_type == "latest":
                assert conformer_pos_enc_layer_type != "legacy_rel_pos"
                assert conformer_self_attn_layer_type != "legacy_rel_selfattn"
            else:
                raise ValueError(f"Unknown rel_pos_type: {conformer_rel_pos_type}")
            args.encoder_conf['selfattention_layer_type'] = conformer_self_attn_layer_type
            args.encoder_conf['pos_enc_layer_type'] = conformer_pos_enc_layer_type 
            args.decoder_conf['selfattention_layer_type'] = conformer_self_attn_layer_type
            args.decoder_conf['pos_enc_layer_type'] = conformer_pos_enc_layer_type 


        # 4. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        
        encoder = encoder_class(args.input_size,vocab_size=vocab_size, pos_enc_class=pos_enc_class,
        **args.encoder_conf)

        # # 5. Decoder
        if args.decoder != 'no_decoder':
            decoder_class = decoder_choices.get_class(args.decoder)
            decoder = decoder_class(
                idim=0,
                input_layer=None,
                **args.decoder_conf,
            )
        else:
            decoder = None

        # 8. Build model
        model = ESPnetMLMEncAsDecoderModel(
            feats_extract=feats_extract,
            odim=odim,
            normalize=normalize,
            encoder=encoder,
            decoder=decoder,
            token_list=token_list,
            **args.model_conf,
        )


        # 9. Initialize
        if args.init is not None:
            initialize(model, args.init)

        # assert check_return_type(model)
        return model


    @classmethod
    def build_model_from_file(
        cls,
        config_file: Union[Path, str] = None,
        model_file: Union[Path, str] = None,
        device: str = "cpu",
    ) -> Tuple[AbsESPnetModel, argparse.Namespace]:
        """Build model from the files.

        This method is used for inference or fine-tuning.

        Args:
            config_file: The yaml file saved when training.
            model_file: The model file saved when training.
            device: Device type, "cpu", "cuda", or "cuda:N".

        """
        # assert check_argument_types()
        if config_file is None:
            assert model_file is not None, (
                "The argument 'model_file' must be provided "
                "if the argument 'config_file' is not specified."
            )
            config_file = Path(model_file).parent / "config.yaml"
        else:
            config_file = Path(config_file)

        with config_file.open("r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        if 'ctc_weight' in args['model_conf'].keys():
            args['model_conf'].pop('ctc_weight')
        args = argparse.Namespace(**args)
        model = cls.build_model(args)
        if not isinstance(model, ESPnetMLMModel):
            raise RuntimeError(
                f"model must inherit {ESPnetMLMModel.__name__}, but got {type(model)}"
            )
        model.to(device)
        if model_file is not None:
            if device == "cuda":
                # NOTE(kamo): "cuda" for torch.load always indicates cuda:0
                #   in PyTorch<=1.4
                device = f"cuda:{torch.cuda.current_device()}"
            state_dict = torch.load(model_file, map_location=device)
            keys = list(state_dict.keys())
            for k in keys:
                if "encoder.embed" in k:
                    state_dict[k.replace("encoder.embed","encoder.speech_embed")] = state_dict.pop(k)
            model.load_state_dict(state_dict)

        return model, args
