import argparse
import logging
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,  # noqa: H301
)
from espnet2.asr.decoder.transformer_decoder import DynamicConvolutionTransformerDecoder
from espnet2.asr.decoder.transformer_decoder import (
    LightweightConvolution2DTransformerDecoder,  # noqa: H301
)
from espnet2.asr.decoder.transformer_decoder import (
    LightweightConvolutionTransformerDecoder,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.decoder import SegDecoder as TransformerDecoder
from espnet.nets.pytorch_backend.tacotron2.decoder import Prenet as DecoderPrenet
# from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.mlm_encoder import MLMTransformerEncoder
from espnet2.asr.encoder.hubert_encoder import FairseqHubertEncoder
from espnet2.asr.encoder.hubert_encoder import FairseqHubertPretrainEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder as TransformerEncoder
from espnet2.asr.encoder.contextual_block_transformer_encoder import (
    ContextualBlockTransformerEncoder,  # noqa: H301
)
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.encoder.wav2vec2_encoder import FairSeqWav2Vec2Encoder
from espnet2.asr.espnet_model import ESPnetMLMModel, ESPnetMLMDecoderModel,ESPnetMLMEncAsDecoderModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
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
from espnet2.train.collate_fn import CommonCollateFn
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

encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        transformer=MLMTransformerEncoder,
        contextual_block_transformer=ContextualBlockTransformerEncoder,
        vgg_rnn=VGGRNNEncoder,
        rnn=RNNEncoder,
        wav2vec2=FairSeqWav2Vec2Encoder,
        hubert=FairseqHubertEncoder,
        hubert_pretrain=FairseqHubertPretrainEncoder,
    ),
    type_check=AbsEncoder,
    default="transformer",
)

decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        transformer=TransformerDecoder,
        transformer_encoder=TransformerEncoder,
        no_decoder=TransformerDecoder
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
            "--ctc_conf",
            action=NestedDictAction,
            default=get_default_kwargs(CTC),
            help="The keyword arguments for CTC class.",
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
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=0)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
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
        assert check_return_type(retval)
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
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetMLMModel:
        assert check_argument_types()
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
        # 4. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        
        encoder = encoder_class(vocab_size=vocab_size,input_size=args.input_size, pos_enc_class=pos_enc_class,
        **args.encoder_conf)
        encoder_output_size = encoder.output_size()

        # # 5. Decoder
        if args.decoder != 'no_decoder' and 'encoder' not in args.decoder:
            decoder_class = decoder_choices.get_class(args.decoder)
            decoder_input_layer = torch.nn.Sequential(
                    DecoderPrenet(
                        idim=odim,
                        **args.pre_decoder_conf
                    ),
                    torch.nn.Linear(256, encoder.output_size()),
                )
            decoder = decoder_class(
                odim=odim,
                input_layer=decoder_input_layer,
                **args.decoder_conf,
            )
        elif 'encoder' in args.decoder:
            decoder_class = decoder_choices.get_class(args.decoder)
            decoder = decoder_class(
                idim=0,
                input_layer=None,
                **args.decoder_conf,
            )
        else:
            decoder = None
        # 6. CTC
        ctc = CTC(
            odim=vocab_size, encoder_output_sizse=encoder_output_size, **args.ctc_conf
        )


        # 8. Build model
        if decoder is not None and 'encoder' not in args.decoder:
            model = ESPnetMLMDecoderModel(
                feats_extract=feats_extract,
                odim=odim,
                normalize=normalize,
                encoder=encoder,
                decoder=decoder,
                ctc=ctc,
                token_list=token_list,
                **args.model_conf,
            )
        if 'encoder' in args.decoder:
            model = ESPnetMLMEncAsDecoderModel(
                feats_extract=feats_extract,
                odim=odim,
                normalize=normalize,
                encoder=encoder,
                decoder=decoder,
                ctc=ctc,
                token_list=token_list,
                **args.model_conf,
            )
        else:
            model = ESPnetMLMModel(
                feats_extract=feats_extract,
                odim=odim,
                normalize=normalize,
                encoder=encoder,
                decoder=decoder,
                ctc=ctc,
                token_list=token_list,
                **args.model_conf,
            )

        # 9. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
