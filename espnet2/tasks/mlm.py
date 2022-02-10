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

import copy
import functools
import yaml
import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type
from espnet2.train.distributed_utils import DistributedOption
from espnet2.iterators.multiple_iter_factory import MultipleIterFactory

from espnet.nets.pytorch_backend.tacotron2.decoder import Prenet as DecoderPrenet

from espnet.nets.pytorch_backend.transformer.encoder import MLMEncoder as TransformerEncoder
from espnet.nets.pytorch_backend.transformer.encoder import MLMDecoder as TransformerDecoder
from espnet.nets.pytorch_backend.conformer.encoder import MLMEncoder as ConformerEncoder
from espnet.nets.pytorch_backend.conformer.encoder import MLMDecoder as ConformerDecoder

from espnet2.tts.sedit.sedit_model import ESPnetMLMModel,ESPnetMLMEncAsDecoderModel,ESPnetMLMTTSModel
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
            mlm_prob_factor = 0.8 #mlm_probs[epoch // 100]
        if 'duration_predictor_layers' in args.model_conf.keys() and args.model_conf['duration_predictor_layers']>0:
            duration_collect=True
        else:
            duration_collect=False
        return MLMCollateFn(feats_extract, float_pad_value=0.0, int_pad_value=0,
        mlm_prob=args.model_conf['mlm_prob']*mlm_prob_factor,mean_phn_span=args.model_conf['mean_phn_span'],attention_window=attention_window,pad_speech=pad_speech,sega_emb=sega_emb,duration_collect=duration_collect)

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
        retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ("text","align_start","align_end")
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

        if "conformer" == args.encoder:
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
            if "conformer"==args.decoder:
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
        if 'duration_predictor_layers' in args.model_conf.keys() and args.model_conf['duration_predictor_layers']>0:
            model = ESPnetMLMTTSModel(
            feats_extract=feats_extract,
            odim=odim,
            normalize=normalize,
            encoder=encoder,
            decoder=decoder,
            token_list=token_list,
            **args.model_conf,
            )
        else:
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


    @classmethod
    def build_multiple_iter_factory(
        cls, args: argparse.Namespace, distributed_option: DistributedOption, mode: str
    ):
        assert check_argument_types()
        iter_options = cls.build_iter_options(args, distributed_option, mode)
        assert len(iter_options.data_path_and_name_and_type) > 0, len(
            iter_options.data_path_and_name_and_type
        )
        dataset_training_portion = {'libritts':0.3, "librispeech":0.3, "librilight":0.3,"vctk":0.1,"librilight_sub":0.3}
        dataset_args = []

        # 1. Sanity check
        dataset_data_path_and_name_and_type = {}
        for path in iter_options.data_path_and_name_and_type:
            if 'splits50' in path[0]:
                dataset_name = 'librilight'
            else:
                dataset_name = path[0].split('/')[-2]
            if dataset_name in dataset_data_path_and_name_and_type:
                dataset_data_path_and_name_and_type[dataset_name].append(path)
            else:
                dataset_data_path_and_name_and_type[dataset_name] = [path]
        dataset_shape_files = {}
        for path in iter_options.shape_files:
            if 'splits50' in path:
                dataset_name = 'librilight'
            else:
                dataset_name = path.split('/')[-3].split('_stats')[0]
            if dataset_name in dataset_shape_files:
                dataset_shape_files[dataset_name].append(path)
            else:
                dataset_shape_files[dataset_name] = [path]
        dataset_num_iters_per_epoch = {}
        for k,v in dataset_data_path_and_name_and_type.items():
            dataset_num_iters_per_epoch[k] = int(iter_options.num_iters_per_epoch*dataset_training_portion[k])

        max_cache_size = iter_options.max_cache_size
        args_24k = copy.deepcopy(args)
        args_24k.feats_extract_conf['fs']=24000
        args_24k.feats_extract_conf['n_fft']=2048
        args_24k.feats_extract_conf['hop_length']=300
        args_24k.feats_extract_conf['win_length']=1200

        args_16k = copy.deepcopy(args)
        args_16k.feats_extract_conf['fs']=16000
        args_16k.feats_extract_conf['n_fft']=1024
        args_16k.feats_extract_conf['hop_length']=200
        args_16k.feats_extract_conf['win_length']=800

        dataset_args = {'libritts':args_24k, 'librispeech':args_16k, 'librilight':args_16k,'vctk':args_24k,'librilight_sub':args_16k}
        # Note that iter-factories are built for each epoch at runtime lazily.
        build_funcs = {
            k: functools.partial(
                cls.build_iter_factory,
                dataset_args[k],
                distributed_option,
                mode,
                kwargs=dict(
                    data_path_and_name_and_type=dataset_data_path_and_name_and_type[k],
                    shape_files=dataset_shape_files[k],
                    num_iters_per_epoch=dataset_num_iters_per_epoch[k],
                    max_cache_size=iter_options.max_cache_size,
                ),
            )
            for k in dataset_data_path_and_name_and_type.keys()
        }
        
        # if 'librilight' in build_funcs.keys():
        #     build_funcs['librilight'] = functools.partial(
        #             cls.build_multiple_iter_factory_for_split,
        #             dataset_args['librilight'],
        #             distributed_option,
        #             mode,
        #             kwargs=dict(
        #                 data_path_and_name_and_type=dataset_data_path_and_name_and_type['librilight'],
        #                 shape_files=dataset_shape_files['librilight'],
        #                 num_iters_per_epoch=dataset_num_iters_per_epoch['librilight'],
        #                 max_cache_size=iter_options.max_cache_size,
        #             )
        #     )

        build_funcs_list = list(build_funcs.values())
        
        # 3. Build MultipleIterFactory
        return MultipleIterFactory(
            build_funcs=build_funcs_list, shuffle=iter_options.train, seed=args.seed,
        )
        # build_funcs_list = [build_funcs['librilight_sub'],build_funcs['vctk'],build_funcs['libritts'],build_funcs['librispeech'],]
        # return MultipleIterFactory(
        #     build_funcs=build_funcs_list, shuffle=False, seed=args.seed,
        # )


    @classmethod
    def build_multiple_iter_factory_for_split(
        cls, args: argparse.Namespace, distributed_option: DistributedOption, mode: str, kwargs: dict = None,
    ):
        assert check_argument_types()
        iter_options = cls.build_iter_options(args, distributed_option, mode)
        assert len(iter_options.data_path_and_name_and_type) > 0, len(
            iter_options.data_path_and_name_and_type
        )
        # Overwrite iter_options if any kwargs is given
        if kwargs is not None:
            for k, v in kwargs.items():
                setattr(iter_options, k, v)

        # 1. Sanity check
        num_splits = None
        for path in [
            path for path, _, _ in iter_options.data_path_and_name_and_type
        ] + list(iter_options.shape_files):
            if not Path(path).is_dir():
                raise RuntimeError(f"{path} is not a directory")
            p = Path(path) / "num_splits"
            if not p.exists():
                raise FileNotFoundError(f"{p} is not found")
            with p.open() as f:
                _num_splits = int(f.read())
                if num_splits is not None and num_splits != _num_splits:
                    raise RuntimeError(
                        f"Number of splits are mismathed: "
                        f"{iter_options.data_path_and_name_and_type[0][0]} and {path}"
                    )
                num_splits = _num_splits

            for i in range(num_splits):
                p = Path(path) / f"split.{i}"
                if not p.exists():
                    raise FileNotFoundError(f"{p} is not found")

        # 2. Create functions to build an iter factory for each splits
        data_path_and_name_and_type_list = [
            [
                (str(Path(p) / f"split.{i}"), n, t)
                for p, n, t in iter_options.data_path_and_name_and_type
            ]
            for i in range(num_splits)
        ]
        shape_files_list = [
            [str(Path(s) / f"split.{i}") for s in iter_options.shape_files]
            for i in range(num_splits)
        ]
        num_iters_per_epoch_list = [
            iter_options.num_iters_per_epoch
            if iter_options.num_iters_per_epoch is not None
            else None
            for i in range(num_splits)
        ]
        max_cache_size = iter_options.max_cache_size / num_splits

        # Note that iter-factories are built for each epoch at runtime lazily.
        build_funcs = [
            functools.partial(
                cls.build_iter_factory,
                args,
                distributed_option,
                mode,
                kwargs=dict(
                    data_path_and_name_and_type=_data_path_and_name_and_type,
                    shape_files=_shape_files,
                    num_iters_per_epoch=_num_iters_per_epoch,
                    max_cache_size=max_cache_size,
                ),
            )
            for (
                _data_path_and_name_and_type,
                _shape_files,
                _num_iters_per_epoch,
            ) in zip(
                data_path_and_name_and_type_list,
                shape_files_list,
                num_iters_per_epoch_list,
            )
        ]

        # 3. Build MultipleIterFactory
        return MultipleIterFactory(
            build_funcs=build_funcs, shuffle=iter_options.train, seed=args.seed
        )
