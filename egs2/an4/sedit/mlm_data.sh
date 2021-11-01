#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0
data_base_path="/mnt/home/v_baihe/data"
dataset_name="toy_librilight"
data_feat_path=${data_base_path}/feat_dir/${dataset_name}
ndev_utt=100
cmd_backend='local'
# General configuration
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages.
skip_train=false     # Skip training stages.
skip_eval=false      # Skip decoding and evaluation stages.
skip_upload=true     # Skip packing and uploading stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=${data_feat_path}/dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Feature extraction related
feats_type=raw       # Feature type (raw or fbank_pitch).
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k               # Sampling rate.
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=30  # Maximum duration in second.
use_xvector=false          # Whether to use x-vector (Require Kaldi).
use_sid=false              # Whether to use speaker id as the inputs (Need utt2spk in data directory).
use_lid=false              # Whether to use language id as the inputs (Need utt2lang in data directory).
feats_extract=fbank        # On-the-fly feature extractor.
feats_normalize=global_mvn # On-the-fly feature normalizer.
fs=16000                   # Sampling rate.
n_fft=1024                 # The number of fft points.
n_shift=256                # The number of shift points.
win_length=null            # Window length.
fmin=80                    # Minimum frequency of Mel basis.
fmax=7600                  # Maximum frequency of Mel basis.
n_mels=80                  # The number of mel basis.
write_collected_feats=false
# Only used for the model using pitch & energy features (e.g. FastSpeech2)
f0min=80  # Maximum f0 for pitch extraction.
f0max=400 # Minimum f0 for pitch extraction.

# Tokenization related
srctexts=""      # Texts to create token list. Multiple items can be specified.
token_type=bpe      # Tokenization type (char or bpe).
nbpe=5000             # The number of BPE vocabulary.
bpemode=unigram     # Mode of BPE (unigram or bpe).
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole
bpe_input_sentence_size=5000000 # Size of input sentence for BPE.
bpe_nlsyms=         # non-linguistic symbols list, separated by a comma, for BPE
bpe_char_cover=1.0  # character coverage when modeling BPE

# ASR model related
mlm_tag=       # Suffix to the result dir for asr model training.
mlm_exp=       # Specify the directory path for ASR experiment.
               # If this option is specified, mlm_tag is ignored.
mlm_stats_dir= # Specify the directory path for ASR statistics.
mlm_config=conf/train_mlm.yaml    # Config for asr model training.
mlm_args=      # Arguments for asr model training, e.g., "--max_epoch 10".
               # Note that it will overwrite args in asr config.
num_splits_asr=1           # Number of splitting for lm corpus.

# Decoding related
batch_size=1
inference_tag=    # Suffix to the result dir for decoding.
inference_config= # Config for decoding.
inference_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
                  # Note that it will overwrite args in inference config.
inference_model=train.loss.ave.pth # Model path for decoding.
                                   # e.g.
                                   # inference_model=train.loss.best.pth
                                   # inference_model=3epoch.pth
                                   # inference_model=valid.acc.best.pth
                                   # inference_model=valid.loss.ave.pth
vocoder_file=none  # Vocoder parameter file, If set to none, Griffin-Lim will be used.
download_model=""  # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set="train_nodev"       # Name of training set.
valid_set="train_dev"     # Name of validation set used for monitoring/tuning network training.
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
bpe_train_text=  # Text file path of bpe training set.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
g2p=none         # g2p method (needed if token_type=phn).
lang=noinfo      # The language type of corpus.
score_opts=                # The options given to sclite scoring
local_score_opts=          # The options given to local/score.sh.
speech_fold_length=80000 # fold_length for speech data.

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

. ./path.sh
. ./cmd.sh

EXCLUDELIST=asimov-[49,59,54,55,84,85,91,92,96,99,204,163,162,186,209,174,222]

# Check feature type
data_feats=${dumpdir}/raw

# Set tag for naming of model directory
if [ -z "${mlm_tag}" ]; then
    if [ -n "${mlm_config}" ]; then
        mlm_tag="$(basename "${mlm_config}" .yaml)_${feats_type}"
    else
        mlm_tag="train_${feats_type}"
    fi
    if [ "${lang}" != noinfo ]; then
        mlm_tag+="_${lang}_${token_type}"
    else
        mlm_tag+="_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        mlm_tag+="${nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${mlm_args}" ]; then
        mlm_tag+="$(echo "${mlm_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
fi

# The directory used for collect-stats mode
if [ -z "${mlm_stats_dir}" ]; then
    if [ "${lang}" != noinfo ]; then
        mlm_stats_dir="${expdir}/mlm_stats_${feats_type}_${lang}_${token_type}"
    else
        mlm_stats_dir="${expdir}/mlm_stats_${feats_type}_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        mlm_stats_dir+="${nbpe}"
    fi
fi

# The directory used for training commands
if [ -z "${mlm_exp}" ]; then
    mlm_exp="${expdir}/mlm_${mlm_tag}"
fi

# local/data_librilight.sh "--dataset ${dataset_name}"

# stage 0: generate wav.scp files
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 local/data_prep_librilight.py --dataset ${dataset_name} --data_base_path ${data_base_path}

    # make a dev set
    utils/subset_data_dir.sh --first ${data_feat_path} "${ndev_utt}" "${data_feat_path}/${valid_set}"
    n=$(($(wc -l < ${data_feat_path}/wav.scp) - ndev_utt))
    utils/subset_data_dir.sh --last ${data_feat_path} "${n}" "${data_feat_path}/${train_set}"
fi
# stage 1: format wav.scp files
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for dset in "${train_set}" "${valid_set}"; do
        utils/copy_data_dir.sh --validate_opts --non-print "${data_feat_path}/${dset}" "${data_feats}/${dset}"
        rm -f ${data_feats}/${dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}
        scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
            --audio-format "${audio_format}" --fs "${fs}" \
            "${data_feat_path}/${dset}/wav.scp" "${data_feats}/${dset}"
    done
    echo "${feats_type}" > "${data_feat_path}/${dset}/feats_type"
fi

mlm_stats_dir="${data_feat_path}/stats_raw"
_logdir="${mlm_stats_dir}/logdir"

# Check token list type
token_listdir="${dumpdir}/token_list/${token_type}"
if [ "${cleaner}" != none ]; then
    token_listdir+="_${cleaner}"
fi
if [ "${token_type}" = phn ]; then
    token_listdir+="_${g2p}"
fi
token_list="${token_listdir}/tokens.txt"
# Check old version token list dir existence
if [ -e data/token_list ] && [ ! -e "${dumpdir}/token_list" ]; then
    log "Default token_list directory path is changed from data to ${dumpdir}."
    log "Copy data/token_list to ${dumpdir}/token_list for the compatibility."
    [ ! -e ${dumpdir} ] && mkdir -p ${dumpdir}
    cp -a "data/token_list" "${dumpdir}/token_list"
fi

bpeprefix="${token_listdir}"/bpe
bpemodel="${bpeprefix}".model
if [ "${token_type}" != bpe ]; then
    bpemodel=none
fi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Generate token_list from ${srctexts}"
    # "nlsyms_txt" should be generated by local/data.sh if need

    # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
    # 0 is reserved for CTC-blank for ASR and also used as ignore-index in the other task
    if [ ! -f ${token_listdir}/text.txt ]; then
        cat ${srctexts} | awk ' { if( NF != 1 ) print $0; } ' >"${token_listdir}/text.txt"
    fi

    if [ "${token_type}" = bpe ]; then

        if [ -n "${bpe_nlsyms}" ]; then
            _opts_spm="--user_defined_symbols=${bpe_nlsyms}"
        else
            _opts_spm=""
        fi

        spm_train \
            --input="${token_listdir}/text.txt" \
            --vocab_size="${nbpe}" \
            --model_type="${bpemode}" \
            --model_prefix="${bpeprefix}" \
            --character_coverage=${bpe_char_cover} \
            --input_sentence_size="${bpe_input_sentence_size}" \
            --shuffle_input_sentence=true \
            ${_opts_spm}

        {
        echo "${blank}"
        echo "${oov}"
        # Remove <unk>, <s>, </s> from the vocabulary
        <"${bpeprefix}".vocab awk '{ if( NR != 1 && NR != 2 && NR != 3 ){ print $1; } }'
        echo "${sos_eos}"
        } > "${token_list}"

    elif [ "${token_type}" = char ] || [ "${token_type}" = phn ]; then
        _opts="--non_linguistic_symbols ${nlsyms_txt}"

        # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
        # 0 is reserved for CTC-blank for ASR and also used as ignore-index in the other task
        ${python} -m espnet2.bin.tokenize_text \
              --token_type "${token_type}" -f 2- \
              --input "${data_feats}/text_tokenlist.txt" --output "${token_list}" \
              --non_linguistic_symbols "${nlsyms_txt}" \
              --cleaner "${cleaner}" \
              --g2p "${g2p}" \
              --write_vocabulary true \
              --add_symbol "${blank}:0" \
              --add_symbol "${oov}:1" \
              --add_symbol "${sos_eos}:-1"

    else
        log "Error: not supported --token_type '${token_type}'"
        exit 2
    fi
fi


# stage 3: collect stats
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # 1. Split the key file
    _scp=wav.scp
    _train_dir="${data_feats}/${train_set}"
    _valid_dir="${data_feats}/${valid_set}"
    _nj=$(min "${nj}" "$(<${_train_dir}/${_scp} wc -l)" "$(<${_valid_dir}/${_scp} wc -l)")
    key_file="${_train_dir}/${_scp}"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/train.${n}.scp"
    done
    utils/split_scp.pl "${key_file}" ${split_scps}
    _opts=
    if [ -n "${mlm_config}" ]; then
        _opts+="--config ${mlm_config} "
    fi
    key_file="${_valid_dir}/${_scp}"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/valid.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}
    # 2. Generate run.sh
    log "Generate '${mlm_stats_dir}/run.sh'. You can resume the process from stage 3 using this script"
    mkdir -p "${mlm_stats_dir}"; echo "${run_args} --stage 3 \"\$@\"; exit \$?" > "${mlm_stats_dir}/run.sh"; chmod +x "${mlm_stats_dir}/run.sh"
    # 3. Submit jobs
    _opts+="--input_size ${n_mels} "
    _opts+="--feats_extract ${feats_extract} "
    _opts+="--feats_extract_conf n_fft=${n_fft} "
    _opts+="--feats_extract_conf hop_length=${n_shift} "
    _opts+="--feats_extract_conf win_length=${win_length} "
    _opts+="--feats_extract_conf fs=${fs} "
    _opts+="--feats_extract_conf fmin=${fmin} "
    _opts+="--feats_extract_conf fmax=${fmax} "
    _opts+="--feats_extract_conf n_mels=${n_mels} "
    ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
                ${python} -m espnet2.bin.mlm_train \
                    --collect_stats true \
                    --token_list ${token_list} \
                    --bpemodel "${bpemodel}" \
                    --use_preprocessor true \
                    --non_linguistic_symbols "${nlsyms_txt}" \
                    --cleaner "${cleaner}" \
                    --g2p "${g2p}" \
                    --train_data_path_and_name_and_type "${_train_dir}/wav.scp,speech,sound" \
                    --valid_data_path_and_name_and_type "${_valid_dir}/wav.scp,speech,sound" \
                    --train_shape_file "${_logdir}/train.JOB.scp" \
                    --valid_shape_file "${_logdir}/valid.JOB.scp" \
                    --output_dir "${_logdir}/stats.JOB" \
                    --write_collected_feats "${write_collected_feats}" \
                    ${_opts} || { cat "${_logdir}"/stats.1.log; exit 1; }
    # 4. Aggregate shape files
    _opts=
    for i in $(seq "${_nj}"); do
        _opts+="--input_dir ${_logdir}/stats.${i} "
    done
    ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${mlm_stats_dir}"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    _mlm_train_dir="${data_feats}/${train_set}"
    _mlm_valid_dir="${data_feats}/${valid_set}"
    log "Stage 4: MLM Training: train_set=${_mlm_train_dir}, valid_set=${_mlm_valid_dir}"

    _opts=
    if [ -n "${mlm_config}" ]; then
        _opts+="--config ${mlm_config} "
    fi

    _feats_type="$(<${_mlm_train_dir}/feats_type)"
    if [ "${_feats_type}" = raw ]; then
        _scp=wav.scp
        _type=sound
        _fold_length="${speech_fold_length}"
    # else
    #     _scp=feats.scp
    #     _type=sound
    #     _fold_length="${asr_speech_fold_length}"
    #     _input_size="$(<${_asr_train_dir}/feats_dim)"

    fi
    if [ "${feats_normalize}" = global_mvn ]; then
        # Default normalization is utterance_mvn and changes to global_mvn
        _opts+="--normalize=global_mvn --normalize_conf stats_file=${mlm_stats_dir}/train/feats_stats.npz "
    fi

    _opts+="--train_data_path_and_name_and_type ${_mlm_train_dir}/${_scp},speech,${_type} "
    _opts+="--train_shape_file ${mlm_stats_dir}/train/speech_shape "

    log "Generate '${mlm_exp}/run.sh'. You can resume the process from stage 4 using this script"
    mkdir -p "${mlm_exp}"; echo "${run_args} --stage 4 \"\$@\"; exit \$?" > "${mlm_exp}/run.sh"; chmod +x "${mlm_exp}/run.sh"

    BASE="srun python -m espnet2.bin.mlm_train"
    _opts+="--use_preprocessor true --bpemodel ${bpemodel} --token_type ${token_type} --token_list ${token_list} --non_linguistic_symbols ${nlsyms_txt} --cleaner ${cleaner} --g2p ${g2p} --valid_data_path_and_name_and_type ${_mlm_valid_dir}/${_scp},speech,${_type} --valid_shape_file ${mlm_stats_dir}/valid/speech_shape --resume true --fold_length ${_fold_length} --output_dir ${mlm_exp} "
    _opts+="--input_size ${n_mels} "
    _opts+="--feats_extract ${feats_extract} "
    _opts+="--feats_extract_conf n_fft=${n_fft} "
    _opts+="--feats_extract_conf hop_length=${n_shift} "
    _opts+="--feats_extract_conf win_length=${win_length} "
    _opts+="--feats_extract_conf fs=${fs} "
    _opts+="--feats_extract_conf fmin=${fmin} "
    _opts+="--feats_extract_conf fmax=${fmax} "
    _opts+="--feats_extract_conf n_mels=${n_mels} "
    DIST_ARGS="--ngpu ${ngpu} --multiprocessing_distributed true --dist_launcher slurm --dist_init_method file://$(pwd)/exp/asr_stats_raw_en_bpe30/.dist_init_$(openssl rand -base64 -hex 12)"

    sbatch --job-name train_mlm --gres gpu:${ngpu} --nodes ${num_nodes} --partition V100x8,P100,2080Ti_mlong,1080Ti_mlong,TitanXx8_mlong,2080Ti,1080Ti,TitanXx8 \
    --exclude $EXCLUDELIST \
    --wrap " $BASE $_opts $DIST_ARGS " 
fi