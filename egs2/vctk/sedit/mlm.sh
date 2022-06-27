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
dataset_name="libritts"
data_feat_path=${data_base_path}/feat_dir/${dataset_name}
ndev_utt=100
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
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw       # Feature type (raw or fbank).
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k               # Sampling rate.
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=20  # Maximum duration in second.
use_xvector=false          # Whether to use x-vector (Require Kaldi).
use_sid=false              # Whether to use speaker id as the inputs (Need utt2spk in data directory).
use_lid=false              # Whether to use language id as the inputs (Need utt2lang in data directory).
feats_extract=fbank        # On-the-fly feature extractor.
feats_normalize= # On-the-fly feature normalizer.
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
phn_as_word=false
# MLM model related
mlm_tag=       # Suffix to the result dir for asr model training.
mlm_exp=       # Specify the directory path for ASR experiment.
               # If this option is specified, mlm_tag is ignored.
mlm_stats_dir= # Specify the directory path for ASR statistics.
mlm_config=conf/train_mlm.yaml    # Config for asr model training.
mlm_args=      # Arguments for asr model training, e.g., "--max_epoch 10".
               # Note that it will overwrite args in asr config.

num_splits_mlm=1           # Number of splitting for lm corpus.

# Decoding related
use_k2=false      # Whether to use k2 based decoder
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
train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training.
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
bpe_train_text=  # Text file path of bpe training set.

nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
g2p=g2p_en          # g2p method (needed if token_type=phn).
lang=noinfo      # The language type of corpus.
score_opts=                # The options given to sclite scoring
local_score_opts=          # The options given to local/score.sh.
speech_fold_length=800 # fold_length for speech data.training.
text_fold_length=150   # fold_length for text data during ASR training.


log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

. ./path.sh
. ./cmd.sh


# Check required arguments
[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
[ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; };
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats=${dumpdir}/raw
elif [ "${feats_type}" = fbank_pitch ]; then
    data_feats=${dumpdir}/fbank_pitch
elif [ "${feats_type}" = fbank ]; then
    data_feats=${dumpdir}/fbank
elif [ "${feats_type}" == extracted ]; then
    data_feats=${dumpdir}/extracted
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

# Use the same text as ASR for bpe training if not specified.
[ -z "${bpe_train_text}" ] && bpe_train_text="${data_feats}/${train_set}/text"

# Check tokenization type
if [ "${lang}" != noinfo ]; then
    token_listdir=data/${lang}_token_list
else
    token_listdir=data/token_list
fi
if [ "${token_type}" = phn ]; then
    token_listdir+="_${g2p}"
fi
bpedir="${token_listdir}/bpe_${bpemode}${nbpe}"
bpeprefix="${bpedir}"/bpe
bpemodel="${bpeprefix}".model
bpetoken_list="${bpedir}"/tokens.txt
chartoken_list="${token_listdir}"/char/tokens.txt
# NOTE: keep for future development.
# shellcheck disable=SC2034
wordtoken_list="${token_listdir}"/word/tokens.txt
phntoken_listdir="${token_listdir}"/phn/tokens.txt

if [ "${token_type}" = phn ]; then
    token_list="${phntoken_listdir}"
elif [ "${token_type}" = bpe ]; then
    token_list="${bpetoken_list}"
elif [ "${token_type}" = char ]; then
    token_list="${chartoken_list}"
    bpemodel=none
elif [ "${token_type}" = word ]; then
    token_list="${wordtoken_list}"
    bpemodel=none
else
    log "Error: not supported --token_type '${token_type}'"
    exit 2
fi

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
    if [ "${token_type}" = phn ]; then
        mlm_tag+="_${g2p}"
    fi
    # Add overwritten arg's info
    if [ -n "${mlm_args}" ]; then
        mlm_tag+="$(echo "${mlm_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        mlm_tag+="_sp"
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
    if [ "${token_type}" = phn ]; then
        mlm_stats_dir+="_${g2p}"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        mlm_stats_dir+="_sp"
    fi
fi

# The directory used for training commands
if [ -z "${mlm_exp}" ]; then
    mlm_exp="${expdir}/mlm_${mlm_tag}"
fi

# if [ -z "${inference_tag}" ]; then
#     if [ -n "${inference_config}" ]; then
#         inference_tag="$(basename "${inference_config}" .yaml)"
#     else
#         inference_tag=inference
#     fi
#     # Add overwritten arg's info
#     if [ -n "${inference_args}" ]; then
#         inference_tag+="$(echo "${inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
#     fi
#     inference_tag+="_mlm_model_$(echo "${inference_mlm_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"

# fi

# ========================== Main stages start from here. ==========================

act_token_type="${token_type}"
if "${phn_as_word}"; then
    act_token_type="word"
fi

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data_${dataset_name}.sh ${local_data_opts}
    fi

    # if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    #     if [ -n "${speed_perturb_factors}" ]; then
    #        log "Stage 2: Speed perturbation: data/${train_set} -> data/${train_set}_sp"
    #        for factor in ${speed_perturb_factors}; do
    #            if [[ $(bc <<<"${factor} != 1.0") == 1 ]]; then
    #                scripts/utils/perturb_data_dir_speed.sh "${factor}" "data/${train_set}" "data/${train_set}_sp${factor}"
    #                _dirs+="data/${train_set}_sp${factor} "
    #            else
    #                # If speed factor is 1, same as the original
    #                _dirs+="data/${train_set} "
    #            fi
    #        done
    #        utils/combine_data.sh "data/${train_set}_sp" ${_dirs}
    #     else
    #        log "Skip stage 2: Speed perturbation"
    #     fi
    # fi

    if [ -n "${speed_perturb_factors}" ]; then
        train_set="${train_set}_sp"
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        if [ "${feats_type}" = raw ]; then
            log "Stage 3: Format wav.scp: data/ -> ${data_feats}"

            # ====== Recreating "wav.scp" ======
            # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
            # shouldn't be used in training process.
            # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
            # and it can also change the audio-format and sampling rate.
            # If nothing is need, then format_wav_scp.sh does nothing:
            # i.e. the input file format and rate is same as the output.

            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"
                rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}
                _opts=
                if [ -e data/"${dset}"/segments ]; then
                    # "segments" is used for splitting wav files which are written in "wav".scp
                    # into utterances. The file format of segments:
                    #   <segment_id> <record_id> <start_time> <end_time>
                    #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                    # Where the time is written in seconds.
                    _opts+="--segments data/${dset}/segments "
                fi
                # shellcheck disable=SC2086
                scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                    "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"

                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
            done

        elif [ "${feats_type}" = fbank ]; then
            log "[Require Kaldi] Stage 3: ${feats_type} extract: data/ -> ${data_feats}"

            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                # 1. Copy datadir
                utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"

                # 2. Feature extract
                _nj=$(min "${nj}" "$(<"${data_feats}${_suf}/${dset}/utt2spk" wc -l)")
                steps/make_fbank.sh --nj "${_nj}" --cmd "${train_cmd}" "${data_feats}${_suf}/${dset}"
                utils/fix_data_dir.sh "${data_feats}${_suf}/${dset}"

                # 3. Derive the the frame length and feature dimension
                scripts/feats/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
                    "${data_feats}${_suf}/${dset}/feats.scp" "${data_feats}${_suf}/${dset}/feats_shape"

                # 4. Write feats_dim
                head -n 1 "${data_feats}${_suf}/${dset}/feats_shape" | awk '{ print $2 }' \
                    | cut -d, -f2 > ${data_feats}${_suf}/${dset}/feats_dim

                # 5. Write feats_type
                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
            done
        else
            log "Error: not supported: --feats_type ${feats_type}"
            exit 2
        fi
    fi


    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Stage 4: Remove long/short data: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${valid_set}"; do
            # Copy data dir
            utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"
            if [ -e "${data_feats}/org/${dset}/utt2sid" ]; then
                cp "${data_feats}/org/${dset}/utt2sid" "${data_feats}/${dset}/utt2sid"
            fi
            if [ -e "${data_feats}/org/${dset}/utt2lid" ]; then
                cp "${data_feats}/org/${dset}/utt2lid" "${data_feats}/${dset}/utt2lid"
            fi

            # Remove short utterances
            _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
            _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
            _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

            # utt2num_samples is created by format_wav_scp.sh
            <"${data_feats}/org/${dset}/utt2num_samples" \
                awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                    '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                    >"${data_feats}/${dset}/utt2num_samples"
            <"${data_feats}/org/${dset}/wav.scp" \
                utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                >"${data_feats}/${dset}/wav.scp"

            # Remove empty text
            <"${data_feats}/org/${dset}/text" \
                awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/${dset}/text"

            # fix_data_dir.sh leaves only utts which exist in all files
            _fix_opts=""
            if [ -e "${data_feats}/org/${dset}/utt2sid" ]; then
                _fix_opts="--utt_extra_files utt2sid "
            fi
            if [ -e "${data_feats}/org/${dset}/utt2lid" ]; then
                _fix_opts="--utt_extra_files utt2lid "
            fi
            # shellcheck disable=SC2086
            utils/fix_data_dir.sh ${_fix_opts} "${data_feats}/${dset}"

            # Filter x-vector
            if "${use_xvector}"; then
                cp "${dumpdir}/xvector/${dset}"/xvector.{scp,scp.bak}
                <"${dumpdir}/xvector/${dset}/xvector.scp.bak" \
                    utils/filter_scp.pl "${data_feats}/${dset}/wav.scp"  \
                    >"${dumpdir}/xvector/${dset}/xvector.scp"
            fi

        done

        # shellcheck disable=SC2002
        cat ${srctexts} | awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/srctexts"
    fi


    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        log "Stage 5: Generate token_list from ${srctexts}"
        # "nlsyms_txt" should be generated by local/data.sh if need

        # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
        # 0 is reserved for CTC-blank for ASR and also used as ignore-index in the other task
        mkdir -p ${token_listdir}
        _opts="--non_linguistic_symbols ${nlsyms_txt}"

        # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
        # 0 is reserved for CTC-blank for ASR and also used as ignore-index in the other task
        ${python} -m espnet2.bin.tokenize_text \
            --token_type "${act_token_type}" -f 2- \
            --input "${data_feats}/srctexts" \
            --output "${token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --write_vocabulary true \
            --add_symbol "${blank}:0" \
            --add_symbol "${oov}:1" \
            --add_symbol "${sos_eos}:-1"
    fi
else
    log "Skip the stages for data preparation"
fi

# ========================== Data preparation is done here. ==========================

if ! "${skip_train}"; then

    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        _mlm_train_dir="${data_feats}/${train_set}"
        _mlm_valid_dir="${data_feats}/${valid_set}"
        log "Stage 10: MLM collect stats: train_set=${_mlm_train_dir}, valid_set=${_mlm_valid_dir}"

        _opts=
        if [ -n "${mlm_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.mlm_train --print_config --optim adam
            _opts+="--config ${mlm_config} "
        fi

        _feats_type="$(<${_mlm_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=mfa_wav.scp
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                # "sound" supports "wav", "flac", etc.
                _type=sound
            fi
                _opts+="--input_size ${n_mels} "
                _opts+="--feats_extract ${feats_extract} "
                _opts+="--feats_extract_conf n_fft=${n_fft} "
                _opts+="--feats_extract_conf hop_length=${n_shift} "
                _opts+="--feats_extract_conf win_length=${win_length} "
                _opts+="--feats_extract_conf fs=${fs} "
                _opts+="--feats_extract_conf fmin=${fmin} "
                _opts+="--feats_extract_conf fmax=${fmax} "
                _opts+="--feats_extract_conf n_mels=${n_mels} "
        else
            _scp=feats.scp
            _type=kaldi_ark
            _input_size="$(<${_mlm_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "
        fi

        # 1. Split the key file
        _logdir="${mlm_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_mlm_train_dir}/${_scp} wc -l)" "$(<${_mlm_valid_dir}/${_scp} wc -l)")

        key_file="${_mlm_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_mlm_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${mlm_stats_dir}/run.sh'. You can resume the process from stage 10 using this script"
        mkdir -p "${mlm_stats_dir}"; echo "${run_args} --stage 10 \"\$@\"; exit \$?" > "${mlm_stats_dir}/run.sh"; chmod +x "${mlm_stats_dir}/run.sh"

        # 3. Submit jobs
        log "MLM collect-stats started... log: '${_logdir}/stats.*.log'"

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.

        # shellcheck disable=SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m espnet2.bin.mlm_train \
                --collect_stats true \
                --use_preprocessor true \
                --bpemodel "${bpemodel}" \
                --token_type "${act_token_type}" \
                --token_list "${token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --normalize none \
                --train_data_path_and_name_and_type "${_mlm_train_dir}/${_scp},speech,${_type}" \
                --train_data_path_and_name_and_type "${_mlm_train_dir}/mfa_text,text,text" \
                --train_data_path_and_name_and_type "${_mlm_train_dir}/mfa_start,align_start,text_float" \
                --train_data_path_and_name_and_type "${_mlm_train_dir}/mfa_end,align_end,text_float" \
                --valid_data_path_and_name_and_type "${_mlm_valid_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_mlm_valid_dir}/mfa_text,text,text" \
                --valid_data_path_and_name_and_type "${_mlm_valid_dir}/mfa_start,align_start,text_float" \
                --valid_data_path_and_name_and_type "${_mlm_valid_dir}/mfa_end,align_end,text_float" \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${mlm_args} || { cat "${_logdir}"/stats.1.log; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${mlm_stats_dir}"

        # # Append the num-tokens at the last dimensions. This is used for batch-bins count
        # <"${mlm_stats_dir}/train/text_shape" \
        #     awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
        #     >"${mlm_stats_dir}/train/text_shape.${token_type}"

        # <"${mlm_stats_dir}/valid/text_shape" \
        #     awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
        #     >"${mlm_stats_dir}/valid/text_shape.${token_type}"
    fi


    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        _mlm_train_dir="${data_feats}/${train_set}"
        _mlm_valid_dir="${data_feats}/${valid_set}"
        log "Stage 7: MLM Training: train_set=${_mlm_train_dir}, valid_set=${_mlm_valid_dir}"

        _opts=
        if [ -n "${mlm_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.mlm_train --print_config --optim adam
            _opts+="--config ${mlm_config} "
        fi

        _feats_type="$(<${_mlm_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=mfa_wav.scp
            # "sound" supports "wav", "flac", etc.
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                _type=sound
            fi
            _fold_length="$((speech_fold_length * 100))"
            _opts+="--input_size ${n_mels} "
            _opts+="--feats_extract ${feats_extract} "
            _opts+="--feats_extract_conf n_fft=${n_fft} "
            _opts+="--feats_extract_conf hop_length=${n_shift} "
            _opts+="--feats_extract_conf win_length=${win_length} "
            _opts+="--feats_extract_conf fs=${fs} "
            _opts+="--feats_extract_conf fmin=${fmin} "
            _opts+="--feats_extract_conf fmax=${fmax} "
            _opts+="--feats_extract_conf n_mels=${n_mels} "
        else
            _scp=feats.scp
            _type=kaldi_ark
            _fold_length="${speech_fold_length}"
            _input_size="$(<${_mlm_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "

        fi
        if [ "${feats_normalize}" = global_mvn ]; then
            # Default normalization is utterance_mvn and changes to global_mvn
            _opts+="--normalize=global_mvn --normalize_conf stats_file=${mlm_stats_dir}/train/feats_stats.npz "
        fi

        if [ "${num_splits_mlm}" -gt 1 ]; then
            # If you met a memory error when parsing text files, this option may help you.
            # The corpus is split into subsets and each subset is used for training one by one in order,
            # so the memory footprint can be limited to the memory required for each dataset.

            _split_dir="${mlm_stats_dir}/splits${num_splits_mlm}"
            if [ ! -f "${_split_dir}/.done" ]; then
                rm -f "${_split_dir}/.done"
                ${python} -m espnet2.bin.split_scps \
                  --scps \
                      "${_mlm_train_dir}/${_scp}" \
                      "${_mlm_train_dir}/mfa_text" \
                      "${mlm_stats_dir}/train/speech_shape" \
                      "${mlm_stats_dir}/train/text_shape.${token_type}" \
                  --num_splits "${num_splits_mlm}" \
                  --output_dir "${_split_dir}"
                touch "${_split_dir}/.done"
            else
                log "${_split_dir}/.done exists. Spliting is skipped"
            fi

            _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/mfa_text,text,text "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/mfa_start,align_start,text_float "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/mfa_end,align_end,text_float "
            _opts+="--train_shape_file ${_split_dir}/speech_shape "
            _opts+="--train_shape_file ${_split_dir}/text_shape.${token_type} "
            _opts+="--multiple_iterator true "

        else
            _opts+="--train_data_path_and_name_and_type ${_mlm_train_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_mlm_train_dir}/mfa_text,text,text "
            _opts+="--train_data_path_and_name_and_type ${_mlm_train_dir}/mfa_start,align_start,text_float "
            _opts+="--train_data_path_and_name_and_type ${_mlm_train_dir}/mfa_end,align_end,text_float "

            # _opts+="--train_data_path_and_name_and_type ${_mlm_train_dir}/text,text,text "
            _opts+="--train_shape_file ${mlm_stats_dir}/train/speech_shape "
            # _opts+="--train_shape_file ${mlm_stats_dir}/train/text_shape.${token_type} "
        fi

        log "Generate '${mlm_exp}/run.sh'. You can resume the process from stage 7 using this script"
        mkdir -p "${mlm_exp}"; echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${mlm_exp}/run.sh"; chmod +x "${mlm_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
        log "MLM training started... log: '${mlm_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${mlm_exp})"
        else
            jobname="${mlm_exp}/train.log"
        fi

        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${mlm_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${mlm_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.mlm_train \
                --use_preprocessor true \
                --bpemodel "${bpemodel}" \
                --token_type "${act_token_type}" \
                --token_list "${token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --valid_data_path_and_name_and_type "${_mlm_valid_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_mlm_valid_dir}/mfa_text,text,text" \
                --valid_data_path_and_name_and_type "${_mlm_valid_dir}/mfa_start,align_start,text_float" \
                --valid_data_path_and_name_and_type "${_mlm_valid_dir}/mfa_end,align_end,text_float" \
                --valid_shape_file "${mlm_stats_dir}/valid/speech_shape" \
                --resume true \
                --fold_length "${_fold_length}" \
                --output_dir "${mlm_exp}" \
                ${_opts} ${mlm_args}
# --fold_length "${text_fold_length}" \
# --valid_data_path_and_name_and_type "${_mlm_valid_dir}/text,text,text" \
# --valid_shape_file "${mlm_stats_dir}/valid/text_shape.${token_type}" \
    fi
else
    log "Skip the training stages"
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
