#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

nnodes=1
ngpu=8
fs=24000
n_fft=2048
n_shift=300
win_length=1200
start_stage=0
stop_stage=1
exp_name=dev

opts=
if [ "${fs}" -eq 48000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

train_set=tr_no_dev
valid_set=dev
test_sets="dev eval1"

train_config=conf/train.yaml
inference_config=conf/decode.yaml

# g2p=g2p_en # Include word separator
g2p=g2p_en_no_space # Include no word separator

# train_set=unseen_tr_no_dev
# valid_set=dev
# test_sets="unseen_dev unseen_eval1"


# train_config=conf/train.yaml
# inference_config=conf/decode.yaml

# # g2p=g2p_en # Include word separator
# g2p=g2p_en_no_space # Include no word separator


# opts=
# if [ "${fs}" -eq 22050 ]; then
#     # To suppress recreation, specify wav format
#     opts="--audio_format wav "
# else
#     opts="--audio_format flac "
# fi

# . utils/parse_options.sh

./mlm.sh \
    --lang en \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length  "${win_length}" \
    --stage ${start_stage} \
    --stop_stage ${stop_stage} \
    --ngpu ${ngpu} \
    --num_nodes ${nnodes} \
    --dataset_name vctk \
    --token_type phn \
    --phn_as_word true \
    --feats_type raw \
    --cleaner none \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    --mlm_config "${train_config}" \
    --mlm_exp "exp/${exp_name}" \
    ${opts} "$@"
