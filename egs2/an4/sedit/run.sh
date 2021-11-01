#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train-clean-460
valid_set=dev-clean
test_sets="dev-clean test-clean"

nnodes=2
ngpu=8
fs=24000
n_fft=2048
n_shift=300
win_length=1200

opts=
if [ "${fs}" -eq 24000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

train_config=conf/train_mlm.yaml
# inference_config=conf/decode.yaml

cleaner=tacotron
g2p=g2p_en_no_space # or g2p_en
local_data_opts="--trim_all_silence true" # trim all silence in the audio

exp_name="exp/mlm_460h_0.25p_l1loss"

./mlm.sh \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --lang en \
    --stage 7 \
    --stop_stage 7 \
    --ngpu ${ngpu} \
    --num_nodes ${nnodes} \
    --dataset_name libritts \
    --token_type bpe \
    --feats_type raw \
    --local_data_opts "${local_data_opts}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    --mlm_config "${train_config}" \
    --mlm_exp "${exp_name}" \
    ${opts} \ "$@"
