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
SECONDS=0

stage=1
stop_stage=100
ndev_utt=100
dataset="libri-light-large-cut-30s"
data_base_path="/mnt/home/v_baihe/data"

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

train_set="train_nodev"
train_dev="train_dev"


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"
    mkdir -p data/{${train_dev},${train_set}}
    python3 local/data_prep_librilight.py --dataset ${dataset} --data_base_path ${data_base_path}
    data_feat_path=${data_base_path}/feat_dir/${dataset}
    # make a dev set
    utils/subset_data_dir.sh --first ${data_feat_path} "${ndev_utt}" "${data_feat_path}/${train_dev}"
    n=$(($(wc -l < ${data_feat_path}/wav.scp) - ndev_utt))
    utils/subset_data_dir.sh --last ${data_feat_path} "${n}" "${data_feat_path}/${train_set}"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
