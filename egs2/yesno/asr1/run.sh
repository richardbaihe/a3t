#!/bin/bash

set -e
set -u
set -o pipefail

train_set="train_nodev"
valid_set="train_dev"
test_sets="train_dev test_yesno"

asr_config=conf/train_asr.yaml
decode_config=conf/decode.yaml

./asr.sh                                        \
    --lang en                                   \
    --audio_format wav                          \
    --feats_type raw                            \
    --token_type char                           \
    --use_lm false                              \
    --asr_config "${asr_config}"                \
    --decode_config "${decode_config}"          \
    --train_set "${train_set}"                  \
    --valid_set "${valid_set}"                  \
    --test_sets "${test_sets}"                  \
    --srctexts "data/${train_set}/text" "$@"
