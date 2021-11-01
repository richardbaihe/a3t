#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[1;36m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color
EXCLUDELIST=asimov-[49,59,54,55,84,85,91,92,96,99,204,163,162,186,209,174,222]

ngpu=8
num_nodes=2

myexpname='asr_train_asr_conformer7_n_fft512_hop_length256_raw_en_bpe5000_sp'
lm_exp="lm_train_lm_transformer2_en_bpe5000"
train_set="train_960"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/tuning/train_asr_conformer7_n_fft512_hop_length256.yaml
lm_config=conf/tuning/train_lm_transformer2.yaml
inference_config=conf/decode_asr.yaml

# num_iters_per_epoch
. utils/parse_options.sh || exit 1;


BASE="srun python -m espnet2.bin.asr_train"
# --batch_size 10 --num_iters_per_epoch 4000 
ARGS="--use_preprocessor true --bpemodel data/en_token_list/bpe_unigram5000/bpe.model --token_type bpe --token_list data/en_token_list/bpe_unigram5000/tokens.txt --non_linguistic_symbols none --cleaner none --g2p none --valid_data_path_and_name_and_type dump/raw/dev/wav.scp,speech,sound --valid_data_path_and_name_and_type dump/raw/dev/text,text,text --valid_shape_file exp/asr_stats_raw_en_bpe5000_sp/valid/speech_shape --valid_shape_file exp/asr_stats_raw_en_bpe5000_sp/valid/text_shape.bpe --resume true --fold_length 80000 --fold_length 150 --output_dir exp/asr_train_asr_conformer7_n_fft512_hop_length256_raw_en_bpe5000_sp --config conf/tuning/train_asr_conformer7_n_fft512_hop_length256.yaml --frontend_conf fs=16k --normalize=global_mvn --normalize_conf stats_file=exp/asr_stats_raw_en_bpe5000_sp/train/feats_stats.npz --train_data_path_and_name_and_type dump/raw/train_960_sp/wav.scp,speech,sound --train_data_path_and_name_and_type dump/raw/train_960_sp/text,text,text --train_shape_file exp/asr_stats_raw_en_bpe5000_sp/train/speech_shape --train_shape_file exp/asr_stats_raw_en_bpe5000_sp/train/text_shape.bpe"

DIST_ARGS="--ngpu ${ngpu} --dist_launcher slurm --dist_init_method file://$(pwd)/exp/${myexpname}/.dist_init_$(openssl rand -base64 -hex 12) --multiprocessing_distributed true"
# V100x8,P100,2080Ti_mlong,1080Ti_mlong,TitanXx8_mlong,2080Ti,1080Ti,TitanXx8
sbatch --job-name ${myexpname} --gres gpu:${ngpu} --nodes ${num_nodes} --partition V100x8 \
    --exclude $EXCLUDELIST \
    --wrap " $BASE $ARGS $DIST_ARGS " 

# srun --mem=32G --cpus-per-task=32 --time=1:30:0  --gres=gpu:1 --partition V100x8 --pty bash