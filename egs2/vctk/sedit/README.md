# 1. ENV
```
set cmd_backend='local' in cmd.sh if you are using a local gpu machine
```
```
set cmd_backend='slurm' in cmd.sh if you are using slurm
```
Known issues:
```
# missing package bc
apt-get install bc

# build sox by yourself to support flac type
sudo apt install flac
wget https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2.tar.gz
tar -zxvf sox-14.4.2.tar.gz
cd sox-14.4.2
./configure
sudo make -s && sudo make install
# check if you are using the recent installed sox
which sox
```

# 2. Prepare data
## Download vctk, preprocess, prepare vocab
```
./run.sh --stage 0 --stop_stage 5
```
## Alignment
```
../../../espnet2/bin/align_english.py
```

## :star: To train a model with seen speakers and eval with unseen speakers
First maintain a list of speakers that you want to exclude from training,
then exclude them from all .scp.

## Statistics of the training features lengths
```
./run.sh --stage 5 --stop_stage 6
```

# 3. Training
```
./run.sh --stage 6 --stop_stage 7
```
