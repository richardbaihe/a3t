#!/usr/bin/env python3

# Copyright 2016  Allen Guo

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_base_path", default='/mnt/home/v_baihe/data/', type=str, required=True)
parser.add_argument("--dataset", default='libri-light-large-cut-30s', type=str, required=True)
args = parser.parse_args()

wav_dir = os.path.join(args.data_base_path, args.dataset)
feat_dir = os.path.join(args.data_base_path, 'feat_dir', args.dataset)
os.makedirs(feat_dir, exist_ok=True)

dir_name_list = os.listdir(wav_dir)
al = []
al_dic = {}

utt_path = []
for dir_name in sorted(dir_name_list):
    spk_path = os.path.join(wav_dir, dir_name)
    spk_list = os.listdir(spk_path)
    for spk_name in sorted(spk_list):
        file_list = os.listdir(os.path.join(spk_path, spk_name))
        if spk_name not in al_dic:
            al_dic[spk_name] = []
        prefix=spk_name + "-" + dir_name
        for f in sorted(file_list):
            # fsize = os.path.getsize(data_dir+f)
            # print(fsize)
            if 'flac' in f:
                f_name, _ = f.split('.')
                al.append(prefix+"-"+f_name)
                al_dic[spk_name].append(prefix+"-"+f_name)
                utt_path.append([prefix+"-"+f_name, "flac -c -d -s "+wav_dir+"/"+dir_name+"/"+spk_name+"/"+f+" |"])
                    # break

with open(os.path.join(feat_dir,'spk2utt'), 'w') as s2u:
    with open(os.path.join(feat_dir,'utt2spk'), 'w') as u2s:
        with open(os.path.join(feat_dir,'wav.scp'), 'w') as scp:
            spks = sorted(list(al_dic.keys()))
            for al in spks:
                print(al, file=s2u, end=' ')
                ll = sorted(al_dic[al])
                print(' '.join(ll), file=s2u)
                for utt in ll:
                    print(utt, al, file=u2s)
            sorted_utt_path = sorted(utt_path, key=lambda x:x[0])
            for i, j in sorted_utt_path:
                print(i, j, file=scp)