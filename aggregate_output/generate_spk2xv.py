import kaldiio,os
from tqdm import tqdm
import torch

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
datasets=['tr_no_dev','dev','eval1']
spk2xvector = {}
for dataset in datasets:
    xv_path = f'{SCRIPT_DIR}/../egs2/vctk/tts1/dump/xvector/{dataset}/xvector.scp'
    new_xv_path = f'{SCRIPT_DIR}/../egs2/vctk/tts1/dump/xvector/{dataset}/xvector_abs_path.scp'
    with open(new_xv_path,'w') as f_out:
        for line in open(xv_path,'r').readlines():
            key, path = line.split()
            f_out.write(" ".join([key, os.path.join(f'{dataset}/../egs2/vctk/tts1/', path)])+'\n')
    xv_loader = kaldiio.load_scp(new_xv_path)

    for uid in tqdm(xv_loader._dict.keys()):
        sid = uid.split("_")[0]
        if sid in spk2xvector:
            continue
        else:
            spk2xvector[sid] = xv_loader[uid]
torch.save(spk2xvector,"vctk_spk2xvector.pt")

datasets=['train-clean-460','dev-clean','test-clean']
spk2xvector = {}
for dataset in datasets:
    xv_path = f'{SCRIPT_DIR}/../egs2/libritts/tts1/dump/xvector/{dataset}/xvector.scp'
    new_xv_path = f'{SCRIPT_DIR}/../egs2/libritts/tts1/dump/xvector/{dataset}/xvector_abs_path.scp'
    with open(new_xv_path,'w') as f_out:
        for line in open(xv_path,'r').readlines():
            key, path = line.split()
            f_out.write(" ".join([key, os.path.join(f'{SCRIPT_DIR}/../egs2/libritts/tts1/', path)])+'\n')
    xv_loader = kaldiio.load_scp(new_xv_path)

    for uid in tqdm(xv_loader._dict.keys()):
        sid = uid.split("_")[0]
        if sid in spk2xvector:
            continue
        else:
            spk2xvector[sid] = xv_loader[uid]
torch.save(spk2xvector,"libritts_spk2xvector.pt")
