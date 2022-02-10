from pathlib import Path
import os
from espnet2.bin.sedit_inference import *
import soundfile
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
fs2_path_dict={
    "LJSpeech":"/mnt/home/v_baihe/projects/espnet/egs2/ljspeech/tts1/exp/kan-bayashi/ljspeech_tts_train_conformer_fastspeech2_raw_phn_tacotron_g2p_en_no_space_train.loss.ave/train.loss.ave_5best.pth",
    "VCTK":"/mnt/home/v_baihe/projects/espnet/egs2/vctk/tts1/exp/kan-bayashi/vctk_tts_train_gst+xvector_conformer_fastspeech2_transformer_teacher_raw_phn_tacotron_g2p_en_no_space_train.loss.ave/train.loss.ave_5best.pth"
}
baseline4_model_dict = {
    "LJSpeech":"/mnt/home/v_baihe/projects/espnet/egs2/ljspeech/sedit/exp/conformer_no_sega",
    "VCTK":"/mnt/home/v_baihe/projects/espnet/egs2/vctk/sedit/exp/conformer_no_sega"
}
model_dict = {
    "LJSpeech":"/mnt/home/v_baihe/projects/espnet/egs2/ljspeech/sedit/exp/conformer",
    "VCTK":"/mnt/home/v_baihe/projects/espnet/egs2/vctk/sedit/exp/conformer"
}

prefix_dict = {
    "LJSpeech":"/mnt/home/v_baihe/projects/espnet/egs2/ljspeech/sedit/dump/raw/tr_no_dev/",
    "VCTK":"/mnt/home/v_baihe/projects/espnet/egs2/vctk/sedit/data/tr_no_dev/",
}

vocoder_dict = {
    "LJSpeech":'ljspeech_parallel_wavegan.v1.long',
    "VCTK":'vctk_parallel_wavegan.v1.long'
}
sr_dict = {
    "LJSpeech":22050,
    "VCTK":24000
}
decode_fn_dict = {
    "LJSpeech":test_ljspeech,
    "VCTK":test_vctk
}

def parse_infofile(path_in_str, dataset):
    if dataset == 'LJSpeech':
        uid = path_in_str.split('/')[-2].split('_')[1]
    elif dataset == 'VCTK':
        uid = "_".join(path_in_str.split('/')[-2].split('_')[1:3])
    new_text = ""
    text_to_replace_or_insert = ""
    with open(path_in_str, "r") as f:
        for line in f.readlines():
            line = line.strip().split(":")
            key, value = line[0], line[1:]
            if key == 'text_to_insert' or key == 'text_to_replace':
                text_to_replace_or_insert = value[0]
            # if key == "audio_file":
            #     uid = value[0].split('/')[-1].split('.')[0]
            if key == "modified_word_list" or key=='inserted_word_list':
                new_text = value[0]
                return uid, new_text,text_to_replace_or_insert

    return -1,-1,-1
operations = ["insert_text", "modify_text"]
#  VCTK, LJSpeech
for dataset in ["VCTK","LJSpeech"]:
    model_name = model_dict[dataset]
    prefix = prefix_dict[dataset]
    vocoder = load_vocoder(vocoder_dict[dataset])
    sr = sr_dict[dataset]
    decode_fn = decode_fn_dict[dataset]
    fs2_model_path = fs2_path_dict[dataset]
    fs2_model, fs2_processor = get_fs2_model(fs2_model_path)
    for oper in operations:
        pathlist = Path(os.path.join(dataset,oper)).glob("**/**/utterance_info.txt")
        for infofile_path in tqdm(pathlist):
            path_in_str = str(infofile_path)
            uid, new_str,text_to_replace_or_insert = parse_infofile(path_in_str,dataset)
            if uid==-1 or new_str==-1:
                raise NotImplementedError()
            # # sedit
            # data_dict = decode_fn(uid,vocoder,prefix,model_name,new_str=new_str)
            # results = data_dict['orgin_replaced']
            # new_wav_path = path_in_str.replace("utterance_info.txt","sedit.wav")
            # soundfile.write(new_wav_path, results, sr)
            # baseline 1
            baseline1_wav = get_baseline1(uid, prefix,vocoder, new_str, fs2_model, fs2_processor)
            new_wav_path = path_in_str.replace("utterance_info.txt","baseline1.wav")
            soundfile.write(new_wav_path, baseline1_wav, sr)
            # baseline 2
            baseline2_wav = get_baseline2(uid, prefix,vocoder, new_str,text_to_replace_or_insert, fs2_model, fs2_processor)
            new_wav_path = path_in_str.replace("utterance_info.txt","baseline2.wav")
            soundfile.write(new_wav_path, baseline2_wav, sr)
            # baseline 3
            baseline3_wav = get_baseline3(uid, prefix,vocoder, new_str, fs2_model, fs2_processor)
            new_wav_path = path_in_str.replace("utterance_info.txt","baseline3.wav")
            soundfile.write(new_wav_path, baseline3_wav, sr)
            # # baseline 4
            # data_dict = decode_fn(uid,vocoder,prefix,baseline4_model_dict[dataset],new_str=new_str)
            # baseline4_wav = data_dict['orgin_replaced']
            # new_wav_path = path_in_str.replace("utterance_info.txt","baseline4.wav")
            # soundfile.write(new_wav_path, baseline4_wav, sr)
            
