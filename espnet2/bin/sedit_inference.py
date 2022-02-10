#!/usr/bin/env python3

"""Script to run the inference of text-to-speeech model."""
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.tasks.mlm import MLMTask
from espnet2.tasks.tts import TTSTask
import matplotlib.pylab as plt
from parallel_wavegan.utils import download_pretrained_model
from pathlib import Path
import torch
import torch.nn.functional as F
from espnet2.torch_utils.device_funcs import to_device

import soundfile
import os
import math
import string
import numpy as np
from espnet2.fileio.read_text import read_2column_text,load_num_sequence_text
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, pad_list,make_pad_mask
from espnet2.bin.align_english import alignment
import librosa
import random

duration_path_dict = {
    "ljspeech":"/mnt/home/v_baihe/projects/espnet/egs2/ljspeech/tts1/exp/kan-bayashi/ljspeech_tts_train_conformer_fastspeech2_raw_phn_tacotron_g2p_en_no_space_train.loss.ave/train.loss.ave_5best.pth",
    "vctk": "/mnt/home/v_baihe/projects/espnet/egs2/vctk/tts1/exp/kan-bayashi/vctk_tts_train_gst+xvector_conformer_fastspeech2_transformer_teacher_raw_phn_tacotron_g2p_en_no_space_train.loss.ave/train.loss.ave_5best.pth",
    "libritts":"/mnt/home/v_baihe/projects/espnet/egs2/libritts/tts1/exp/kan-bayashi/libritts_tts_train_gst+xvector_conformer_fastspeech2_transformer_teacher_raw_phn_tacotron_g2p_en_no_space_train.loss/train.loss.ave_5best.pth"
}

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# default loading fs2 duration preditor trained with libritts
# default loading './exp/{}/train.loss.best.pth'
# default loading wavegan vocoder trained with libritts

PHONEME = '/mnt/home/jiahong/tools/english2phoneme/phoneme'
MODEL_DIR = '/mnt/home/jiahong/tools/alignment/aligner/english'


def get_baseline1(uid, prefix,vocoder, new_str, model, processor, xv_path='/mnt/home/v_baihe/projects/espnet/aggregate_output/vctk_spk2xvector.pt', return_mel=False):
    decode_conf = {}
    decode_conf.update(use_teacher_forcing=False)
    decode_conf.update(alpha=1.0)
    cfg = decode_conf
    if model.tts.spk_embed_dim is not None:
        spk_id = uid.split("_")[0]
        spk2xvector = torch.load(xv_path)
        spemd = spk2xvector[spk_id]
    else:
        spemd = None
    full_origin_str,wav_path = read_data(uid, prefix)
    wav_org, rate = librosa.load(wav_path, sr=model.feats_extract.fs)

    data = processor("<dummy>", dict(text=new_str, speech=wav_org))
    batch = dict(text=data['text'], speech=data['speech'], spembs=spemd)
    batch = to_device(batch, 'cpu')
    output_dict = model.inference(**batch, **cfg)
    if output_dict.get("feat_gen_denorm") is not None:
        out_feat = output_dict["feat_gen_denorm"]
    else:
        out_feat = output_dict["feat_gen"]
    if return_mel:
        return out_feat
    baseline1_wav = vocoder(out_feat).detach().float().data.cpu().numpy()
    return baseline1_wav

def get_baseline2(uid, prefix,vocoder, new_str,target_str, model, processor,xv_path='/mnt/home/v_baihe/projects/espnet/aggregate_output/vctk_spk2xvector.pt'):
    decode_conf = {}
    decode_conf.update(use_teacher_forcing=False)
    decode_conf.update(alpha=1.0)
    cfg = decode_conf
    if model.tts.spk_embed_dim is not None:
        spk_id = uid.split("_")[0]
        spk2xvector = torch.load(xv_path)
        spemd = spk2xvector[spk_id]
    else:
        spemd = None
    full_origin_str,wav_path = read_data(uid, prefix)
    wav_org, rate = librosa.load(wav_path, sr=model.feats_extract.fs)
    origin_speech = torch.tensor(np.array(wav_org,dtype=np.float32)).unsqueeze(0)
    speech_lengths = torch.tensor(len(wav_org)).unsqueeze(0)
    input_feat, feats_lengths = model.feats_extract(origin_speech, speech_lengths)
    input_feat = input_feat.squeeze(0)
    fs, hop_length = (model.feats_extract.fs, model.feats_extract.hop_length)

    mfa_start, mfa_end, old_phns, new_phns, span_tobe_replaced, span_tobe_added = get_phns_and_spans(wav_path, full_origin_str, new_str)
    old_span = [int(mfa_start[span_tobe_replaced[0]]*fs/hop_length), int(mfa_start[span_tobe_replaced[1]]*fs/hop_length) ]


    data = processor("<dummy>", dict(text=target_str, speech=wav_org))
    batch = dict(text=data['text'], speech=data['speech'], spembs=spemd)
    batch = to_device(batch, 'cpu')
    output_dict = model.inference(**batch, **cfg)
    eos_duration = output_dict['duration'].tolist()[-1]
    if output_dict.get("feat_gen_denorm") is not None:
        out_feat = output_dict["feat_gen_denorm"]
    else:
        out_feat = output_dict["feat_gen"]
    out_feat_target = out_feat[:-eos_duration]
    out_feat = torch.cat([input_feat[:old_span[0]], out_feat_target, input_feat[old_span[1]:]])
    baseline2_wav = vocoder(out_feat).detach().float().data.cpu().numpy()
    return baseline2_wav

def get_baseline3(uid, prefix,vocoder, new_str, model, processor,xv_path='/mnt/home/v_baihe/projects/espnet/aggregate_output/vctk_spk2xvector.pt'):
    decode_conf = {}
    decode_conf.update(use_teacher_forcing=False)
    decode_conf.update(alpha=1.0)
    cfg = decode_conf
    if model.tts.spk_embed_dim is not None:
        spk_id = uid.split("_")[0]
        spk2xvector = torch.load(xv_path)
        spemd = spk2xvector[spk_id]
    else:
        spemd = None
    full_origin_str,wav_path = read_data(uid, prefix)
    wav_org, rate = librosa.load(wav_path, sr=model.feats_extract.fs)
    origin_speech = torch.tensor(np.array(wav_org,dtype=np.float32)).unsqueeze(0)
    speech_lengths = torch.tensor(len(wav_org)).unsqueeze(0)
    input_feat, feats_lengths = model.feats_extract(origin_speech, speech_lengths)
    input_feat = input_feat.squeeze(0)
    fs, hop_length = (model.feats_extract.fs, model.feats_extract.hop_length)

    mfa_start, mfa_end, old_phns, new_phns, span_tobe_replaced, span_tobe_added = get_phns_and_spans(wav_path, full_origin_str, new_str)
    old_span = [int(mfa_start[span_tobe_replaced[0]]*fs/hop_length), int(mfa_start[span_tobe_replaced[1]]*fs/hop_length) ]

    data = processor("<dummy>", dict(text=new_str, speech=wav_org))
    batch = dict(text=data['text'], speech=data['speech'], spembs=spemd)
    batch = to_device(batch, 'cpu')
    output_dict = model.inference(**batch, **cfg)
    durations = output_dict['duration'].tolist()[:-1]
    eos_duration = output_dict['duration'].tolist()[-1]

    if output_dict.get("feat_gen_denorm") is not None:
        out_feat = output_dict["feat_gen_denorm"]
    else:
        out_feat = output_dict["feat_gen"]
    out_feat_target = out_feat[sum(durations[:span_tobe_added[0]]):sum(durations[:span_tobe_added[1]])]
    out_feat = torch.cat([input_feat[:old_span[0]], out_feat_target, input_feat[old_span[1]:]])
    baseline3_wav = vocoder(out_feat).detach().float().data.cpu().numpy()
    return baseline3_wav


def get_unk_phns(word_str):
    tmpbase = '/tmp/h32bai.'
    f = open(tmpbase + 'temp.words', 'w')
    f.write(word_str)
    f.close()
    os.system(PHONEME + ' ' + tmpbase + 'temp.words' + ' ' + tmpbase + 'temp.phons')
    f = open(tmpbase + 'temp.phons', 'r')
    lines2 = f.readline().strip().split()
    f.close()
    phns = []
    for phn in lines2:
        phons = phn.replace('\n', '').replace(' ', '')
        seq = []
        j = 0
        while (j < len(phons)):
            if (phons[j] > 'Z'):
                if (phons[j] == 'j'):
                    seq.append('JH')
                elif (phons[j] == 'h'):
                    seq.append('HH')
                else:
                    seq.append(phons[j].upper())
                j += 1
            else:
                p = phons[j:j+2]
                if (p == 'WH'):
                    seq.append('W')
                elif (p in ['TH', 'SH', 'HH', 'DH', 'CH', 'ZH', 'NG']):
                    seq.append(p)
                elif (p == 'AX'):
                    seq.append('AH0')
                else:
                    seq.append(p + '1')
                j += 2
        phns.extend(seq)
    return phns

def words2phns_yuan(line):
    # yuan's word2phns
    dictfile = MODEL_DIR+'/dict'
    tmpbase = '/tmp/h32bai.'
    line = line.strip()
    words = []
    for pun in [',', '.', ':', ';', '!', '?', '"', '(', ')', '--', '---']:
        line = line.replace(pun, ' ')
    for wrd in line.split():
        if (wrd[-1] == '-'):
            wrd = wrd[:-1]
        if (wrd[0] == "'"):
            wrd = wrd[1:]
        if wrd:
            words.append(wrd)
    ds = set([])
    word2phns_dict = {}
    with open(dictfile, 'r') as fid:
        for line in fid:
            word = line.split()[0]
            ds.add(word)
            if word not in word2phns_dict.keys():
                word2phns_dict[word] = " ".join(line.split()[1:])
    
    phns = []
    wrd2phns = {}
    for index, wrd in enumerate(words):
        if wrd == '[MASK]':
            wrd2phns[str(index)+"_"+wrd] = [wrd]
            phns.append(wrd)
        elif (wrd.upper() not in ds):
            wrd2phns[str(index)+"_"+wrd.upper()] = get_unk_phns(wrd)
            phns.extend(get_unk_phns(wrd))
        else:
            wrd2phns[str(index)+"_"+wrd.upper()] = word2phns_dict[wrd.upper()].split()
            phns.extend(word2phns_dict[wrd.upper()].split())

    return phns, wrd2phns

def load_vocoder(vocoder_tag="parallel_wavegan/libritts_parallel_wavegan.v1"):
    vocoder_tag = vocoder_tag.replace("parallel_wavegan/", "")
    vocoder_file = download_pretrained_model(vocoder_tag)
    vocoder_config = Path(vocoder_file).parent / "config.yml"


    vocoder = TTSTask.build_vocoder_from_file(
                    vocoder_config, vocoder_file, None, 'cpu'
                )
    return vocoder

def load_model(model_name):
    config_path='{}/config.yaml'.format(model_name)
    model_path = '{}/train.loss.best.pth'.format(model_name)
    if not os.path.exists(model_path):
        model_path = '{}/train.loss.best.pth'.format(model_name)

    mlm_model, args = MLMTask.build_model_from_file(config_file=config_path,
                                 model_file=model_path)
    return mlm_model, args

def get_text_and_wav(uid,prefix='dump/raw/dev-clean/',sr=24000):
    wav_path = read_2column_text(prefix+'wav.scp')[uid]
    text = read_2column_text(prefix+'text')[uid]
    # wav_org, rate = soundfile.read(wav_path, always_2d=False)
    wav_org, rate = librosa.load(wav_path, sr)
    return text, wav_org

def read_data(uid,prefix):
    mfa_text = read_2column_text(prefix+'/text')[uid]
    mfa_wav_path = read_2column_text(prefix+'/wav.scp')[uid]
    if 'mnt' not in mfa_wav_path:
        mfa_wav_path = prefix.split('dump')[0] + mfa_wav_path
    return mfa_text, mfa_wav_path
 
def get_align_data(uid,prefix):
    mfa_path = prefix+"mfa_"
    mfa_text = read_2column_text(mfa_path+'text')[uid]
    mfa_start = load_num_sequence_text(mfa_path+'start',loader_type='text_float')[uid]
    mfa_end = load_num_sequence_text(mfa_path+'end',loader_type='text_float')[uid]
    mfa_wav_path = read_2column_text(mfa_path+'wav.scp')[uid]
    return mfa_text, mfa_start, mfa_end, mfa_wav_path

def get_mel_span(uid, model_name,prefix):
    mlm_model,args = load_model(model_name)
    mfa_text, mfa_start, mfa_end, mfa_wav_path = get_align_data(uid,prefix)
    align_start=torch.tensor(mfa_start).unsqueeze(0)
    align_end =torch.tensor(mfa_end).unsqueeze(0)
    align_start = torch.floor(mlm_model.feats_extract.fs*align_start/mlm_model.feats_extract.hop_length).int()
    align_end = torch.floor(mlm_model.feats_extract.fs*align_end/mlm_model.feats_extract.hop_length).int()
    return align_start[0].tolist(), align_end[0].tolist(),mfa_text

def get_fs2_model(model_name):
    model, config = TTSTask.build_model_from_file(model_file=model_name)
    processor = TTSTask.build_preprocess_fn(config, train=False)
    # model = torch.load('/mnt/home/v_baihe/projects/espnet/egs2/{}/tts1/exp/fs2_model.pt'.format(model_name))
    # processor = torch.load('/mnt/home/v_baihe/projects/espnet/egs2/{}/tts1/exp/fs2_processor.pt'.format(model_name))
    return model, processor

def duration_predict(old_phns, fs, hop_length,fs2_model, fs2_processor,wav_org, sid=None):
    phns = [i if i != 'sp' else '<blank>' for i in old_phns ]
    text = fs2_processor.token_id_converter.tokens2ids(phns)
    text = torch.tensor(text).unsqueeze(0)
    text = F.pad(text, [0, 1], "constant", fs2_model.tts.eos)
    ilens = torch.tensor(text.shape[1]).unsqueeze(0)
    x_masks = fs2_model.tts._source_mask(ilens)
    if fs2_model.tts.spk_embed_dim is not None and sid is not None:
        spembs = torch.tensor(sid).unsqueeze(0)
    else:
        spembs = None
    hs, _ = fs2_model.tts.encoder(text, x_masks)
    if fs2_model.tts.use_gst:
        origin_speech = torch.tensor(np.array(wav_org,dtype=np.float32)).unsqueeze(0)
        speech_lengths = torch.tensor(len(wav_org)).unsqueeze(0)
        ys,_ = fs2_model.feats_extract(origin_speech, speech_lengths )
        style_embs = fs2_model.tts.gst(ys)
        hs = hs + style_embs.unsqueeze(1)
    if spembs is not None:
        hs = fs2_model.tts._integrate_with_spk_embed(hs, spembs)
    d_masks = make_pad_mask(ilens).to(text.device)
    d_outs = fs2_model.tts.duration_predictor.inference(hs, d_masks)  # (B, T_text)
    d_outs = d_outs*hop_length/fs
    return d_outs[0].tolist()[:-1]

def get_masked_mel_boundary(mfa_start, mfa_end, fs, hop_length, span_tobe_replaced):
    align_start=torch.tensor(mfa_start).unsqueeze(0)
    align_end =torch.tensor(mfa_end).unsqueeze(0)
    align_start = torch.floor(fs*align_start/hop_length).int()
    align_end = torch.floor(fs*align_end/hop_length).int()
    span_boundary=[align_start[0].tolist()[span_tobe_replaced[0]],align_end[0].tolist()[span_tobe_replaced[1]-1]]
    return span_boundary

def get_phns_and_spans(wav_path, old_str, new_str):
    old_phns, mfa_start, mfa_end = [], [], []
    times2,word2phns = alignment(wav_path, old_str)
    for item in times2:
        mfa_start.append(float(item[1]))
        mfa_end.append(float(item[2]))
        old_phns.append(item[0])

    new_phns, new_word2phns = words2phns_yuan(new_str)
    span_tobe_replaced = [0,len(old_phns)-1]
    span_tobe_added = [0,len(new_phns)-1]
    left_index = 0
    new_phns_left = []
    sp_count = 0
    # find the left different index
    for key in word2phns.keys():
        idx, wrd = key.split('_')
        if wrd=='sp':
            sp_count +=1
            new_phns_left.append('sp')
        else:
            idx = str(int(idx) - sp_count)
            if idx+'_'+wrd in new_word2phns:
                left_index+=len(new_word2phns[idx+'_'+wrd])
                new_phns_left.extend(word2phns[key].split())
            else:
                span_tobe_replaced[0] = len(new_phns_left)
                span_tobe_added[0] = len(new_phns_left)
                break
    # reverse word2phns and new_word2phns
    right_index = 0
    new_phns_right = []
    sp_count = 0
    word2phns_max_index = int(list(word2phns.keys())[-1].split('_')[0])
    new_word2phns_max_index = int(list(new_word2phns.keys())[-1].split('_')[0])
    new_phns_middle = []
    for key in list(word2phns.keys())[::-1]:
        idx, wrd = key.split('_')
        if wrd=='sp':
            sp_count +=1
            new_phns_right = ['sp']+new_phns_right
        else:
            idx = str(new_word2phns_max_index-(word2phns_max_index-int(idx)-sp_count))
            if idx+'_'+wrd in new_word2phns:
                right_index-=len(new_word2phns[idx+'_'+wrd])
                new_phns_right = word2phns[key].split() + new_phns_right
            else:
                span_tobe_replaced[1] = len(old_phns) - len(new_phns_right)
                new_phns_middle = new_phns[left_index:right_index]
                span_tobe_added[1] = len(new_phns_left) + len(new_phns_middle)
                if len(new_phns_middle) == 0:
                    span_tobe_added[1] = min(span_tobe_added[1]+1, len(new_phns))
                    span_tobe_added[0] = max(0, span_tobe_added[0]-1)
                    span_tobe_replaced[0] = max(0, span_tobe_replaced[0]-1)
                    span_tobe_replaced[1] = min(span_tobe_replaced[1]+1, len(old_phns))
                break
    new_phns = new_phns_left+new_phns_middle+new_phns_right

    return mfa_start, mfa_end, old_phns, new_phns, span_tobe_replaced, span_tobe_added

def duration_adjust_factor(original_dur, pred_dur, phns):
    length = 0
    accumulate = 0
    factor_list = []
    for ori,pred,phn in zip(original_dur, pred_dur,phns):
        if pred==0 or phn=='sp':
            continue
        else:
            # accumulate += ori/pred
            # length+=1
            factor_list.append(ori/pred)
    factor_list = np.array(factor_list)
    factor_list.sort()
    if len(factor_list)<5:
        return 1
    # length = int(len(factor_list)*0.9)
    length = 2
    return np.average(factor_list[length:-length])
    # return accumulate/length

def prepare_features_with_duration(mlm_model, old_str, new_str, wav_path,duration_preditor_path,sid=None):
    wav_org, rate = librosa.load(wav_path, sr=mlm_model.feats_extract.fs)
    fs = mlm_model.feats_extract.fs
    hop_length = mlm_model.feats_extract.hop_length

    mfa_start, mfa_end, old_phns, new_phns, span_tobe_replaced, span_tobe_added = get_phns_and_spans(wav_path, old_str, new_str)

    # if '[MASK]' in new_str:
    #     old_span_boundary = get_masked_mel_boundary(mfa_start, mfa_end, fs, hop_length, span_tobe_replaced)
    #     return wav_org, old_phns, mfa_start, mfa_end, old_span_boundary, old_span_boundary

    # 2. get new alignment start and end list
    fs2_model, fs2_processor = get_fs2_model(duration_preditor_path)
    old_durations = duration_predict(old_phns, fs, hop_length,fs2_model, fs2_processor,wav_org,sid=sid )    
    original_old_durations = [e-s for e,s in zip(mfa_end, mfa_start)]
    if '[MASK]' in new_str:
        new_phns = old_phns
        span_tobe_added = span_tobe_replaced
        d_factor_left = duration_adjust_factor(original_old_durations[:span_tobe_replaced[0]],old_durations[:span_tobe_replaced[0]], old_phns[:span_tobe_replaced[0]])
        d_factor_right = duration_adjust_factor(original_old_durations[span_tobe_replaced[1]:],old_durations[span_tobe_replaced[1]:], old_phns[span_tobe_replaced[1]:])
        d_factor = (d_factor_left+d_factor_right)/2
        new_durations_adjusted = [d_factor*i for i in old_durations]
    else:
        d_factor = duration_adjust_factor(original_old_durations,old_durations, old_phns)
    # d_factor = 2
        new_durations = duration_predict(new_phns, fs, hop_length,fs2_model, fs2_processor,wav_org ,sid=sid)
        new_durations_adjusted = [d_factor*i for i in new_durations]
        if old_phns[span_tobe_replaced[0]] == new_phns[span_tobe_added[0]]:
            new_durations_adjusted[span_tobe_added[0]] = original_old_durations[span_tobe_replaced[0]]
        if old_phns[span_tobe_replaced[1]] == new_phns[span_tobe_added[1]]:
            new_durations_adjusted[span_tobe_added[1]] = original_old_durations[span_tobe_replaced[1]]
    new_span_duration_sum = sum(new_durations_adjusted[span_tobe_added[0]:span_tobe_added[1]])
    old_span_duration_sum = sum(original_old_durations[span_tobe_replaced[0]:span_tobe_replaced[1]])
    duration_offset =  new_span_duration_sum - old_span_duration_sum
    new_mfa_start = mfa_start[:span_tobe_replaced[0]]
    new_mfa_end = mfa_end[:span_tobe_replaced[0]]
    for i in new_durations_adjusted[span_tobe_added[0]:span_tobe_added[1]]:
        if len(new_mfa_end) ==0:
            new_mfa_start.append(0)
            new_mfa_end.append(i)
        else:
            new_mfa_start.append(new_mfa_end[-1])
            new_mfa_end.append(new_mfa_end[-1]+i)
    new_mfa_start += [i+duration_offset for i in mfa_start[span_tobe_replaced[1]:]]
    new_mfa_end += [i+duration_offset for i in mfa_end[span_tobe_replaced[1]:]]
    
    # 3. get new wav 
    left_index = int(np.floor(mfa_start[span_tobe_replaced[0]]*fs))
    right_index = int(np.ceil(mfa_end[span_tobe_replaced[1]-1]*fs))
    new_blank_wav = np.zeros((int(np.ceil(new_span_duration_sum*fs)),), dtype=wav_org.dtype)
    new_wav_org = np.concatenate([wav_org[:left_index], new_blank_wav, wav_org[right_index:]])

    # 4. get old and new mel span to be mask
    old_span_boundary = get_masked_mel_boundary(mfa_start, mfa_end, fs, hop_length, span_tobe_replaced)
    new_span_boundary=get_masked_mel_boundary(new_mfa_start, new_mfa_end, fs, hop_length, span_tobe_added)
    
    return new_wav_org, new_phns, new_mfa_start, new_mfa_end, old_span_boundary, new_span_boundary

def prepare_features(mlm_model,processor, wav_path, old_str,new_str,duration_preditor_path, sid=None):

    wav_org, phns_list, mfa_start, mfa_end, old_span_boundary, new_span_boundary = prepare_features_with_duration(mlm_model, old_str, 
    new_str, wav_path,duration_preditor_path,sid=sid)
    
    speech = np.array(wav_org,dtype=np.float32)
    align_start=np.array(mfa_start)
    align_end =np.array(mfa_end)
    text = np.array(processor(uid='1', data={'text':" ".join(phns_list)})['text'])
    span_boundary = np.array(new_span_boundary)
    batch=[('1', {"speech":speech,"align_start":align_start,"align_end":align_end,"text":text,"span_boundary":span_boundary})]
    speech_lengths = torch.tensor(len(wav_org)).unsqueeze(0)
    
    return batch, speech_lengths, old_span_boundary, new_span_boundary

def decode_with_model(mlm_model, processor, collate_fn, wav_path, old_str, new_str,duration_preditor_path, sid=None, decoder=False,use_teacher_forcing=False):
    fs, hop_length = mlm_model.feats_extract.fs, mlm_model.feats_extract.hop_length

    batch,speech_lengths,old_span_boundary,new_span_boundary = prepare_features(mlm_model,processor,wav_path,old_str,new_str,duration_preditor_path, sid)
    feats = collate_fn(batch)[1]
    # wav_len * 80
    set_all_random_seed(9999)
    rtn = mlm_model.inference(**feats,span_boundary=new_span_boundary,use_teacher_forcing=use_teacher_forcing)
    output = rtn['feat_gen'] 
    if 0 in output[0].shape and 0 not in output[-1].shape:
        output_feat = torch.cat(output[1:-1]+[output[-1].squeeze()], dim=0).cpu()
    elif 0 not in output[0].shape and 0 in output[-1].shape:
        output_feat = torch.cat([output[0].squeeze()]+output[1:-1], dim=0).cpu()
    elif 0 in output[0].shape and 0 in output[-1].shape:
        output_feat = torch.cat(output[1:-1], dim=0).cpu()
    else:
        output_feat = torch.cat([output[0].squeeze(0)]+ output[1:-1]+[output[-1].squeeze(0)], dim=0).cpu()
    
    # wav_org, rate = soundfile.read(
    #             wav_path, always_2d=False)
    wav_org, rate = librosa.load(wav_path, sr=mlm_model.feats_extract.fs)
    origin_speech = torch.tensor(np.array(wav_org,dtype=np.float32)).unsqueeze(0)
    speech_lengths = torch.tensor(len(wav_org)).unsqueeze(0)
    input_feat, feats_lengths = mlm_model.feats_extract(origin_speech, speech_lengths)
    return wav_org, input_feat.squeeze(), output_feat, old_span_boundary, new_span_boundary, fs, hop_length


def decode_for_mcd(model_name, wav_path, old_str, new_str,duration_preditor_path, sid=None, decoder=False,use_teacher_forcing=True):

    mlm_model,train_args = load_model(model_name)
    mlm_model.eval()
    processor = MLMTask.build_preprocess_fn(train_args, False)
    collate_fn = MLMTask.build_collate_fn(train_args, False)
    fs, hop_length = mlm_model.feats_extract.fs, mlm_model.feats_extract.hop_length

    batch,speech_lengths,old_span_boundary,new_span_boundary = prepare_features(mlm_model,processor,wav_path,old_str,new_str,duration_preditor_path, sid)
    feats = collate_fn(batch)[1]
    # wav_len * 80
    set_all_random_seed(9999)
    rtn = mlm_model.inference(**feats,span_boundary=new_span_boundary,use_teacher_forcing=use_teacher_forcing)
    output = rtn['feat_gen'] 
    if 0 in output[0].shape and 0 not in output[-1].shape:
        output_feat = torch.cat(output[1:-1]+[output[-1].squeeze()], dim=0).cpu()
    elif 0 not in output[0].shape and 0 in output[-1].shape:
        output_feat = torch.cat([output[0].squeeze()]+output[1:-1], dim=0).cpu()
    elif 0 in output[0].shape and 0 in output[-1].shape:
        output_feat = torch.cat(output[1:-1], dim=0).cpu()
    else:
        output_feat = torch.cat([output[0].squeeze(0)]+ output[1:-1]+[output[-1].squeeze(0)], dim=0).cpu()
    
    # wav_org, rate = soundfile.read(
    #             wav_path, always_2d=False)
    wav_org, rate = librosa.load(wav_path, sr=mlm_model.feats_extract.fs)
    origin_speech = torch.tensor(np.array(wav_org,dtype=np.float32)).unsqueeze(0)
    speech_lengths = torch.tensor(len(wav_org)).unsqueeze(0)
    input_feat, feats_lengths = mlm_model.feats_extract(origin_speech, speech_lengths)

    mfa_start, mfa_end, old_phns, new_phns, span_tobe_replaced, span_tobe_added = get_phns_and_spans(wav_path, old_str, new_str)

    return input_feat.squeeze(), output_feat,span_tobe_replaced, old_span_boundary, new_span_boundary

def get_mlm_output(model_name, wav_path, old_str, new_str,duration_preditor_path, sid=None, decoder=False,use_teacher_forcing=False):

    mlm_model,train_args = load_model(model_name)
    mlm_model.eval()
    processor = MLMTask.build_preprocess_fn(train_args, False)
    collate_fn = MLMTask.build_collate_fn(train_args, False)

    return decode_with_model(mlm_model, processor, collate_fn, wav_path, old_str, new_str,duration_preditor_path, sid=sid, decoder=decoder, use_teacher_forcing=use_teacher_forcing)

def plot_data(data, figsize=(16, 4), span_boundary=None, titles=None):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none')
        if span_boundary:
            for x in span_boundary[i]:
                axes[i].axvline(x=x,color='red')
        if titles is not None:
            axes[i].title.set_text(titles[i])

def plot_mel_and_vocode_wav(model_name, wav_path,full_origin_str, old_str, new_str, vocoder,duration_preditor_path,sid=None, non_autoreg=True):
    wav_org, input_feat, output_feat, old_span_boundary, new_span_boundary, fs, hop_length = get_mlm_output(
                                                            model_name,
                                                            wav_path,
                                                            old_str,
                                                            new_str, 
                                                            duration_preditor_path,
                                                            use_teacher_forcing=non_autoreg,
                                                            sid=sid
                                                            )
    # return output_feat[new_span_boundary[0]:new_span_boundary[1]]
    plot_data((input_feat.float().data.cpu().numpy().T,
              output_feat.float().data.cpu().numpy().T),
              span_boundary=[old_span_boundary,new_span_boundary],
             titles=['original spec', 'new spec'])
    replaced_wav = vocoder(output_feat).detach().float().data.cpu().numpy()
    # vocoder_origin_wav = vocoder(input_feat).detach().float().data.cpu().numpy()

    old_time_boundary = [hop_length * x  for x in old_span_boundary]
    new_time_boundary = [hop_length * x  for x in new_span_boundary]
    wav_org_replaced = np.concatenate([wav_org[:old_time_boundary[0]], replaced_wav[new_time_boundary[0]:new_time_boundary[1]], wav_org[old_time_boundary[1]:]])
    # "origin_vocoder":vocoder_origin_wav, 
    data_dict = {"prediction":replaced_wav, 
                "orgin_replaced":wav_org_replaced,
                "origin":wav_org}
    return data_dict, old_span_boundary

def merge_two_data(uid1,uid2, prefix):
    # text
    # wav
    # align_star
    # align_end
    str1,wav_path1 = read_data(uid1, prefix)
    str2,wav_path2 = read_data(uid2, prefix)
    new_str = str1+' '+str2
    wav_org1, sr = librosa.load(wav_path1, sr=24000)
    wav_org2, sr = librosa.load(wav_path2, sr=24000)
    new_wav = np.concatenate([wav_org1, wav_org2])
    _, mfa_start1, mfa_end1, _ = get_align_data(uid1,prefix)
    _, mfa_start2, mfa_end2, _ = get_align_data(uid2,prefix)
    new_mfa_start = mfa_start1+[i+mfa_end1[-1] for i in mfa_start2]
    new_mfa_end = mfa_end1+[i+mfa_end1[-1] for i in mfa_end2]
    new_prefix = os.path.join(prefix,'merged')
    os.makedirs(new_prefix,exist_ok=True)
    with open(os.path.join(new_prefix,'text'),'w') as f_out:
        f_out.write(uid2+' '+new_str)
    with open(os.path.join(new_prefix,'wav.scp'),'w') as f_out:
        new_wav_path = os.path.join(new_prefix,'merged.wav')
        soundfile.write(new_wav_path, new_wav, sr)
        f_out.write(uid2+' '+new_wav_path)
    with open(os.path.join(new_prefix,'mfa_start'),'w') as f_out:
        f_out.write(uid2+' '+" ".join([str(i) for i in new_mfa_start]))
    with open(os.path.join(new_prefix,'mfa_end'),'w') as f_out:
        f_out.write(uid2+' '+" ".join([str(i) for i in new_mfa_end]))

def read_emotion_data(speaker_id, text_tag, emo_tag, level_tag):
    # The sentences were presented using different emotion (in parentheses is the three letter code used in the third part of the filename):

    # - Anger (ANG)
    # - Disgust (DIS)
    # - Fear (FEA)
    # - Happy/Joy (HAP)
    # - Neutral (NEU)
    # - Sad (SAD)

    # and emotion level (in parentheses is the two letter code used in the fourth part of the filename):

    # - Low (LO)
    # - Medium (MD)
    # - High (HI)
    # - Unspecified (XX)
    text_dict = {
        "IEO": "It's eleven o'clock",
        "TIE": "That is exactly what happened",
        "IOM": "I'm on my way to the meeting",
        "IWW": "I wonder what this is about" ,
        "TAI": "The airplane is almost full",
        "MTI": "Maybe tomorrow it will be cold",
        "IWL": "I would like a new alarm clock",
        "ITH": "I think I have a doctor's appointment",
        "DFA": "Don't forget a jacket",
        "ITS": "I think I've seen this before",
        "TSI": "The surface is slick",
        "WSI": "We'll stop in a couple of minutes"
    }
    filename = "/mnt/scratch/jiahong/emotion_datasets/CREMA-D/AudioWAV/"+ '_'.join([speaker_id, text_tag, emo_tag, level_tag ])+'.wav'
    return text_dict[text_tag], filename
    
    
def test_libritts(uid, vocoder, prefix='dump/raw/dev-clean/', model_name="conformer",old_str="", new_str=""):
    # uid = "1272_128104_000003_000001"
    duration_preditor_path= duration_path_dict['libritts']
    sid = uid.split('_')[0]
    xv_path = '/mnt/home/v_baihe/projects/espnet/aggregate_output/libritts_spk2xvector.pt'
    spk2xvector = torch.load(xv_path) 
    spemd = spk2xvector[sid]
    full_origin_str,wav_path = read_data(uid, prefix)
    print(full_origin_str)
    if not old_str:
        old_str = full_origin_str
    if not new_str:
        new_str = input("input the new string:")
    results_dict, old_span = plot_mel_and_vocode_wav(model_name, wav_path,full_origin_str, old_str, new_str,vocoder,duration_preditor_path,sid=spemd)
    return results_dict

def test_vctk(uid, vocoder, prefix='dump/raw/dev', model_name="conformer", old_str="",new_str=""):
    # sid = uid.split('_')[0]
    # duration_preditor_path = duration_path_dict['vctk']
    # xv_path = '/mnt/home/v_baihe/projects/espnet/aggregate_output/vctk_spk2xvector.pt'
    # spk2xvector = torch.load(xv_path) 
    # spemd = spk2xvector[sid]
    duration_preditor_path = duration_path_dict['ljspeech']
    spemd = None
    full_origin_str,wav_path = read_data(uid, prefix)
    print(full_origin_str)
    if not old_str:
        old_str = full_origin_str
    if not new_str:
        new_str = input("input the new string:")
    results_dict, old_span = plot_mel_and_vocode_wav(model_name, wav_path,full_origin_str, old_str, new_str,vocoder,duration_preditor_path,sid=spemd)
    return results_dict

def plot_vctk(uid, vocoder=None, prefix='dump/raw/dev', model_name="conformer", old_str="",new_str=""):
    duration_preditor_path = duration_path_dict['ljspeech']
    spemd = None
    full_origin_str,wav_path = read_data(uid, prefix)
    print(full_origin_str)
    if not old_str:
        old_str = full_origin_str
    if not new_str:
        new_str = input("input the new string:")
    wav_org, input_feat, output_feat, old_span_boundary, new_span_boundary, fs, hop_length = get_mlm_output(
                                                            model_name,
                                                            wav_path,
                                                            old_str,
                                                            new_str, 
                                                            duration_preditor_path,
                                                            use_teacher_forcing=True,
                                                            sid=spemd
                                                            )
    
    fs2_model_path = "/mnt/home/v_baihe/projects/espnet/egs2/vctk/tts1/exp/kan-bayashi/vctk_tts_train_gst+xvector_conformer_fastspeech2_transformer_teacher_raw_phn_tacotron_g2p_en_no_space_train.loss.ave/train.loss.ave_5best.pth"
    fs2_model, fs2_processor = get_fs2_model(fs2_model_path)
    tts_feat = get_baseline1(uid, prefix,vocoder, old_str, fs2_model, fs2_processor,return_mel=True)
    results_dict = {
        "original_mel": input_feat.float().data.cpu().numpy().T,
        "sedit_mel": output_feat.float().data.cpu().numpy().T,
        "tts_mel": tts_feat.float().data.cpu().numpy().T,
        "old_span": old_span_boundary,
        "new_span": new_span_boundary
    }

    return results_dict


def plot_ljspeech(uid, vocoder=None, prefix='dump/raw/dev', model_name="conformer", old_str="",new_str=""):
    duration_preditor_path = None#duration_path_dict['ljspeech']
    spemd = None
    full_origin_str,wav_path = read_data(uid, prefix)
    print(full_origin_str)
    if not old_str:
        old_str = full_origin_str
    if not new_str:
        new_str = input("input the new string:")
    wav_org, input_feat, output_feat, old_span_boundary, new_span_boundary, fs, hop_length = get_mlm_output(
                                                            model_name,
                                                            wav_path,
                                                            old_str,
                                                            new_str, 
                                                            duration_preditor_path,
                                                            use_teacher_forcing=True,
                                                            sid=spemd
                                                            )
    
    fs2_model_path = "/mnt/home/v_baihe/projects/espnet/egs2/ljspeech/tts1/exp/kan-bayashi/ljspeech_tts_train_conformer_fastspeech2_raw_phn_tacotron_g2p_en_no_space_train.loss.ave/train.loss.ave_5best.pth"
    fs2_model, fs2_processor = get_fs2_model(fs2_model_path)
    tts_feat = get_baseline1(uid, prefix,vocoder, old_str, fs2_model, fs2_processor,return_mel=True)
    results_dict = {
        "original_mel": input_feat.float().data.cpu().numpy().T,
        "sedit_mel": output_feat.float().data.cpu().numpy().T,
        "tts_mel": tts_feat.float().data.cpu().numpy().T,
        "old_span": old_span_boundary,
        "new_span": new_span_boundary
    }

    return results_dict

def test_ljspeech(uid, vocoder, prefix='dump/raw/dev', model_name="conformer", old_str="",new_str=""):
    duration_preditor_path = duration_path_dict['ljspeech']
    full_origin_str,wav_path = read_data(uid, prefix)
    print(full_origin_str)
    if not old_str:
        old_str = full_origin_str
    if not new_str:
        new_str = input("input the new string:")
    results_dict, old_span = plot_mel_and_vocode_wav(model_name, wav_path,full_origin_str, old_str, new_str,vocoder,duration_preditor_path)
    return results_dict

def test_cremad(uid, vocoder, model_name="conformer", old_str="", new_str=""):
    duration_preditor_path = duration_path_dict['ljspeech']
    speaker_id, text_tag,emo_tag,level_tag=uid.split("_")
    full_origin_str,wav_path = read_emotion_data(speaker_id, text_tag, emo_tag, level_tag)
    print(full_origin_str)
    if not old_str:
        old_str = full_origin_str
    if not new_str:
        new_str = input("input the new string:")
    results_dict, old_span = plot_mel_and_vocode_wav(model_name, wav_path,full_origin_str, old_str, new_str,vocoder,duration_preditor_path)
    return results_dict


if __name__ == "__main__":
    # vctk_vocoder = load_vocoder('vctk_parallel_wavegan.v1.long')
    # uid = "1090_ITS_HAP_XX"
    # model_name="/mnt/home/v_baihe/projects/espnet/egs2/vctk/sedit/exp/conformer"
    # new_str = "I think I've seen [MASK]"
    # data_dict = test_cremad(uid,vctk_vocoder, model_name, new_str=new_str)
    # display_audios(data_dict,sr=24000)

    # vocoder = load_vocoder('vctk_parallel_wavegan.v1.long')
    # model_name="/mnt/home/v_baihe/projects/espnet/egs2/vctk/sedit/exp/conformer"
    # uid = "1090_ITS_SAD_XX"
    # new_str = "I think I've seen this [MASK]"
    # data_dict = test_cremad(uid,vocoder,model_name,new_str)

    vocoder = load_vocoder('ljspeech_parallel_wavegan.v1.long')
    # model_name="/mnt/home/v_baihe/projects/espnet/egs2/ljspeech/sedit/exp/conformer"
    # uid = "LJ049-0010"
    # prefix='/mnt/home/v_baihe/projects/espnet/egs2/ljspeech/sedit/dump/raw/dev/'
    # new_str = "who responded to the happy event with dispatch."
    # data_dict = test_ljspeech(uid,vocoder, prefix, model_name, new_str=new_str)
    uid = "LJ050-0045"
    model_name="/mnt/home/v_baihe/projects/espnet/egs2/ljspeech/sedit/exp/ablation_transformer"
    prefix='/mnt/home/v_baihe/projects/espnet/egs2/ljspeech/sedit/dump/raw/eval1/'
    new_str = "and of the [MASK] Projects Agency of the Department of Defense,"
    data_dict = plot_ljspeech(uid,prefix=prefix,model_name=model_name,new_str=new_str)

    # vocoder = load_vocoder('vctk_parallel_wavegan.v1.long')
    # model_name="/mnt/home/v_baihe/projects/espnet/egs2/vctk/sedit/exp/conformer"
    # uid = "p243_313"
    # # prefix='/mnt/home/v_baihe/projects/espnet/egs2/vctk/sedit/dump/raw/tr_no_dev/'
    # prefix = '/mnt/home/v_baihe/projects/espnet/egs2/vctk/sedit/data/tr_no_dev/'
    # new_str = "for that [MASK] given"
    # data_dict = test_vctk(uid,vocoder,prefix,model_name,new_str=new_str)

    # libritts_vocoder = load_vocoder('libritts_parallel_wavegan.v1')
    # model_name="/mnt/home/v_baihe/projects/espnet/egs2/libritts/sedit/exp/conformer"
    # uid1 = '1089_134686_000007_000005'
    # uid2 = '1089_134686_000007_000003'
    # prefix='/mnt/home/v_baihe/projects/espnet/egs2/libritts/sedit/dump/raw/test-clean/'
    # merge_two_data(uid1,uid2,prefix)
    # new_str="[MASK] What music?"
    # data_dict = test_libritts(uid2,libritts_vocoder,os.path.join(prefix,'merged'),model_name,new_str=new_str)
    # display_audios(data_dict,sr=24000)
