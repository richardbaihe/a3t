import os
from espnet2.bin.sedit_inference import *
from espnet2.torch_utils.device_funcs import to_device
import kaldiio
import soundfile
from tqdm import tqdm

set_all_random_seed(9999)
def save_wav_to_path(wav_input, left_index,right_index, path, prefix, sr,uid):
    wav_replaced = wav_input[left_index:right_index]
    wav_unreplaced = np.concatenate([wav_input[:left_index], wav_input[right_index:]])
    os.makedirs(os.path.join(path, prefix,'full'),exist_ok=True)
    os.makedirs(os.path.join(path, prefix,'replaced'),exist_ok=True)
    os.makedirs(os.path.join(path, prefix,'unreplaced'),exist_ok=True)
    soundfile.write(os.path.join(path, prefix,'full', uid+'.wav'), wav_input, sr)
    soundfile.write(os.path.join(path, prefix,'replaced', uid+'.wav'), wav_replaced, sr)
    soundfile.write(os.path.join(path, prefix,'unreplaced', uid+'.wav'), wav_unreplaced, sr)

    # os.makedirs(os.path.join(path+'22050', prefix,'full'),exist_ok=True)
    # os.makedirs(os.path.join(path+'22050', prefix,'replaced'),exist_ok=True)
    # os.makedirs(os.path.join(path+'22050', prefix,'unreplaced'),exist_ok=True)
    # soundfile.write(os.path.join(path+'22050', prefix,'full', uid+'.wav'), wav_input, 22050)
    # soundfile.write(os.path.join(path+'22050', prefix,'replaced', uid+'.wav'), wav_replaced, 22050)
    # soundfile.write(os.path.join(path+'22050', prefix,'unreplaced', uid+'.wav'), wav_unreplaced, 22050)

def calculate_mcd(gt_path, pred_path, shiftms):
    os.system("python /mnt/home/v_baihe/projects/espnet/utils/mcd_calculate.py --mcep_dim 80 --wavdir {} --gtwavdir {} --f0min 80 --f0max 7600 --shiftms {} --silenced 1".format(pred_path, gt_path, shiftms))


def decode_vctk(path, base1=True, base2=True, gt=True, base4=False,vocoder_base=True):
    dataset = ["p279_401","p314_418", "p265_347", "p277_456", "p266_416", "p304_417", "p228_362","p334_418", "p227_393","p340_418","p288_404","p288_406","p268_402","p308_415","p341_403","p343_395","p243_392","p274_465","p269_396","p265_345","p306_349","p361_417","p259_473","p270_455","p318_418","p302_307","p336_415","p302_308","p237_342","p246_351"]

    fs2_model_path = duration_path_dict['vctk']
    fs2_model, processor = get_fs2_model(fs2_model_path)
    vocoder = load_vocoder('vctk_parallel_wavegan.v1.long')
    model_name="/mnt/home/v_baihe/projects/espnet/egs2/vctk/sedit/exp/conformer"
    prefix = '/mnt/home/v_baihe/projects/espnet/egs2/vctk/sedit/data/eval1/'
    xv_path = '/mnt/home/v_baihe/projects/espnet/aggregate_output/vctk_spk2xvector.pt'
    spk2xvector = torch.load(xv_path)

    decode_conf = {}
    decode_conf.update(use_teacher_forcing=False)
    decode_conf.update(alpha=1.0)
    cfg = decode_conf
    sr = fs2_model.feats_extract.fs
    hop_length = fs2_model.feats_extract.hop_length
    
    span_diff = 0
    for uid in tqdm(dataset):
        spk_id = uid.split("_")[0]
        full_origin_str,wav_path = read_data(uid, prefix)
        wav_org, rate = librosa.load(wav_path, sr)

        token_list = full_origin_str.split()
        split = len(token_list)//3
        new_str = " ".join(token_list[:split]+['[MASK]']+token_list[-split:])

        if base4:
            # baseline4
            if fs2_model.tts.spk_embed_dim is not None:
                spemd = spk2xvector[spk_id]
            else:
                spemd = None
            input_feat,out_feat, span_tobe_replaced, old_span, new_span = decode_for_mcd(model_name+'_no_sega', wav_path, full_origin_str, new_str,fs2_model_path, sid=spemd)
            ours_wav_full = vocoder(out_feat).detach().float().data.cpu().numpy()
            left_index = int(new_span[0]*hop_length)
            right_index = int(new_span[1]*hop_length)
            save_wav_to_path(ours_wav_full, left_index, right_index, path=path, prefix='baseline4', sr=sr, uid=uid)
        else:
            # sedit
            if fs2_model.tts.spk_embed_dim is not None:
                spemd = spk2xvector[spk_id]
            else:
                spemd = None
            input_feat,out_feat, span_tobe_replaced, old_span, new_span = decode_for_mcd(model_name, wav_path, full_origin_str, new_str,fs2_model_path, sid=spemd)
            ours_wav_full = vocoder(out_feat).detach().float().data.cpu().numpy()
            left_index = int(new_span[0]*hop_length)
            right_index = int(new_span[1]*hop_length)
            save_wav_to_path(ours_wav_full, left_index, right_index, path=path, prefix='sedit', sr=sr, uid=uid)
            span_diff += (new_span[1]-old_span[1])
        if vocoder_base:
            left_index = int(old_span[0]*hop_length)
            right_index = int(old_span[1]*hop_length)
            vocoder_wav = vocoder(input_feat).detach().float().data.cpu().numpy()
            save_wav_to_path(vocoder_wav, left_index, right_index, path=path, prefix='vocoder', sr=sr, uid=uid)
        # GT
        if gt:
            left_index = int(old_span[0]*hop_length)
            right_index = int(old_span[1]*hop_length)
            save_wav_to_path(wav_org, left_index, right_index, path=path, prefix='gt', sr=sr, uid=uid)

        # base1
        if base1:
            data = processor("<dummy>", dict(text=full_origin_str, speech=wav_org))
            batch = dict(text=data['text'], speech=data['speech'], spembs=spk2xvector[spk_id])
            batch = to_device(batch, 'cpu')
            output_dict = fs2_model.inference(**batch, **cfg)
            if output_dict.get("feat_gen_denorm") is not None:
                out_feat = output_dict["feat_gen_denorm"]
            else:
                out_feat = output_dict["feat_gen"]
            baseline1_wav = vocoder(out_feat).detach().float().data.cpu().numpy()
            durations = output_dict['duration'].tolist()[:-1]
            left_index = int(sum(durations[:span_tobe_replaced[0]])*hop_length)
            masked_length = int(sum(durations[span_tobe_replaced[0]:span_tobe_replaced[1]])*hop_length)
            right_index = left_index+masked_length
            save_wav_to_path(baseline1_wav, left_index, right_index, path=path, prefix='baseline1', sr=sr, uid=uid)

        # base2
        if base2:
            data = processor("<dummy>", dict(text=" ".join(token_list[split:-split]), speech=wav_org))
            batch = dict(text=data['text'], speech=data['speech'], spembs=spk2xvector[spk_id])
            batch = to_device(batch, 'cpu')
            output_dict = fs2_model.inference(**batch, **cfg)
            durations = output_dict['duration'].tolist()[:-1]
            eos_duration = output_dict['duration'].tolist()[-1]
            if output_dict.get("feat_gen_denorm") is not None:
                out_feat = output_dict["feat_gen_denorm"]
            else:
                out_feat = output_dict["feat_gen"]
            out_feat = torch.cat([input_feat[:old_span[0]], out_feat[:-eos_duration], input_feat[old_span[1]:]])
            baseline2_wav = vocoder(out_feat).detach().float().data.cpu().numpy()
            
            masked_length = int(sum(durations)*hop_length)
            left_index = int(old_span[0]*hop_length)
            right_index = left_index+masked_length
            save_wav_to_path(baseline2_wav, left_index, right_index, path=path, prefix='baseline2', sr=sr, uid=uid)
    print('span_diff:{}'.format(span_diff/len(dataset)))

def decode_ljspeech(path, base1=True, base2=True, gt=True,base4=False,vocoder_base=True, ablation_prefix=None):
    dataset = ["LJ050-0205","LJ050-0130","LJ050-0069","LJ050-0139","LJ050-0218","LJ050-0181","LJ050-0078","LJ050-0145","LJ050-0191","LJ050-0113","LJ050-0255","LJ050-0167","LJ050-0047","LJ050-0038","LJ050-0051","LJ050-0222","LJ050-0072","LJ050-0193","LJ050-0153","LJ050-0057","LJ050-0155","LJ050-0043","LJ050-0202","LJ050-0165","LJ050-0086","LJ050-0107","LJ050-0142","LJ050-0121","LJ050-0033","LJ050-0049"]

    fs2_model_path = duration_path_dict['ljspeech']
    fs2_model, processor = get_fs2_model(fs2_model_path)
    vocoder = load_vocoder('ljspeech_parallel_wavegan.v3')
    model_name="/mnt/home/v_baihe/projects/espnet/egs2/ljspeech/sedit/exp/conformer"
    prefix = '/mnt/home/v_baihe/projects/espnet/egs2/ljspeech/sedit/data/eval1/'
    decode_conf = {}
    decode_conf.update(use_teacher_forcing=False)
    decode_conf.update(alpha=1.0)
    cfg = decode_conf
    sr = fs2_model.feats_extract.fs
    hop_length = fs2_model.feats_extract.hop_length
    span_diff = 0
    for uid in tqdm(dataset):
        spk_id = uid.split("_")[0]
        full_origin_str,wav_path = read_data(uid, prefix)
        wav_org, rate = librosa.load(wav_path, sr)
        token_list = full_origin_str.split()
        split = len(token_list)//3
        new_str = " ".join(token_list[:split]+['[MASK]']+token_list[-split:])
        if ablation_prefix:
            input_feat,out_feat, span_tobe_replaced, old_span, new_span = decode_for_mcd(model_name.replace('conformer',ablation_prefix), wav_path, full_origin_str, new_str,fs2_model_path)
            ours_wav_full = vocoder(out_feat).detach().float().data.cpu().numpy()
            left_index = int(new_span[0]*hop_length)
            right_index = int(new_span[1]*hop_length)
            save_wav_to_path(ours_wav_full, left_index, right_index, path=path, prefix=ablation_prefix, sr=sr, uid=uid)
            continue
        if base4:
            # baseline4
            input_feat,out_feat, span_tobe_replaced, old_span, new_span = decode_for_mcd(model_name+'_no_sega', wav_path, full_origin_str, new_str,fs2_model_path)
            ours_wav_full = vocoder(out_feat).detach().float().data.cpu().numpy()
            left_index = int(new_span[0]*hop_length)
            right_index = int(new_span[1]*hop_length)
            save_wav_to_path(ours_wav_full, left_index, right_index, path=path, prefix='baseline4', sr=sr, uid=uid)
        else:
            # sedit
            input_feat,out_feat, span_tobe_replaced, old_span, new_span = decode_for_mcd(model_name, wav_path, full_origin_str, new_str,fs2_model_path)
            ours_wav_full = vocoder(out_feat).detach().float().data.cpu().numpy()
            left_index = int(new_span[0]*hop_length)
            right_index = int(new_span[1]*hop_length)
            save_wav_to_path(ours_wav_full, left_index, right_index, path=path, prefix='sedit', sr=sr, uid=uid)
            span_diff += (new_span[1]-old_span[1])
        if vocoder_base:
            left_index = int(old_span[0]*hop_length)
            right_index = int(old_span[1]*hop_length)
            vocoder_wav = vocoder(input_feat).detach().float().data.cpu().numpy()
            save_wav_to_path(vocoder_wav, left_index, right_index, path=path, prefix='vocoder', sr=sr, uid=uid)
        
        # GT
        left_index = int(old_span[0]*hop_length)
        right_index = int(old_span[1]*hop_length)
        save_wav_to_path(wav_org, left_index, right_index, path=path, prefix='gt', sr=sr, uid=uid)

        # base1
        if base1:
            data = processor("<dummy>", dict(text=full_origin_str))
            batch = dict(text=data['text'])
            batch = to_device(batch, 'cpu')
            output_dict = fs2_model.inference(**batch, **cfg)
            if output_dict.get("feat_gen_denorm") is not None:
                out_feat = output_dict["feat_gen_denorm"]
            else:
                out_feat = output_dict["feat_gen"]
            baseline1_wav = vocoder(out_feat).detach().float().data.cpu().numpy()
            durations = output_dict['duration'].tolist()[:-1]
            left_index = int(sum(durations[:span_tobe_replaced[0]])*hop_length)
            masked_length = int(sum(durations[span_tobe_replaced[0]:span_tobe_replaced[1]])*hop_length)
            right_index = left_index+masked_length
            save_wav_to_path(baseline1_wav, left_index, right_index, path=path, prefix='baseline1', sr=sr, uid=uid)

        # base2
        if base2:
            data = processor("<dummy>", dict(text=" ".join(token_list[split:-split])))
            batch = dict(text=data['text'])
            batch = to_device(batch, 'cpu')
            output_dict = fs2_model.inference(**batch, **cfg)
            durations = output_dict['duration'].tolist()[:-1]
            eos_duration = output_dict['duration'].tolist()[-1]

            if output_dict.get("feat_gen_denorm") is not None:
                out_feat = output_dict["feat_gen_denorm"]
            else:
                out_feat = output_dict["feat_gen"]
            out_feat = torch.cat([input_feat[:old_span[0]], out_feat[:-eos_duration], input_feat[old_span[1]:]])
            baseline2_wav = vocoder(out_feat).detach().float().data.cpu().numpy()
            
            masked_length = int(sum(durations)*hop_length)
            left_index = int(old_span[0]*hop_length)
            right_index = left_index+masked_length
            save_wav_to_path(baseline2_wav, left_index, right_index, path=path, prefix='baseline2', sr=sr, uid=uid)
    print('span_diff:{}'.format(span_diff/len(dataset)))

if __name__ == "__main__":
    output_path = '/mnt/home/v_baihe/projects/espnet/aggregate_output/mcd/ljspeech'
    #decode_ljspeech(output_path, base1=True, base2=True, gt=False,base4=False,vocoder_base=False)
    decode_ljspeech(output_path, base1=False, base2=False, gt=False,base4=True,vocoder_base=False, ablation_prefix=None)
    decode_ljspeech(output_path, base1=False, base2=False, gt=False,base4=False,vocoder_base=False, ablation_prefix=None)
    shiftms = 256
    for wav_type in ['replaced']:
        for method in ['sedit','baseline4']:
            
            pred_path =  os.path.join(output_path, method,wav_type)
            gt_path = os.path.join(output_path, 'gt',wav_type)
            print("{}_{}:".format(method, wav_type))
            calculate_mcd(gt_path, pred_path, shiftms)

    # output_path = '/mnt/home/v_baihe/projects/espnet/aggregate_output/mcd/vctk'
    # #decode_vctk(output_path, base1=True, base2=True, gt=False,base4=False,vocoder_base=False)
    # shiftms = 300
    # for wav_type in ['replaced']:
    #     # 'baseline1', 'baseline2', 'sedit',
    #     for method in ['baseline4','vocoder']:
    #         pred_path =  os.path.join(output_path, method,wav_type)
    #         gt_path = os.path.join(output_path, 'gt',wav_type)
    #         print("{}_{}:".format(method, wav_type))
    #         calculate_mcd(gt_path, pred_path, shiftms)