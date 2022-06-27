import os
from espnet2.bin.sedit_inference import *
from espnet2.torch_utils.device_funcs import to_device
import kaldiio
import soundfile
from tqdm import tqdm

def save_wav_to_path(wav_input, left_index,right_index, path, prefix, sr,uid):
    wav_replaced = wav_input[left_index:right_index]
    wav_unreplaced = np.concatenate([wav_input[:left_index], wav_input[right_index:]])
    os.makedirs(os.path.join(path, prefix,'full'),exist_ok=True)
    os.makedirs(os.path.join(path, prefix,'replaced'),exist_ok=True)
    os.makedirs(os.path.join(path, prefix,'unreplaced'),exist_ok=True)
    soundfile.write(os.path.join(path, prefix,'full', uid+'.wav'), wav_input, sr)
    soundfile.write(os.path.join(path, prefix,'replaced', uid+'.wav'), wav_replaced, sr)
    soundfile.write(os.path.join(path, prefix,'unreplaced', uid+'.wav'), wav_unreplaced, sr)

    os.makedirs(os.path.join(path+'22050', prefix,'full'),exist_ok=True)
    os.makedirs(os.path.join(path+'22050', prefix,'replaced'),exist_ok=True)
    os.makedirs(os.path.join(path+'22050', prefix,'unreplaced'),exist_ok=True)
    soundfile.write(os.path.join(path+'22050', prefix,'full', uid+'.wav'), wav_input, 22050)
    soundfile.write(os.path.join(path+'22050', prefix,'replaced', uid+'.wav'), wav_replaced, 22050)
    soundfile.write(os.path.join(path+'22050', prefix,'unreplaced', uid+'.wav'), wav_unreplaced, 22050)


set_all_random_seed(9999)
dataset = ["p279_401","p314_418", "p265_347", "p277_456", "p266_416", "p304_417", "p228_362","p334_418", "p227_393","p340_418","p288_404","p288_406","p268_402","p308_415","p341_403","p343_395","p243_392","p274_465","p269_396","p265_345","p306_349","p361_417","p259_473","p270_455","p318_418","p302_307","p336_415","p302_308","p237_342","p246_351"]

fs2_model_path = duration_path_dict['vctk']
fs2_model, processor = get_fs2_model(fs2_model_path)
vocoder = load_vocoder('vctk_parallel_wavegan.v1.long')
model_name="{PATH2thisproject}/a3t/egs2/vctk/sedit/exp/conformer"
prefix = '{PATH2thisproject}/a3t/egs2/vctk/sedit/data/eval1/'
xv_path = '{PATH2thisproject}/a3t/aggregate_output/vctk_spk2xvector.pt'
spk2xvector = torch.load(xv_path)

decode_conf = {}
decode_conf.update(use_teacher_forcing=False)
decode_conf.update(alpha=1.0)
cfg = decode_conf
sr = fs2_model.feats_extract.fs
window_length = fs2_model.feats_extract.hop_length
path = '{PATH2thisproject}/a3t/aggregate_output/mcd/vctk'
for uid in tqdm(dataset):
    spk_id = uid.split("_")[0]
    full_origin_str,wav_path = read_data(uid, prefix)
    wav_org, rate = librosa.load(wav_path, sr)
    token_list = full_origin_str.split()
    split = len(token_list)//3
    new_str = " ".join(token_list[:split]+['[MASK]']+token_list[-split:])
    input_feat, span_tobe_replaced, old_span, new_span = get_input_feat_and_masked_span(model_name, wav_path, full_origin_str, new_str,fs2_model_path)
    # GT
    left_index = int(old_span[0]*window_length)
    right_index = int(old_span[1]*window_length)
    save_wav_to_path(wav_org, left_index, right_index, path=path, prefix='gt', sr=sr, uid=uid)

    # ours  
    data_dict = test_vctk(uid,vocoder,prefix,model_name,new_str=new_str)
    ours_wav_full = data_dict['prediction']
    left_index = int(new_span[0]*window_length)
    right_index = int(new_span[1]*window_length)
    save_wav_to_path(ours_wav_full, left_index, right_index, path=path, prefix='sedit', sr=sr, uid=uid)

    # base1
    data = processor("<dummy>", dict(text=full_origin_str, speech=wav_org))
    batch = dict(text=data['text'], speech=data['speech'], spembs=spk2xvector[spk_id])
    batch = to_device(batch, 'cpu')
    output_dict = fs2_model.inference(**batch, **cfg)
    out_feat = output_dict["feat_gen"]
    baseline1_wav = vocoder(out_feat).detach().float().data.cpu().numpy()
    durations = output_dict['duration'].tolist()[:-1]
    left_index = int(sum(durations[:span_tobe_replaced[0]])*window_length)
    masked_length = int(sum(durations[span_tobe_replaced[0]:span_tobe_replaced[1]])*window_length)
    right_index = left_index+masked_length
    save_wav_to_path(baseline1_wav, left_index, right_index, path=path, prefix='baseline1', sr=sr, uid=uid)

    # base2
    data = processor("<dummy>", dict(text=" ".join(token_list[split:-split]), speech=wav_org))
    batch = dict(text=data['text'], speech=data['speech'], spembs=spk2xvector[spk_id])
    batch = to_device(batch, 'cpu')
    output_dict = fs2_model.inference(**batch, **cfg)
    durations = output_dict['duration'].tolist()[:-1]
    eos_duration = output_dict['duration'].tolist()[-1]

    out_feat = torch.cat([input_feat[:old_span[0]], output_dict["feat_gen"][:-eos_duration], input_feat[old_span[1]:]])
    baseline2_wav = vocoder(out_feat).detach().float().data.cpu().numpy()
    
    masked_length = int(sum(durations)*window_length)
    left_index = int(old_span[0]*window_length)
    right_index = left_index+masked_length
    save_wav_to_path(baseline2_wav, left_index, right_index, path=path, prefix='baseline2', sr=sr, uid=uid)
