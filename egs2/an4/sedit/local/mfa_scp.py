import os
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import tgt

# 'train-clean-460',"test-clean",
for file_name in ["dev-clean"]:
    count=0
    dump_dir = 'dump/raw'
    tgt_dir = '/mnt/home/v_baihe/data/LibriTTS/mfa/TextGrid/{}'.format(file_name)
    wav_scp = open(os.path.join(dump_dir, file_name,'wav.scp'),'r')
    text_scp = open(os.path.join(dump_dir, file_name,'text'),'r')
    with open(os.path.join(dump_dir, file_name,'mfa_start'),'w') as f_start, open(os.path.join(dump_dir, file_name,'mfa_end'),'w') as f_end, open(os.path.join(dump_dir, file_name, "no_algin_files"),'w') as f_no, open(os.path.join(dump_dir, file_name, "mfa_text"),'w') as f_text_new, open(os.path.join(dump_dir, file_name, "mfa_wav.scp"),'w') as f_wav_new:
        for line,line_text in tqdm(zip(wav_scp.readlines(),text_scp.readlines())):
            sps = line.rstrip().split(maxsplit=1)
            k, wav_path = sps
            speaker = k.split('_')[0]
            chapter = k.split('_')[1]
            tgt_path = os.path.join(tgt_dir, speaker,chapter, k+".TextGrid")
            try:
                textgrid = tgt.io.read_textgrid(tgt_path)
            except:
                f_no.write(k+'\n')
                count+=1
                continue
            f_wav_new.write(line)
            f_text_new.write(line_text)
            tier = textgrid.get_tier_by_name("phones")
            start_list = [k]
            end_list = [k]
            for t in tier._objects:
                start_list.append(str(t.start_time))
                end_list.append(str(t.end_time))
            f_start.write(" ".join(start_list)+'\n')
            f_end.write(" ".join(end_list)+'\n')
    print(count)
