#!/usr/bin/env python

""" Usage:
      align_english.py wavfile trsfile outwordfile outphonefile
"""

import os
import sys
from tqdm import tqdm
import multiprocessing as mp
from espnet2.text.phoneme_tokenizer import G2p_en

g2p_tokenzier=G2p_en(no_space=True)

PHONEME = '{PATH2thisproject}/a3t/tools/english2phoneme/phoneme'
MODEL_DIR = '{PATH2thisproject}/a3t/tools/alignment/aligner/english'
HVITE = '{PATH2thisproject}/a3t/tools/htk/HTKTools/HVite'
HCOPY = '{PATH2thisproject}/a3t/tools/htk/HTKTools/HCopy'

def prep_txt(line, tmpbase, dictfile):
 
    words = []

    line = line.strip()
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
    with open(dictfile, 'r') as fid:
        for line in fid:
            ds.add(line.split()[0])

    unk_words = set([])
    with open(tmpbase + '.txt', 'w') as fwid:
        for wrd in words:
            if (wrd.upper() not in ds):
                unk_words.add(wrd.upper())
            fwid.write(wrd + ' ')
        fwid.write('\n')

    #generate pronounciations for unknows words using 'letter to sound'
    with open(tmpbase + '_unk.words', 'w') as fwid:
        for unk in unk_words:
            fwid.write(unk + '\n')
    try:
        os.system(PHONEME + ' ' + tmpbase + '_unk.words' + ' ' + tmpbase + '_unk.phons')
    except:
        print('english2phoneme error!')
        sys.exit(1)

    #add unknown words to the standard dictionary, generate a tmp dictionary for alignment 
    fw = open(tmpbase + '.dict', 'w')
    with open(dictfile, 'r') as fid:
        for line in fid:
            fw.write(line)
    f = open(tmpbase + '_unk.words', 'r')
    lines1 = f.readlines()
    f.close()
    f = open(tmpbase + '_unk.phons', 'r')
    lines2 = f.readlines()
    f.close()
    for i in range(len(lines1)):
        wrd = lines1[i].replace('\n', '')
        phons = lines2[i].replace('\n', '').replace(' ', '')
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

        fw.write(wrd + ' ')
        for s in seq:
            fw.write(' ' + s)
        fw.write('\n')
    fw.close()

def prep_mlf(txt, tmpbase):

    with open(tmpbase + '.mlf', 'w') as fwid:
        fwid.write('#!MLF!#\n')
        fwid.write('"' + tmpbase + '.lab"\n')
        fwid.write('sp\n')
        wrds = txt.split()
        for wrd in wrds:
            fwid.write(wrd.upper() + '\n')
            fwid.write('sp\n')
        fwid.write('.\n')

def gen_res(tmpbase, outfile1, outfile2):
    with open(tmpbase + '.txt', 'r') as fid:
        words = fid.readline().strip().split()
    words = txt.strip().split()
    words.reverse()

    with open(tmpbase + '.aligned', 'r') as fid:
        lines = fid.readlines()
    i = 2
    times1 = []
    times2 = []
    while (i < len(lines)):
        if (len(lines[i].split()) >= 4) and (lines[i].split()[0] != lines[i].split()[1]):
            phn = lines[i].split()[2]
            pst = (int(lines[i].split()[0])/1000+125)/10000
            pen = (int(lines[i].split()[1])/1000+125)/10000
            times2.append([phn, pst, pen])
        if (len(lines[i].split()) == 5):
            if (lines[i].split()[0] != lines[i].split()[1]):
                wrd = lines[i].split()[-1].strip()
                st = (int(lines[i].split()[0])/1000+125)/10000
                j = i + 1
                while (lines[j] != '.\n') and (len(lines[j].split()) != 5):
                    j += 1
                en = (int(lines[j-1].split()[1])/1000+125)/10000
                times1.append([wrd, st, en])
        i += 1

    with open(outfile1, 'w') as fwid:
        for item in times1:
            if (item[0] == 'sp'):
                fwid.write(str(item[1]) + ' ' + str(item[2]) + ' SIL\n')
            else:
                wrd = words.pop()
                fwid.write(str(item[1]) + ' ' + str(item[2]) + ' ' + wrd + '\n')
    if words:
        print('not matched::' + alignfile)
        sys.exit(1)

    with open(outfile2, 'w') as fwid:
        for item in times2:
            fwid.write(str(item[1]) + ' ' + str(item[2]) + ' ' + item[0] + '\n')

def alignment(wav_path, text_string):
    tmpbase = '/tmp/' + os.environ['USER'] + '_' + str(os.getpid())

    #prepare wav and trs files
    try:
        os.system('sox ' + wav_path + ' -r 16000 ' + tmpbase + '.wav remix -')
    except:
        print('sox error!')
        return None
    
    #prepare clean_transcript file
    try:
        prep_txt(text_string, tmpbase, MODEL_DIR + '/dict')
    except:
        print('prep_txt error!')
        return None

    #prepare mlf file
    try:
        with open(tmpbase + '.txt', 'r') as fid:
            txt = fid.readline()
        prep_mlf(txt, tmpbase)
    except:
        print('prep_mlf error!')
        return None

    #prepare scp
    try:
        os.system(HCOPY + ' -C ' + MODEL_DIR + '/16000/config ' + tmpbase + '.wav' + ' ' + tmpbase + '.plp')
    except:
        print('HCopy error!')
        return None

    #run alignment
    try:
        os.system(HVITE + ' -a -m -t 10000.0 10000.0 100000.0 -I ' + tmpbase + '.mlf -H ' + MODEL_DIR + '/16000/macros -H ' + MODEL_DIR + '/16000/hmmdefs -i ' + tmpbase +  '.aligned '  + tmpbase + '.dict ' + MODEL_DIR + '/monophones ' + tmpbase + '.plp 2>&1 > /dev/null') 
    except:
        print('HVite error!')
        return None

    with open(tmpbase + '.txt', 'r') as fid:
        words = fid.readline().strip().split()
    words = txt.strip().split()
    words.reverse()

    with open(tmpbase + '.aligned', 'r') as fid:
        lines = fid.readlines()
    i = 2
    times2 = []
    word2phns = {}
    current_word = ''
    index = 0
    while (i < len(lines)):
        splited_line = lines[i].strip().split()
        if (len(splited_line) >= 4) and (splited_line[0] != splited_line[1]):
            phn = splited_line[2]
            pst = (int(splited_line[0])/1000+125)/10000
            pen = (int(splited_line[1])/1000+125)/10000
            times2.append([phn, pst, pen])
            # splited_line[-1]!='sp'
            if len(splited_line)==5:
                current_word = str(index)+'_'+splited_line[-1]
                word2phns[current_word] = phn
                index+=1
            elif len(splited_line)==4:
                word2phns[current_word] += ' '+phn 
        i+=1
    return times2,word2phns

def worker(example):

    line_wav, line_text = example

    k, wavfile = line_wav.rstrip().split(maxsplit=1)
    _, trsfile = line_text.rstrip().split(maxsplit=1)

    times2,_ = alignment(wavfile, trsfile)
    text = [k]
    start_list = [k]
    end_list = [k]
    for item in times2:
        start_list.append(str(item[1]))
        end_list.append(str(item[2]))
        text.append(item[0])
    res = (line_wav, " ".join(text), " ".join(start_list), " ".join(end_list))
    # q.put(res)
    return res

if __name__ == "__main__":
    pool = mp.Pool(mp.cpu_count())
    for file_name in ["tr_no_dev", "dev", "eval1"]:
        dump_dir = 'dump/raw'
        wav_scp = open(os.path.join(dump_dir, file_name,'wav.scp'),'r')
        text_scp = open(os.path.join(dump_dir, file_name,'text'),'r')

        f_start = open(os.path.join(dump_dir, file_name,'mfa_start'),'w') 
        f_end = open(os.path.join(dump_dir, file_name,'mfa_end'),'w')
        f_text_new = open(os.path.join(dump_dir, file_name, "mfa_text"),'w') 
        f_wav_new = open(os.path.join(dump_dir, file_name, "mfa_wav.scp"),'w')
        examples = []
        CHUNCK_SIZE = 1000
        p = 0
        for line_wave, line_text in zip(wav_scp, text_scp):
            examples.append((line_wave, line_text))
            if len(examples) == CHUNCK_SIZE:
                results = pool.map(worker, examples)
                p += len(examples)
                for m in results:
                    if m is not None and len(m[1])>0:
                        f_wav_new.write(m[0])
                        f_wav_new.flush()
                        f_text_new.write(m[1] + '\n')
                        f_text_new.flush()
                        f_start.write(m[2] + '\n')
                        f_start.flush()
                        f_end.write(m[3] + '\n')
                        f_end.flush()
                print(p)
                examples = []
        if len(examples) > 0:
            results = pool.map(worker, examples)
            for m in results:
                if m is not None and len(m[1])>0:
                    f_wav_new.write(m[0])
                    f_wav_new.flush()
                    f_text_new.write(m[1] + '\n')
                    f_text_new.flush()
                    f_start.write(m[2] + '\n')
                    f_start.flush()
                    f_end.write(m[3] + '\n')
                    f_end.flush()


    for file_name in ["tr_no_dev", "dev", "eval1"]:
            dump_dir = 'dump/raw'
            text_name = os.path.join(dump_dir, file_name, "mfa_text")
            wav_name = os.path.join(dump_dir, file_name, "mfa_wav.scp")
            start_name = os.path.join(dump_dir, file_name,'mfa_start')
            end_name = os.path.join(dump_dir, file_name,'mfa_end')
            f_text = open(text_name,'r') 
            f_wav = open(wav_name,'r')
            f_start = open(start_name,'r') 
            f_end = open(end_name,'r')

            fn_text = open(text_name+'n','w') 
            fn_wav = open(wav_name+'n','w')
            fn_start = open(start_name+'n','w') 
            fn_end = open(end_name+'n','w')

            for line_text, line_wav,line_start, line_end in zip(f_text, f_wav,f_start,f_end):
                if ' ' in line_text:
                    fn_text.write(line_text)
                    fn_wav.write(line_wav)
                    fn_start.write(line_start)
                    fn_end.write(line_end)
            os.system('mv {} {}'.format(text_name+'n', text_name))
            os.system('mv {} {}'.format(wav_name+'n', wav_name))
            os.system('mv {} {}'.format(start_name+'n', start_name))
            os.system('mv {} {}'.format(end_name+'n', end_name))