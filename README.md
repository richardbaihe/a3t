# $\text{A}^3\text{T}$: Alignment-Aware Acoustic and Text Pretraining for Speech Synthesis and Editing


Code for paper [$\text{A}^3\text{T}$: Alignment-Aware Acoustic and Text Pretraining for Speech Synthesis and Editing](https://arxiv.org/abs/2203.09690). Load our checkpoints from [HuggingFace Model Hub](https://huggingface.co/richardbaihe).

## 0. Setup
This repos is forked from [ESPnet](https://github.com/espnet/espnet), please setup your environment according to ESPnet's instruction.

Hint: Once you can run ESPnet's TTS example, there should be no issue to run our code.

## 1. Downloadable Checkpoints:
- LJSpeech Pre-trained Model
- VCTK Pre-trained Model
- LibriTTS Pre-trained Model

## 2. Inference with speech editing and new speaker TTS

## 3. Train you own model

1. ProjectPath: `espnet/egs2/vctk/sedit`

2. Data path:

   I would suggest to link the processed data from my project path `/mnt/home/v_baihe/projects/espnet/egs2/an4/sedit/dump` and `/mnt/home/v_baihe/projects/espnet/egs2/an4/sedit/data`

   Otherwise, run `./data_prepare.sh` under ProjectPath.

   And then run `./local/submit_align_job.sh` under ProjectPath.

   And then run `mlm.sh`'s stage 5 and stage 6 with `run.sh`

3. Training:

   `mlm.sh`'s stage 7 is training, which can be done by modifiying `run.sh`

   