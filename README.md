# $\text{A}^3\text{T}$: Alignment-Aware Acoustic and Text Pretraining for Speech Synthesis and Editing 

Code for [paper](https://arxiv.org/abs/2203.09690) $\text{A}^3\text{T}$: [Alignment-Aware Acoustic and Text Pretraining for Speech Synthesis and Editing. Download our checkpoints from [HuggingFace Model Hub](https://huggingface.co/richardbaihe).

:fire: This work has been implemented by the [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/vctk/ernie_sat), where they extend $\text{A}^3\text{T}$ to a multilingual version. 

## Note: This repo works well in my Slurm environment. I haven't but going to test this repo in a new development environment when I have time, to give more details of the setup process. Before that, I am sorry but you need to debug by yourself if errors occurs (most of them should be "path not exists, xxx not installed"). If you just want to learn how did we implement the pre-training model, please take a look at this class [`ESPnetMLMEncAsDecoderModel`](https://github.com/richardbaihe/a3t/blob/aab2d836173371ff3aebcb0fb4ed1480e4c8a5ce/espnet2/tts/sedit/sedit_model.py#L348).

## 0. Setup
This repos is forked from [ESPnet](https://github.com/espnet/espnet), please setup your environment according to ESPnet's instruction.

Hint: Once you can run ESPnet's TTS example, there should be no issue to run our code.

Noting that our forced aligner and phoneme tokenizer are from [HTK](https://htk.eng.cam.ac.uk). 


Our codebase support the training and evaluation for LJSpeech, VCTK, and LibriTTS. Currently, we only take the VCTK as an example in the README.

## 1. Data preprocess
After setup ESPnet environment, go to the folder `egs2/VCTK/sedit`, and run `mlm.sh`'s step 1-5.


## 2. Inference with speech editing or new speaker TTS
We provide a python script for vctk speech editing and prompt-based TTS decoding `bin/sedit_inference.py`, where you can find an example in the `main` function.


## 3. Train you own model
Now, go to this path: `egs2/VCTK/sedit`.
After finished step 1-5 in `mlm.sh`, run `../../../espnet2/bin/align_english.py`
Then, run step 6 of `mlm.sh` to prepare features for training.
Then run step 7 of `mlm.sh` for training.


## To cite our work:
```
@InProceedings{pmlr-v162-bai22d,
  title =   {{A}$^3${T}: Alignment-Aware Acoustic and Text Pretraining for Speech Synthesis and Editing},
  author =       {Bai, He and Zheng, Renjie and Chen, Junkun and Ma, Mingbo and Li, Xintong and Huang, Liang},
  booktitle =   {Proceedings of the 39th International Conference on Machine Learning},
  pages =   {1399--1411},
  year =   {2022},
  volume =   {162},
  series =   {Proceedings of Machine Learning Research},
  month =   {17--23 Jul},
  publisher =    {PMLR},
  pdf =   {https://proceedings.mlr.press/v162/bai22d/bai22d.pdf},
  url =   {https://proceedings.mlr.press/v162/bai22d.html},
}
```
```
@inproceedings{bai2021segatron,
  title={Segatron: Segment-aware transformer for language modeling and understanding},
  author={Bai, He and Shi, Peng and Lin, Jimmy and Xie, Yuqing and Tan, Luchen and Xiong, Kun and Gao, Wen and Li, Ming},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={14},
  pages={12526--12534},
  year={2021}
}
```
