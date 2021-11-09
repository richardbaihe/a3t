# Speech Editing

1. ProjectPath: `espnet/egs2/an4/sedit`

2. Data path:

   I would suggest to link the processed data from my project path `/mnt/home/v_baihe/projects/espnet/egs2/an4/sedit/dump` and `/mnt/home/v_baihe/projects/espnet/egs2/an4/sedit/data`

   Otherwise, run `./data_prepare.sh` under ProjectPath.

   And then run `./local/submit_align_job.sh` under ProjectPath.

   And then run `mlm.sh`'s stage 5 and stage 6 with `run.sh`

3. Training:

   `mlm.sh`'s stage 7 is training, which can be done by modifiying `run.sh`

   