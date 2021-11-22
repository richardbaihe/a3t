#!/usr/bin/env python3
from espnet2.tasks.mlm import MLMTask
import os
# os.environ['TORCH_DISTRIBUTED_DEBUG']="DETAIL"

def get_parser():
    parser = MLMTask.get_parser()
    return parser


def main(cmd=None):
    r"""MLM training.

    Example:

        % python asr_train.py asr --print_config --optim adadelta \
                > conf/train_asr.yaml
        % python asr_train.py --config conf/train_asr.yaml
    """
    MLMTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
