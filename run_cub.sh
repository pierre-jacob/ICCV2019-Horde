#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset CUB \
                                        --embedding 512 512 512 512 512 \
                                        --ho-dim 8192 8192 8192 8192 \
                                        --use_horde \
                                        --trainable \
                                        --cascaded