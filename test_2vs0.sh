#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate roberto
#for d in */; do
d="C:\Users\rocho\RL_proyect\results\2vs0\Join_pass_env"
for m in $(find $d/pyt_save -name '*.pt'); do
    echo "python /home/amtc/roberto/rl_soccer/test_2vs0.py --model_path $PWD/$d --meta_file $m --gpu 0.15"
    python /home/amtc/roberto/rl_soccer/test_2vs0.py --model_path $PWD/$d --meta_file $m --gpu 0.15
done
    #fi
#done
