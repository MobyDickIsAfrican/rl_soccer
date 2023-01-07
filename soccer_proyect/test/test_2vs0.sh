#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate roberto
#for d in */; do
d="/home/amtc/roberto/roberto/Proyecto/2vs0/2021-08-26_18-10-15_td3_soccer_goal_pass_join_2vs0_0.1"
for m in $(find $d/pyt_save -name '*.pt'); do
    echo "python /home/amtc/roberto/rl_soccer/test_2vs0.py --model_path $PWD/$d --meta_file $m --gpu 0.15"
    python /home/amtc/roberto/rl_soccer/test_2vs0.py --model_path $PWD/$d --meta_file $m --gpu 0.15
done
    #fi
#done
