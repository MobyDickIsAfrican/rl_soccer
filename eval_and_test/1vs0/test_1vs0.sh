#!/bin/bash

conda init bash
conda activate rl_soccer
#for d in */; do
    d="2020-09-12_23-41-47_td3_soccer_goal_1vs0_simple_v2_0.05"
    #if [[ $d == *"simple_v2"* ]]; then
        for m in $(find $d/tf1_save -name '*.meta'); do
            echo "python /home/amtc/pavan/rl_soccer/models/TD3/paper/1vs0/test_1vs0.py --model_path $PWD/$d --meta_file $m --gpu 0.15"
            python /home/amtc/pavan/rl_soccer/models/TD3/paper/1vs0/test_1vs0.py --model_path $PWD/$d --meta_file $m --gpu 0.15
        done
    #fi
#done
