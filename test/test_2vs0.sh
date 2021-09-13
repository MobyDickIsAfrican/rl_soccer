#!/bin/bash

conda init bash
conda activate rl_soccer
#for d in */; do
    d="2021-08-26_18-10-15_td3_soccer_goal_pass_join_2vs0_0.1"
    #if [[ $d == *"simple_v2"* ]]; then
        for m in $(find $d/pyt_save -name '*.meta'); do
            echo "python /home/amtc/roberto/rl_soccer/test/test_2vs0.py --model_path $PWD/$d --meta_file $m --gpu 0.15"
            python /home/amtc/pavan/rl_soccer/test/test_2vs0.py --model_path $PWD/$d --meta_file $m --gpu 0.15
        done
    #fi
#done
