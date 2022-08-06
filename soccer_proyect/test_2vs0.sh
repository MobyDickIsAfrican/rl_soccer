#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate roberto_soccer
for d in $(find "/home/amtc/roberto/Proyecto/2vs0_new/" -mindepth 1 -maxdepth 1 -type d ); 
do
    echo $d
    python /home/amtc/roberto/rl_soccer/get_best.py --model_path $d --success_thres 0.9 --pass_thres 0
    m_path=$d"/pyt_save/"
    for m in $(find $m_path -name '*.pt'); do
        echo "python /home/amtc/roberto/rl_soccer/test_2vs0.py --model_path $d --meta_file $m --gpu 0.15"
        python /home/amtc/roberto/rl_soccer/test_2vs0.py --model_path $d --meta_file $m --gpu 1
    done
done
    #fi
#done
