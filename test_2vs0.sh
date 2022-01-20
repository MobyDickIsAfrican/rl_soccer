#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate roberto388
for d in $(find "/media/amtc/ab170e70-f9d7-4d20-a751-0c11c1ac74881/roberto/resultados_2vs0/" -mindepth 1 -maxdepth 1 -type d ); 
do
    echo $d
    python /home/amtc/roberto/rl_soccer/get_best.py --model_path $d --success_thres 0.9 --pass_thres 0
    for m in $(find $d/pyt_save/best_models -name '*.pt'); do
        echo "python /home/amtc/roberto/rl_soccer/test_2vs0.py --model_path $d --meta_file $m --gpu 0.15"
        python /home/amtc/roberto/rl_soccer/test_2vs0.py --model_path $d --meta_file $m --gpu 1
    done
done
    #fi
#done
