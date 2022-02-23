#!/bin/bash

agents_file=$1
model_path=$2
num=$3
save_path=$4

gpu=$(echo "scale=2 ; 0.75 / $num" | bc)

combinations=()


for (( i=0; i<18; i++ )); do
    for (( j=$i+1; j<18; j++ )); do
        combinations+=( $i )
        combinations+=( $j )
    done
done

(
for (( i=0; i<${#combinations[@]}; i+=(2*$num) )); do 
    for (( j=0; j<(2*$num); j+=2 )); do
        gnome-terminal --disable-factory -- /bin/bash -c -i "conda init bash; conda activate rl_soccer; cd /home/amtc/pavan/rl_soccer/models/TD3/paper/2vs2; python eval_2vs2.py $agents_file --model_path $model_path --team_1_idx ${combinations[i+j]} --team_2_idx ${combinations[i+j+1]} --gpu $gpu --save_path $save_path; sleep 10" &
    done
    wait
done
)

(
for (( i=0; i<${#combinations[@]}; i+=(2*$num) )); do 
    for (( j=0; j<(2*$num); j+=2 )); do
        gnome-terminal --disable-factory -- /bin/bash -c -i "conda init bash; conda activate rl_soccer; cd /home/amtc/pavan/rl_soccer/models/TD3/paper/2vs2; python eval_2vs2.py $agents_file --model_path $model_path --team_1_idx ${combinations[i+j]} --team_2_idx ${combinations[i+j+1]} --gpu $gpu --save_path $save_path --order away_home; sleep 10" &
    done
    wait
done
)

