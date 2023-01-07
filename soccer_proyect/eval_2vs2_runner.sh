#!/bin/bash

from_=$1
to_=$2
agents_file=$3
model_path=$4
save_path=$5
order=$6

for ((i=0; i<24; i++)); do
    for ((j=$i+1; j<24; j++)); do 
      combinations+=( $i )
           combinations+=( $j )
    done
done
declare -i N_games=${#combinations[@]}

for ((j=$from_; j<to_; j+=2)); do
	if [[ $((j)) -lt N_games ]]
		then 
			python /soccer_proyect/eval_2vs2.py $agents_file --model_path $model_path  --team_1_idx ${combinations[j]} --team_2_idx ${combinations[j+1]} --save_path $save_path --order $order;
		else
            break
    fi
done