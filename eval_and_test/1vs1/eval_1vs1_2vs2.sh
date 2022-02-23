#!/bin/bash

agents_file=$1
model_path=$2
num=$3
save_path=$4

gpu=$(echo "scale=2 ; 0.75 / $num" | bc)

combinations_team=()
combinations_matches=()


for (( i=0; i<11; i++ )); do
    for (( j=$i; j<11; j++ )); do
        combinations_team+=( $i )
        combinations_team+=( $j )
    done
done

for (( i=0; i<(${#combinations_team[@]}*(${#combinations_team[@]}-1)/2); i+=2 )); do
    for (( j=$i+2; j<(${#combinations_team[@]}*(${#combinations_team[@]}-1)/2); j+=2 )); do
        combinations_matches+=( {combinations_team[i]} )
        combinations_matches+=( {combinations_team[i+1]} )
        combinations_matches+=( {combinations_team[j]} )
        combinations_matches+=( {combinations_team[j+1]} )
done


(
for (( i=0; i<${#combinations_matches[@]}; i+=(4*$num) )); do 
    for (( j=0; j<(4*$num); j+=4 )); do
        gnome-terminal --disable-factory -- /bin/bash -c -i "conda init bash; conda activate rl_soccer; cd /home/amtc/pavan/rl_soccer/models/TD3/paper/1vs1; python eval_1vs1_2vs2.py $agents_file --model_path $model_path --agent_1_idx ${combinations_matches[i+j]} --agent_2_idx ${combinations_matches[i+j+1]} --agent_3_idx ${combinations_matches[i+j+2]} --agent_4_idx ${combinations_matches[i+j+3]} --gpu $gpu --save_path $save_path; sleep 10" &
    done
    wait
done
)


(
for (( i=0; i<${#combinations_matches[@]}; i+=(4*$num) )); do 
    for (( j=0; j<(4*$num); j+=4 )); do
        gnome-terminal --disable-factory -- /bin/bash -c -i "conda init bash; conda activate rl_soccer; cd /home/amtc/pavan/rl_soccer/models/TD3/paper/1vs1; python eval_1vs1_2vs2.py $agents_file --model_path $model_path --agent_1_idx ${combinations_matches[i+j]} --agent_2_idx ${combinations_matches[i+j+1]} --agent_3_idx ${combinations_matches[i+j+2]} --agent_4_idx ${combinations_matches[i+j+3]} --gpu $gpu --save_path $save_path --order away_home; sleep 10" &
    done
    wait
done
)


(
for (( i=0; i<${#combinations_matches[@]}; i+=(4*$num) )); do 
    for (( j=0; j<(4*$num); j+=4 )); do
        gnome-terminal --disable-factory -- /bin/bash -c -i "conda init bash; conda activate rl_soccer; cd /home/amtc/pavan/rl_soccer/models/TD3/paper/1vs1; python eval_1vs1_2vs2.py $agents_file --model_path $model_path --agent_1_idx ${combinations_matches[i+j]} --agent_2_idx ${combinations_matches[i+j+1]} --agent_3_idx ${combinations_matches[i+j+3]} --agent_4_idx ${combinations_matches[i+j+2]} --gpu $gpu --save_path $save_path; sleep 10" &
    done
    wait
done
)


(
for (( i=0; i<${#combinations_matches[@]}; i+=(4*$num) )); do 
    for (( j=0; j<(4*$num); j+=4 )); do
        gnome-terminal --disable-factory -- /bin/bash -c -i "conda init bash; conda activate rl_soccer; cd /home/amtc/pavan/rl_soccer/models/TD3/paper/1vs1; python eval_1vs1_2vs2.py $agents_file --model_path $model_path --agent_1_idx ${combinations_matches[i+j]} --agent_2_idx ${combinations_matches[i+j+1]} --agent_3_idx ${combinations_matches[i+j+3]} --agent_4_idx ${combinations_matches[i+j+2]} --gpu $gpu --save_path $save_path --order away_home; sleep 10" &
    done
    wait
done
)

