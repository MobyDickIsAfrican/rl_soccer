#!/bin/bash


model_path=$1
num=$2
save_path=$3

gpu=$(echo "scale=2 ; 0.75 / $num" | bc)

min=8000000
max=10000000

combinations=()

cd $model_path


for d in */; do
    if [[ $d == *"simple_v2"* ]]; then
        for m in $(find $d/tf1_save -name '*.meta'); do
            mfile="$(basename "$m")";
            fbname="$(echo "$mfile" | cut -d. -f1)";
            if [ ${#fbname} -ge 12 ]; then
                if [ ${fbname:12} -ge $min ] && [ ${fbname:12} -le $max ]; then
                    combinations+=( $d );
                    combinations+=( $m );
                fi
            fi
        done
    fi
done

(
for (( i=0; i<${#combinations[@]}; i+=(2*$num) )); do 
    for (( j=0; j<(2*$num); j+=2 )); do
        gnome-terminal --disable-factory -- /bin/bash -c -i "conda init bash; conda activate rl_soccer; cd $model_path; python /home/amtc/pavan/rl_soccer/models/TD3/paper/1vs1/test_1vs1.py --model_path ${combinations[i+j]} --meta_file ${combinations[i+j+1]} --gpu $gpu --save_path $save_path" &
    done
    wait
done
)


(
for (( i=0; i<${#combinations[@]}; i+=(2*$num) )); do 
    for (( j=0; j<(2*$num); j+=2 )); do
        gnome-terminal --disable-factory -- /bin/bash -c -i "conda init bash; conda activate rl_soccer; cd $model_path; python /home/amtc/pavan/rl_soccer/models/TD3/paper/1vs1/test_1vs1.py --model_path ${combinations[i+j]} --meta_file ${combinations[i+j+1]} --gpu $gpu --save_path $save_path --order away_home" &
    done
    wait
done
)


