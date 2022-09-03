#!/bin/bash

# variables defined in the calling of the script
agents_file=$1
model_path=$2
save_path=$3
order=$4


# variables defined in the calling of the script

run_games (){
   declare -i window=0
   declare -i pane=0
   session="2vs2_games_$order"
   tmux new-session -d -s $session 
   # save combinations in an array:
   combinations=()
   ## iterate through all the agents: 
   # generating the 
   for ((i=0; i<24; i++)); do
    for ((j=$i+1; j<24; j++)); do 
      combinations+=( $i )
           combinations+=( $j )
       done
   done

   declare -i N_games=${#combinations[@]}
   declare -i num=$(( $N_games/12 ))
   for (( i=0; i<N_games; i+=($num) )); do 
      j=$(( $i + $num )) 
      if [[ $pane -ne 0 ]] 
      then  
         tmux splitw -t $session:$window -d 
         tmux select-layout -t $session:$window tiled
         tmux select-pane -t $pane
      fi
      tmux send-keys -t $session:$window "bash eval_2vs2_runner.sh $i $j $agents_file $model_path $save_path $order" Enter
      pane+=1
   done
}

run_games