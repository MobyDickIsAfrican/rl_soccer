#!/bin/bash

save_path=$1
model_path=$2

declare -i window=0
declare -i pane=0
session="2vs2_games_$order"
tmux new-session -d -s $session 
for ((i=0; i<3; i++)); do
    if [[ $pane -ne 0 ]] 
      then  
         tmux splitw -t $session:$window -d 
         tmux select-layout -t $session:$window tiled
         tmux select-pane -t $pane
      fi
      tmux send-keys -t $session:$window "python exp_2vs2Dense.py --save_path $1 --model_path $2" Enter
      pane+=1
done