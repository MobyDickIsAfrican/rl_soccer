docker run --name mujocoSoccer -v /home/roberto.cholaky/cachefs:/results -v /home/roberto.cholaky/rl_soccer/models:/models  --gpus device=1 -t -d mujocosoccerimage
