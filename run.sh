docker run --name mujocoSoccer -v /home/roberto.cholaky/cachefs:/results -v /home/roberto.cholaky/rl_soccer/models:/models  --cpus 20 --gpus device=1 -t -d mujocosoccerimage
