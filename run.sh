#!/bin/bash


docker run  --name mujocoSoccer -e NVIDIA_VISIBLE_DEVICES=0 -rm -d -t mujocoimage
docker exec -it mujocoSoccer bash