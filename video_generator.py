import torch
from env_2vs0_pass import stage_soccerTraining_pass
import time
import numpy as np
import cv2
from math import ceil
import copy
import os
import datetime

# FIXME HACER 500 VIDEOS CON ARCHIVO DE TEXTO.
path = "C:\\Users\\rocho\\RL_proyect\\results\\resultados\\exp7\\pyt_save\\model2999999.pt"
os.mkdir(f"C:\\Users\\rocho\\RL_proyect\\passing_res_28_12_21_{datetime.datetime.now().strftime('%H_%M_%S')}")
os.chdir(f"C:\\Users\\rocho\\RL_proyect\\passing_res_28_12_21_{datetime.datetime.now().strftime('%H_%M_%S')}")
f = open("pass_counting.txt", "a")
for run in range(20):
	pass_n = 0
	model = torch.load(path)
	max_ep_len = ceil(30 / 0.1)
	images = []
	env  = stage_soccerTraining_pass(team_1=2, team_2=0,task_kwargs={ "time_limit": 30, "disable_jump": True, 
							"dist_thresh": 0.03, 'control_timestep': 0.1, "random_seed":30, "observables": "all"}, render_mode_list=[30])
	o, d, ep_ret, ep_len = env.reset(), False, np.array([0]*(2), dtype='float32'), 0

	while not(d or (ep_len == max_ep_len)):
		images.append(copy.deepcopy(env.render()[0]))
		time.sleep(1/60)
		received_pass = [obs["stats_i_received_pass"] for obs in env.timestep.observation]
		if np.any(received_pass):
			pass_n +=1
		# Take deterministic actions at test time (noise_scale=0)
		o = o[np.newaxis, :]
		actions = model.act(torch.as_tensor(o[:2], dtype=torch.float32).cuda()).cpu().detach().numpy()
		o, r, d, _ = env.step([actions[0,i, :] for i in range(2)])
		ep_ret += r
		ep_len += 1
	
	f.write(f"run {run}: {pass_n} \n")


	w, h = images[0].shape[:2]
	out = cv2.VideoWriter(f'pass_fail_run_{run}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 5, (h, w), isColor=True)

	for i in range(len(images)):
		frame = images[i]
		out.write(frame)
	out.release()
f.close()



