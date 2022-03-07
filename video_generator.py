import torch
from env_2vs0_pass import stage_soccerTraining_pass
import time
import numpy as np
import pandas as pd
import cv2
from math import ceil
import copy
import os
import datetime
from tqdm.auto import tqdm
from pathlib import Path
import shutil


exp_path = "D:\\rl_soccer\\results\\resultados\\2vs0_final"

def generate_vid(save_path, model_path, model_name):
	model_name = model_name
	os.chdir(save_path)
	os.mkdir(f"{save_path}\\passing_res_{model_name}_{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M')}")
	os.chdir(f"{save_path}\\passing_res_{model_name}_{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M')}")
	f = open("pass_counting.txt", "a")
	for run in range(10):
		pass_n = 0
		model = torch.load(model_path)
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
		goal= np.any(np.array(env.timestep.reward))
		
		f.write(f"run {run}: pass {pass_n} goal? {goal} \n")


		w, h = images[0].shape[:2]
		out = cv2.VideoWriter(f'{model_name}_{run}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20, (h, w), isColor=True)

		for i in range(len(images)):
			frame = images[i]
			out.write(frame)
		out.release()
	f.close()


if __name__ == "__main__":
	path = "D:\\rl_soccer\\results\\resultados\\T1_2vs0_final\\2022-01-07_00-03-58_td3_soccer_goal_pass_join_2vs0_01_07_2022_00_03_58"
	model_name = "2889999"
	model= "model"+model_name+".pt"
	model_path = os.path.join(path, "pyt_save", model)
	
	save_path = os.path.join(path, "videos")
	generate_vid(save_path, model_path, model)









