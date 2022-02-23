from email.policy import default
import os
import shutil
import numpy as np
import pandas as pd
import glob
import argparse
from video_generator import generate_vid
def get_best_runs(success_threshold, pass_rate, path):
    model_path = os.path.join(path, "pyt_save")
    destination_folder = os.path.join(model_path, "best_models")
    if not os.path.exists(os.path.join(model_path ,"best_models")):
        progress = pd.read_csv(os.path.join(path, "progress.txt"), sep="\t")
        progress = progress[(progress["Success rate"]>success_threshold) & (progress["passAverage"]>pass_rate)]
        the_epochs = progress.Epoch
        
        dir_list = np.array(os.listdir(model_path))
        print(dir_list)
        selected_models = dir_list[the_epochs]
        for model in selected_models:
            origin = os.path.join(model_path, model)
            if not os.path.exists(destination_folder):
                os.mkdir(destination_folder)
            destination_folder = os.path.join(destination_folder, model)
            shutil.move(origin, destination_folder)

def get_best_20_videos(path):
    model_path = os.path.join(path, "pyt_save")
    destination_folder = os.path.join(path, "videos")
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)
    progress = pd.read_csv(os.path.join(path, "progress.txt"), sep="\t")
    progress = progress[(progress["Success rate"]==1) & (progress["passAverage"]>0.4)]
    the_epochs = progress.Epoch
    
    files = os.listdir(model_path)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(model_path, x)))

    dir_list = np.array(files)
    
    selected_models = dir_list[the_epochs]
    print(len(selected_models))
    for model in selected_models:
        generate_vid(destination_folder, os.path.join(model_path, model), model)
    
    
    
            
            
	


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Model path." )
    parser.add_argument("--success_thres", type=float, help="minimum threshold we are expecting from the models", default=0.9)
    parser.add_argument("--pass_thres", type=float, help= "The minimum pass rate we are expecting in the model", default=0)
    args = parser.parse_args()
    model_path = args.model_path
    success_thres = args.success_thres
    pass_thres = args.pass_thres
    get_best_20_videos("D:\\rl_soccer\\results\\resultados\\2vs0_final\\2022-01-07_00-03-58_td3_soccer_goal_pass_join_2vs0_01_07_2022_00_03_58")
