import os
import shutil
import numpy as np
import pandas as pd
import argparse
def get_best_runs(success_threshold, pass_rate, path):
	progress = pd.read_csv(os.path.join(path, "progress.txt"), sep="\t")
	progress = progress[(progress["Success rate"]>success_threshold) & (progress["passAverage"]>pass_rate)]
	the_epochs = progress.Epoch
	model_path = os.path.join(path, "pyt_save")
	dir_list = np.array(os.listdir(model_path))
	selected_models = dir_list[the_epochs]
	for model in selected_models:
		origin = os.path.join(model_path, model)
		destination_folder = os.path.join(model_path, "best_models")
		if not os.path.exists(destination_folder):
			os.mkdir(destination_folder)
		destination_folder = os.path.join(destination_folder, model)
		shutil.move(origin, destination_folder)
	


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Model path.")
    parser.add_argument("--success_thres", type=float, help="minimum threshold we are expecting from the models")
    parser.add_argument("--pass_thres", float, help= "The minimum pass rate we are expecting in the model")
    args = parser.parse_args()
    model_path = args.model_path
    success_thres = args.success_thres
    pass_thres = args.pass_thres
    get_best_runs(model_path, success_thres, pass_thres)
