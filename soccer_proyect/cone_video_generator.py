from env_2vs0_pass import stage_soccerTraining_pass
from plotter import * 
import os
import torch
from math import ceil
import copy

def generate_cone_2vs2_vid():
    for run in range(10):
        model_home1 = torch.load("model2939999.pt")
        model_home2 = torch.load("model2939999.pt")
        fig, ax = plt.subplots(1,2, squeeze=False)
        max_ep_len = ceil(30 / 0.1)
        images = []
        env  = stage_soccerTraining_pass(team_1=2, team_2=0,task_kwargs={ "time_limit": 30, "disable_jump": True, 
                                "dist_thresh": 0.03, 'control_timestep': 0.1, "random_seed":30, "observables": "all"}, render_mode_list=[30])
        obs, d, ep_ret, ep_len = torch.from_numpy(env.reset()[None, ...]).cuda().float(), False, np.array([0]*(4), dtype='float32'), 0
        obs_original = env.timestep.observation[0]
        while not(d or (ep_len == max_ep_len)):
            intercept_areas = env.get_Area(obs_original)
            ax[0,0].cla()
            ax[0, 1].cla()
            teams=["home"]*2
            ax[0,0].set_xbound(-env.pitch_size[0], env.pitch_size[0])
            ax[0,0].set_ybound(-env.pitch_size[1], env.pitch_size[1])
            own_orientation = obs_original["sensors_gyro"][0, -1]
            # we get the position of the teammate
            teammate_pos = [(-obs_original[f'teammate_0_ego_position'][0, :2]).tolist()]
            # get the orientation of teammates players:
            teammate_orientation = [env.get_angle(obs_original[f'teammate_{i}_ego_orientation']) for i in range(env.team_1-1)]
            o_area = intercept_areas[f'intercept_area_teammate_0']
            o_centroid = intercept_areas["intercept_centroid_teammate_0"]
            o_area_pass = intercept_areas[f"intercept_area_pass_teammate_{0}"]
            generate_teams([[0,0]]+teammate_pos, [own_orientation]+teammate_orientation, teams, ax[0,0])
            generate_text([o_area, o_area_pass], ax[0, 0]) 
            generate_centroid_text([o_centroid, intercept_areas[f"intercept_centroid_pass_teammate_{0}"]], ax[0, 0])
            

            images.append(copy.deepcopy(env.render()[0]))
            ax[0, 1].imshow(env.render()[0])
            plt.waitforbuttonpress()
            a = [model_home1.act(obs[:,0, :-12].unsqueeze(1)).cpu().detach(), model_home1.act(obs[:,1, :-12].unsqueeze(1)).cpu().detach()]
            obs, r, d, _ = env.step(a)
            obs = torch.from_numpy(obs[None, ...]).cuda().float()
            obs_original = env.timestep.observation[0]
            ep_len += 1


if __name__ == "__main__":
    generate_cone_2vs2_vid()