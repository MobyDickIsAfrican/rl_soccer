from gym.envs.registration import register
import gym
from dm_soccer2gym import wrapper
import hashlib
import dm_soccer2gym


def make(task_name, task_kwargs={}, render_mode_list=None):
    # register environment
    """
    if "reach" in task_name:
        task_name = task_name.replace("reach", "")
        aux = task_name.find("vs")
        if aux == -1:
            raise ValueError("Invalid task")
        team_1 = int(task_name[:aux])
        team_2 = int(task_name[(aux + 2):])
        if not(team_1 == team_2 or (team_1 == 0 and team_2 > 0) or (team_1 > 0 and team_2 == 0)):
            raise ValueError("Invalid task")
        if team_1 < team_2:
            m = team_2
            team_2 = team_1
            team_1 = m
        prehash_id = f"dm_soccer_{team_1}_vs_{team_2}_reach_{task_kwargs.get('rew_type', 'sparse')}"
        h = hashlib.md5(prehash_id.encode())
        gym_id = h.hexdigest()+'-v0'

        # avoid re-registering
        if gym_id not in gym_id_list:
            register(
                id=gym_id,
                entry_point='dm_soccer2gym.wrapper:DmReachWrapper',
                kwargs={'team_1': team_1, 'team_2': team_2, 'task_kwargs': task_kwargs,
                       'render_mode_list': render_mode_list}
            )
    """
    if "goal" in task_name:
        task_name = task_name.replace("goal", "")
        aux = task_name.find("vs")
        if aux == -1:
            raise ValueError("Invalid task")
        team_1 = int(task_name[:aux])
        team_2 = int(task_name[(aux + 2):])
        if not(team_1 == team_2 or (team_1 == 0 and team_2 > 0) or (team_1 > 0 and team_2 == 0)):
            raise ValueError("Invalid task")
        if team_1 < team_2:
            m = team_2
            team_2 = team_1
            team_1 = m
        prehash_id = f"dm_soccer_{team_1}_vs_{team_2}_goal_{task_kwargs.get('rew_type', 'sparse')}" + \
                     f"{task_kwargs.get('observables', 'core')}_obs"
        h = hashlib.md5(prehash_id.encode())
        gym_id = h.hexdigest()+'-v0'

        # avoid re-registering
        if gym_id not in gym_id_list:
            register(
                id=gym_id,
                entry_point='dm_soccer2gym.wrapper:DmGoalWrapper',
                kwargs={'team_1': team_1, 'team_2': team_2, 'task_kwargs': task_kwargs,
                       'render_mode_list': render_mode_list}
            )
    else:
        aux = task_name.find("vs")
        if aux == -1:
            raise ValueError("Invalid task")
        team_1 = int(task_name[:aux])
        team_2 = int(task_name[(aux + 2):])
        if not(team_1 == team_2 or (team_1 == 0 and team_2 > 0) or (team_1 > 0 and team_2 == 0)):
            raise ValueError("Invalid task")
        if team_1 < team_2:
            m = team_2
            team_2 = team_1
            team_1 = m
        prehash_id = f"dm_soccer_{team_1}_vs_{team_2}"
        h = hashlib.md5(prehash_id.encode())
        gym_id = h.hexdigest()+'-v0'

        # avoid re-registering
        if gym_id not in gym_id_list:
            register(
                id=gym_id,
                entry_point='dm_soccer2gym.wrapper:DmSoccerWrapper',
                kwargs={'team_1': team_1, 'team_2': team_2, 'task_kwargs': task_kwargs,
                       'render_mode_list': render_mode_list}
            )
    # add to gym id list
    gym_id_list.append(gym_id)

    # make the Open AI env
    return gym.make(gym_id)


def create_render_mode(name, show=True, return_pixel=True, height=480, width=640, camera_id=0, overlays=(),
             depth=False, scene_option=None):

    render_kwargs = { 'height': height, 'width': width, 'camera_id': camera_id,
                              'overlays': overlays, 'depth': depth, 'scene_option': scene_option}
    render_mode_list[name] = {'show': show, 'return_pixel': return_pixel, 'render_kwargs': render_kwargs}



gym_id_list = []

