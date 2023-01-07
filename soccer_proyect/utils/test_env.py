import stage_wrappers_env
import numpy 

env = stage_wrappers_env.stage_soccerTraining(2, 0)
action = [1, 2]
env.reset()
print(env.step(action))