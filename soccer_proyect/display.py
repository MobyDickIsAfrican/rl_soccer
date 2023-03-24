import functools
from absl import app
from absl import flags
from dm_control import viewer
from dm_control.locomotion import soccer
from dm_soccer2gym.wrapper import DmGoalWrapper
import numpy as np
import PIL
import matplotlib.pyplot as plt

random_state = np.random.RandomState(42)
env = soccer.load(
    team_size=3,
    time_limit=45.,
    random_state=random_state,
    disable_walker_contacts=False,
    walker_type=soccer.WalkerType.BOXHEAD,
)
env.reset()

pixels = []
#cameras = random_state.choice(env.physics.model.ncam, 6, replace=False)
cameras = [2]
for camera_id in cameras:
  pixels.append(env.physics.render(camera_id=camera_id, width=240))
plt.imshow(pixels[0])
pass