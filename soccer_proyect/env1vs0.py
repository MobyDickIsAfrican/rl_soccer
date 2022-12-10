from dm_soccer2gym.wrapper import DmGoalWrapper
import numpy as np


class Env1vs0(DmGoalWrapper):

    def getObservation(self):
        o = super().getObservation()
        o = o.reshape(1, -1)
        return o, None