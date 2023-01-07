import numpy as np
import os
path = "D:\\rl_soccer\\results\\resultados\\ResultadosEtapa1"
nash_rating = np.load(os.path.join(path, "nash_averaging.npy"))
print(np.max(nash_rating))
print(np.argmax(nash_rating))

