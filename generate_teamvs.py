import os
import numpy as np 
import matplotlib.pyplot as plt
from pybrium import nash_average
import torch

path = "D:\\rl_soccer\\results\\resultados\\ResultadosEtapa1\\second_evaluation"
path2 = "D:\\rl_soccer\\results\\resultados\\ResultadosEtapa1"
files = os.listdir(path)

results = np.zeros((48, 48))
print(results)
for file_name in files:
    filename = file_name.replace(".txt", "").split("vs")
    total_wins = 0
    total_games = 0
    with open(os.path.join(path, file_name)) as f:
        for line in f: 
            match_results = list(map(int, line.replace("\n", "").split(" ")))
            total_games += sum(match_results)
            total_wins += (match_results[0] - match_results[-1])/total_games 
    results[int(filename[0]), int(filename[1])] = total_wins
    results[int(filename[1]), int(filename[0])] = -total_wins

minimum_value = np.min(results)
winrates = torch.tensor(results-minimum_value, dtype=torch.float32)
nash_equilibrium, nash_rating = nash_average(winrates, steps=(2 ** 18))



plt.matshow(results)
plt.xlabel("Column Teams")
plt.ylabel("Row Teams")
plt.title("Payoff matrix")
plt.colorbar()
plt.show()

val = np.squeeze(nash_rating.numpy(), -1)
fig, ax = plt.subplots()
ax.bar(range(48), val.tolist())
ax.set_xlabel("Teams")
ax.set_ylabel("Nash Rating")
ax.set_title("Nash Rating per teams")
ax.plot(range(48), [max(val)]*48)
plt.show()
np.save(os.path.join(path2, "nash_averaging.npy"), nash_rating.numpy())
