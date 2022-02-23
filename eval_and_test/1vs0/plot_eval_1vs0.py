
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullFormatter
import numpy as np
import os
from pybrium import nash_average
import torch


def highlight_cell(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("num_agents", type=int, help="Num agents")
    parser.add_argument("results_path", type=str, help="Path where results are")
    parser.add_argument("--num_iters", type=int, help="Number of iterations", default=(2 ** 14))

    args = parser.parse_args()
    
    num_agents = args.num_agents
    results_path = args.results_path
    
    payoff_matrix = np.zeros((num_agents, num_agents), dtype=np.float32)
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    
    for j in range(num_agents):
        for i in range(num_agents):
                    
            if j == i:
                continue
            
            elif j < i:
                with open(os.path.join(results_path, '%dvs%d.txt' % (j, i))) as f:
                    infos = f.readlines()
                
                w, t, l = np.array([[int(x.strip()) for x in info.split(" ")] \
                                    for info in infos if info != '\n']).sum(axis=0)
                
                payoff_matrix[j, i] = (w - l) / (w + t + l)
           
            else:
                payoff_matrix[j, i] = -payoff_matrix[i, j]

    nash_equilibrium, nash_rating = nash_average(torch.tensor(payoff_matrix), steps=args.num_iters)
    nash_rating = nash_rating.numpy().squeeze()
    
    idx = np.argsort(-nash_rating)
    
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    # fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6.4, 4.8))

    im = ax.imshow(payoff_matrix[:, idx][idx, :], cmap='RdYlGn', vmin=payoff_matrix.min(), vmax=payoff_matrix.max())

    [highlight_cell(x, y, ax=ax, color="k", linewidth=2) for x in range(num_agents) \
                                                         for y in range(num_agents)]

    # Major ticks
    ax.set_xticks(np.arange(0, num_agents, 1))
    ax.set_yticks(np.arange(0, num_agents, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, num_agents, 1), minor=True)
    ax.set_yticks(np.arange(-.5, num_agents, 1), minor=True)

    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    
    ax.set_ylabel('row agent', fontsize=12)
    # ax.set_xlabel('col agent', fontsize=12)
    ax.set_xticks(np.arange(num_agents))
    ax.set_yticks(np.arange(num_agents))
    ax.set_yticklabels(['%d' % x for x in (idx + 1)])

    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    ax.tick_params(axis=u'both', which=u'minor',length=0)

    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", size=1.2, pad=0.3, sharex=ax)
    # fig.suptitle("Expected goal difference among best agents of 1v0 trials")
    # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # for (i, j), z in np.ndenumerate((met_ord['GenerationalDistance'])):
    #     ax.text(j, i, '{:0.6f}'.format(z), ha='center', va='center',
    #             bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    #plt.show()
    sorted_m_min = -np.sort(-nash_rating) - nash_rating.min()
    ax2.bar(np.arange(num_agents), height=sorted_m_min.tolist(), color="tab:blue", label="DR")
    ax2.set_ylabel('Nash Rating', fontsize=12)
    ax2.set_xlabel('col agent', fontsize=12)
    ax2.set_xticks(np.arange(0, num_agents, 1))
    plt.setp(ax.get_xticklabels(), visible=False) 
    ax2.set_xticklabels(['%d' % x for x in (idx + 1)], rotation=90)
    ax2.tick_params(axis=u'both', which=u'minor',length=0)
    ax2.grid('on')
    ax2.legend(loc="upper right")

    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    print(repr(payoff_matrix))
    plt.tight_layout()
    plt.savefig("payoff_1vs0.pdf", bbox_inches='tight', pad_inches=0)
    
    plt.show()

