
import csv
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


SHOW_DOMINANT = True
SHOW_TOP = False
SCALE = True
filter_run = ''


if __name__ == "__main__":

    info = []

    with open('stats.csv', 'r') as csvfile:

        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:
            info.append(list(row))

    info = np.array(info)
    print(info.shape)

    run_name, it, suc_rate, mean_len, std_len, mean_vel, std_vel = np.split(info, 7, axis=-1)
    run_name = info[:, 0]
        
    dense_runs = np.where(np.char.find(run_name, 'simple_v2') >= 0)[0]
    run_name = run_name[dense_runs]
    it = info[dense_runs, 1]
    
    suc_rate = np.array(info[dense_runs, 2], dtype=np.float)
    mean_len = np.array(info[dense_runs, 3], dtype=np.float)
    std_len = np.array(info[dense_runs, 4], dtype=np.float)
    mean_vel = np.array(info[dense_runs, 5], dtype=np.float)
    std_vel = np.array(info[dense_runs, 6], dtype=np.float)
    
    if len(filter_run) > 0:
        idx = np.where(np.char.find(run_name, filter_run) >= 0)[0]
        mean_vel_filtered = mean_vel[idx]
        mean_len_filtered = mean_len[idx]
    
    if SCALE:
        scaler = MinMaxScaler(copy=False)
        scaler.fit(mean_len.reshape(-1, 1))
        scaler.transform(mean_len.reshape(-1, 1))[:, 0]
        scaler.fit(mean_vel.reshape(-1, 1))
        scaler.transform(mean_vel.reshape(-1, 1))[:, 0]
        std_len /= std_len.max() - std_len.min()
        std_vel /= std_vel.max() - std_vel.min()
        
        if len(filter_run) > 0:
            idx = np.where(np.char.find(run_name, filter_run) >= 0)[0]
            mean_vel_filtered = mean_vel[idx]
            mean_len_filtered = mean_len[idx]
    
    if len(filter_run) > 0:
        idx = np.where(np.char.find(run_name, filter_run) >= 0)[0]
        mean_vel_filtered = mean_vel[idx]
        mean_len_filtered = mean_len[idx]

    fig = plt.figure(figsize=(15, 10), dpi=300)
    ax = fig.add_subplot('111')

    un_runs = np.unique(run_name)
    
    c_i = 0
    c_list = ['blue', 'red', 'green', 'purple', 'yellow']
    c_map = {}
    for run in un_runs:
        if 'simple_v2' not in run:
            continue
        c_map[run] = c_list[c_i]
        c_i += 1

    if not SHOW_TOP:

        for id_ in np.random.permutation(len(it)):
        
            run = run_name[id_]

            if suc_rate[id_] < 0.999:
                continue

            if SHOW_DOMINANT:

                val_1 = mean_len[id_]
                val_2 = mean_vel[id_]
                
                if len(filter_run) == 0:
                    if np.any([(val_1 > mean_len) & (val_2 < mean_vel)]):
                        continue
                else:
                    if run != filter_run or np.any([(val_1 > mean_len_filtered) & (val_2 < mean_vel_filtered)]):
                        continue
            
            print(run, it[id_])

            # print(mean_len[id_], mean_vel[id_], std_len[id_], std_vel[id_])
            ax.errorbar(mean_len[id_], mean_vel[id_], xerr=std_len[id_], yerr=std_vel[id_], marker='s', c=c_map[run],
                        alpha=1., markeredgecolor='black', elinewidth=2., ecolor='black')
            ax.errorbar(mean_len[id_], mean_vel[id_], xerr=std_len[id_], yerr=std_vel[id_], marker='s', c=c_map[run],
                        alpha=1., markeredgecolor='black', elinewidth=1)

    else:

        win_all = np.where(suc_rate > 0.999)
        all_metrics = np.stack([mean_len, std_len, mean_vel, std_vel], axis=-1)
        all_metrics = all_metrics[win_all]
        all_metrics[:, (0, 1, 3)] *= -1.
        all_metrics[:, (1, 3)] *= 0
        np.set_printoptions(threshold=np.inf)
        print(repr(all_metrics))
        

        id_ = np.argmax(np.sum(all_metrics - all_metrics.mean(axis=0), axis=-1))
        # id_ = np.argmax(np.sum(all_metrics))
        id_ = win_all[0][id_]
        run = run_name[id_]      

        ax.errorbar(mean_len[id_], mean_vel[id_], xerr=std_len[id_], yerr=std_vel[id_], marker='s', c=c_map[run],
                    alpha=1., markeredgecolor='black', elinewidth=2., ecolor='black')
        ax.errorbar(mean_len[id_], mean_vel[id_], xerr=std_len[id_], yerr=std_vel[id_], marker='s', c=c_map[run],
                    alpha=1., markeredgecolor='black', elinewidth=1)
        print(run, it[id_])

    ax.grid('on')
    # ax.set_xlim3d(0.95, 1.)
    if SHOW_DOMINANT:
        ax.set_title('PF of Mean Episode Length and Average Velocity to\n Ball for Agents with 100 % Success Rate')
    else:
        ax.set_title('Scatter Plot of Mean + Std of Episode Length and Average Velocity to\n Ball for Agents with 100 % Success Rate')
    if not SCALE:
        ax.set_xlim(75, 205)
        ax.set_ylim(2.2, 4.8)
        ax.set_ylabel('Average Vel. to Ball')
        ax.set_xlabel('Episode Length')
    else:
        ax.set_ylabel('Scaled Average Vel. to Ball')
        ax.set_xlabel('Scaled Episode Length')

    plt.show()
