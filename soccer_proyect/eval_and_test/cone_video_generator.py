from environments.env2vs0 import Env2vs0
from environments.env2vs2Dense import Env2vs2
from eval_and_test.plotter import * 
import os
import torch
from math import ceil
import copy

def generate_cone_2vs2_vid():
    for run in range(10):
        # loadear un equipo de 2vs0 que no esté entrenado

        # generar un gráfico que tenga la foto y referenciar todo hacia el jugador 1: 
        fig, ax = plt.subplots()
        max_ep_len = ceil(30 / 0.1)
        images = []
        home = 2
        away = 2
        # loadear ambiente de 2vs0 o incluso 2vs2 o 2vs1:
        env  = Env2vs2(team_1=home, team_2=away,task_kwargs={ "time_limit": 30, "disable_jump": True, 
                                "dist_thresh": 0.03, 'control_timestep': 0.1, "random_seed":30, "observables": "all"}, render_mode_list=[30])
        # Iniciar ambiente
        obs, d, ep_ret, ep_len = env.reset(), False, np.array([0]*(4), dtype='float32'), 0
        home_obs, away_obs = obs
        # observación original
        obs_original = env.timestep.observation[0]
        while not(d or (ep_len == max_ep_len)):
            # calcular posiciones de los jugadores con referencia la jugador 0:
            # calcular posiciones con referencia a la cancha (como en el test):
            own_orientation, teammate_pos, teammate_orientation, opp_pos, opp_orientation = env.get_positions_orientations(obs_original, 0)
            player_position2pitch_position = [np.array(env.dmcenv._physics_proxy.bind(env.dmcenv.task.players[0].walker.root_body).xpos[:2]) for i in range(1+len(teammate_pos+opp_pos))]
            # calcular interseccion de áreas:
            intercept_areas = env.get_Area(obs_original, 0)
            # reiniciar plot
            
            ax.cla()
            # mover esto mas arriba
            #images.append(copy.deepcopy(env.render()[0]))
            #ax.imshow(env.render()[0])
            # definir la distribución de equipos (deberías hacerlo con team_1 y team_2 del soccer env)
            teams=["home"]*(1+len(teammate_pos)) + ["away"]*(1+len(opp_pos))
            # seteamos el bound 
            ax.set_xbound(-env.pitch_size[0], env.pitch_size[0])
            ax.set_ybound(-env.pitch_size[1], env.pitch_size[1])
            # generar los equipos en la foto de la cancha
            all_player_pos = [np.array([0,0])]+teammate_pos + opp_pos
            all_player_orientation = [own_orientation] + teammate_orientation + opp_orientation
            players_pos = [ego_ref for pitch_ref, ego_ref in zip(player_position2pitch_position, all_player_pos)]
            generate_teams(players_pos, all_player_orientation, teams, ax)
            # generar textos de areas e intersecciones en la foto de la cancha
            o_area = [value for key, value in intercept_areas.items() if "intercept" in key and "area" in key and not "pass" in key]
            o_area_pass = [value for key, value in intercept_areas.items() if "pass" in key and "area" in key]
            o_centroid = [value for key, value in intercept_areas.items() if "centroid" in key and "intercept" in key and not "pass" in key]
            o_pass_centroid = [value for key, value in intercept_areas.items() if "centroid" in key and "pass" in key]
            generate_text(o_area+o_area_pass, ax) 
            # generar los textos de los centroides
            generate_centroid_text(o_centroid+o_pass_centroid, ax)
            fig.tight_layout()
            plt.waitforbuttonpress(3)
            intercept_areas = env.get_Area(obs_original, 0)
            a = [env.action_space.sample() for _ in range(home+away)]
            obs, r, d, _ = env.step(a)
            obs_original = env.timestep.observation[0]
            ep_len += 1


if __name__ == "__main__":
    generate_cone_2vs2_vid()