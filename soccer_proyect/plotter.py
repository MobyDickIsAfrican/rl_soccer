import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.lines import Line2D
import numpy as np
import math
from dm_soccer2gym.wrapper import polar_mod, polar_ang
# ploteando un cono:
def fix_angle(angle):
    if abs(angle)>np.pi:
        if angle<0:
            return 2*np.pi + angle
        else: 
            return -2*np.pi + angle
    else: 
        return angle

def to_degree(angle):
    return angle*180/np.pi

def draw_agent(position, orientation,ax,home_away="home", number=1):
    player_width = 0.5
    radious = 5
    orientation = orientation
    if number==0:
        proyected_position=tuple([position[0]+radious*np.cos(orientation),position[1]+radious*np.sin(orientation)])
    else: 
        proyected_position=tuple([position[0]-radious*np.sin(orientation),position[1]+radious*np.cos(orientation)])
    if ax is None:
        ax = plt.gca()
    if home_away=="home":
        color = 'b'
    else: 
        color='r'
    
    line = Line2D(*zip(position, proyected_position), linestyle='--', color='k', alpha=0.1)
    agent = Circle(position, player_width, color=color, alpha=1)
    ax.text(*position, str(number))
    ax.add_artist(line)
    ax.add_artist(agent)
    
    return agent

def draw_cone(position, orientation, ax, home_away='home', number=0, angle=np.pi/4, radious=5):
    if number==0:
        orientation = fix_angle(orientation)
        angles = [fix_angle(orientation-angle), fix_angle(orientation+angle)]
        
    else:
        orientation += np.pi/2
        angles = [fix_angle(orientation-angle), fix_angle(orientation+angle)]

    angles = [to_degree(a_angle) for a_angle in angles]
    if ax is None:
        ax = plt.gca()
    if home_away=="home":
        color = 'b'
    else: 
        color='r'
    w1 = Wedge(position, radious, theta1=angles[0], theta2=angles[1], alpha=0.5, color=color)
    

    ax.add_artist(w1)
    return w1

def generate_teams(positions, orientations, teams, ax):
    for i, (position, orientation, home_away) in enumerate(zip(positions, orientations, teams)):
        draw_agent(position, orientation, ax, home_away=home_away, number=i)
        draw_cone(position, orientation, ax, home_away=home_away, number=i)
        if i==0:
            [draw_pass_cone(position, a_position, a_orientation, ax, home_away) for j, (a_position, a_orientation) in enumerate(zip(positions, orientations)) if i!=j and teams[j]==home_away]

def generate_ball(position, ax=None):
    if ax is None:
        ax = plt.gca()
    imagebox = Circle(position, 0.3, color='k', alpha=1)
    
    ax.add_artist(imagebox)

def get_angle(rotation_matrix):
    rotation_matrix = rotation_matrix.reshape(3,3)
    phi = np.arctan2(rotation_matrix[1,0],rotation_matrix[0,0])
    return phi

def generate_text(Area_observations, ax=None):
    area_text = [f"Area_0,{i+1}: {Area_observations[i]:.2f}" for i in range(len(Area_observations))]
    if ax is None:
        ax = plt.gca()
    ax.text(-24, 20, ", ".join(area_text))    


def draw_pass_cone(first_player_pos, position, orientation, ax, home_away='home'):
    radious=2.5
    angle = np.pi/6
    proyected_position = np.array([position[0]-radious*np.sin(orientation),position[1]+radious*np.cos(orientation)])
    radious = polar_mod(proyected_position)
    angle_ego = polar_ang(proyected_position[None, ...])
    draw_cone(first_player_pos, angle_ego, ax, home_away, angle=angle/2, radious=radious)

