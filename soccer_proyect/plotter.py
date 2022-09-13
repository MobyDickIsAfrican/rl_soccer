import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.lines import Line2D
import numpy as np
import math
# ploteando un cono:
def draw_agent(position, orientation,ax,home_away="home", number=1):
    player_width = 0.5
    radious = 5
    proyected_position=tuple([position[0]+radious*np.cos(orientation),position[1]+radious*np.sin(orientation)])
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

def draw_cone(position, orientation, ax, home_away='home'):
    radious = 5
    orientation = orientation*180/math.pi
    angles = [orientation-45, orientation+45]
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
        draw_cone(position, orientation, ax, home_away=home_away)

def generate_ball(position, ax=None):
    if ax is None:
        ax = plt.gca()
    imagebox = Circle(position, 0.3, color='k', alpha=1)
    
    ax.add_artist(imagebox)

def get_angle(rotation_matrix):
    rotation_matrix = rotation_matrix.reshape(3,3)
    phi = np.arctan(rotation_matrix[1,0]/rotation_matrix[0,0])
    return phi