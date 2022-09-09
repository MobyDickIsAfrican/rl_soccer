import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.lines import Line2D
import numpy as np
import random

# ploteando un cono:
def draw_agent(position, orientation,ax,home_away="home"):
    player_width = 3e-2
    radious = 5e-2
    orientation = orientation*np.pi/180
    proyected_position=tuple([position[0]+3*radious*np.cos(orientation),position[1]+3*radious*np.sin(orientation)])
    if ax is None:
        ax = plt.gca()
    if home_away=="home":
        color = 'b'
    else: 
        color='r'
    
    line = Line2D(*zip(position, proyected_position), linestyle='--', color='k', alpha=0.1)
    agent = Circle(position, player_width, color=color, alpha=1)
    ax.add_artist(line)
    ax.add_artist(agent)
    
    return agent

def draw_cone(position, orientation, ax, home_away='home'):
    radious = 5e-2
    angles = [orientation-45, orientation+45]
    if ax is None:
        ax = plt.gca()
    if home_away=="home":
        color = 'b'
    else: 
        color='r'
    w1 = Wedge(position, 3*radious, theta1=angles[0], theta2=angles[1], alpha=0.5, color=color)
    ax.add_artist(w1)
    return w1

def generate_teams(positions, orientations, teams, ax):
    for position, orientation, home_away in zip(positions, orientations, teams):
        draw_agent(position, orientation, ax, home_away=home_away)
        draw_cone(position, orientation, ax, home_away=home_away)


fig, ax = plt.subplots()
teams = ["home"]*2+["away"]*2
positions=[(random.random(), random.random()) for _ in range(4)]
orientations = [359*random.random() for _ in range(4)]
generate_teams(positions, orientations, teams, ax)
plt.show()