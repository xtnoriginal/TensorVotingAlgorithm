import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def  show_feature_vertex(V,F,Sharp_edge_v, Corner_v):
    fig = plt.figure(figsize=(4, 4))

    ax = fig.add_subplot(111, projection='3d')

    for i in Sharp_edge_v:
        ax.scatter(V[i][0], V[i][1], V[i][2],color='blue')  # plot the point (2,3,4) on the figure


    for i in Corner_v:
        ax.scatter(V[i][0], V[i][1], V[i][2],color='red')  # plot the point (2,3,4) on the figure

    plt.show()


def show_feature_line(V,F,Edge):

    fig = plt.figure(figsize=(10, 10))

    ax = plt.axes(projection='3d')
    ax.mouse_init()

    for i in range(len(Edge)):
        Temp = Edge[i]

        ax.scatter(V[int(Temp[0])][0], V[int(Temp[0])][1], V[int(Temp[0])][2], color='blue')
        ax.scatter(V[int(Temp[1])][0], V[int(Temp[1]) ][1], V[int(Temp[1])][2], color='blue')



    for i in range(len(Edge)):
        Temp =  Edge[i]

        x = [V[int(Temp[0])][0], V[int(Temp[1])][0]]
        y = [V[int(Temp[0])][1], V[int(Temp[1])][1]]
        z = [V[int(Temp[0])][2], V[int(Temp[1])][2]]

        ax.plot3D(x , y , z ,'green')

    plt.show()
    #plt.savefig("out.png")



def plot_line(x,y):
    plt.plot(x,y)
    plt.show()

