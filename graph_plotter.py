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


    fig = plt.figure(figsize=(4, 4))

    ax = plt.axes(projection='3d')
    ax.mouse_init()

    for i in range(len(Edge)):
        Temp = Edge[i]

        ax.scatter(V[int(Temp[0])-1][0], V[int(Temp[0])-1][1], V[int(Temp[0])-1][2], color='blue')
        ax.scatter(V[int(Temp[1])-1][0], V[int(Temp[1]) - 1][1], V[int(Temp[1]) - 1][2], color='blue')



    for i in range(len(Edge)):
        Temp =  Edge[i]

        x = [V[int(Temp[0])-1][0], V[int(Temp[1])-1][0]]
        y = [V[int(Temp[0])-1][1], V[int(Temp[1])-1][1]]
        z = [V[int(Temp[0])-1][2], V[int(Temp[1])-1][2]]

        ax.plot3D(x , y , z ,'green')

    # rotate the axes and update
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.001)

    plt.show()



def plot_line(x,y):


    plt.plot(x,y)
    plt.show()

x = [1.4844,
    1.4844,
    1.4844,
    1.4844,
    1.4844,
    1.4844,
    1.4844,
    1.4844,
    1.4844,
    1.4844,
    1.4844,
    1.4844,
    1.4844,
    1.4844,
    1.4844,
    1.4844,
    1.4844,
    1.4844,
    1.4844,
    1.4844,
    1.4844,
    1.4844,
    1.4844,
    1.4844,
    1.9247,
    1.9247,
    1.9247,
    1.9247,
    1.9247,
    1.9247,
    1.9247,
    1.9247,
    1.9247,
    1.9247,
    1.9247,
    1.9247,
    1.9247,
    1.9247,
    1.9247,
    1.9247,
    1.9247,
    1.9247,
    1.9247,
    1.9247,
    1.9247,
    1.9247,
    1.9247,
    1.9247,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
    2.0888,
]

y = [    0,
    0.0120,
    0.0241,
    0.0361,
    0.0482,
    0.0602,
    0.0723,
    0.0843,
    0.0964,
    0.1084,
    0.1205,
    0.1325,
    0.1446,
    0.1566,
    0.1687,
    0.1807,
    0.1928,
    0.2048,
    0.2169,
    0.2289,
    0.2410,
    0.2530,
    0.2651,
    0.2771,
    0.2892,
    0.3012,
    0.3133,
    0.3253,
    0.3373,
    0.3494,
    0.3614,
    0.3735,
    0.3855,
    0.3976,
    0.4096,
    0.4217,
    0.4337,
    0.4458,
    0.4578,
    0.4699,
    0.4819,
    0.4940,
    0.5060,
    0.5181,
    0.5301,
    0.5422,
    0.5542,
    0.5663,
    0.5783,
    0.5904,
    0.6024,
    0.6145,
    0.6265,
    0.6386,
    0.6506,
    0.6627,
    0.6747,
    0.6867,
    0.6988,
    0.7108,
    0.7229,
    0.7349,
    0.7470,
    0.7590,
    0.7711,
    0.7831,
    0.7952,
    0.8072,
    0.8193,
    0.8313,
    0.8434,
    0.8554,
    0.8675,
    0.8795,
    0.8916,
    0.9036,
    0.9157,
    0.9277,
    0.9398,
    0.9518,
    0.9639,
    0.9759,
    0.9880,
    1.0000]

#plot_line(x,y)