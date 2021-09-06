import numpy as np
import numpy.matlib as Mnp
import open3d as o3d
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_model(filename):
    '''
    Reads in 3D model
    :param filename:
    :return:
    '''
    print("Reading triangle Mesh ...")
    mesh = o3d.io.read_triangle_mesh(filename)
    print(mesh)
    return mesh

def compute_average_length(V, F):
    '''
    compute_average_length - compute the average length of a mesh
    Input:
         V,F are the vertex and face of a mesh.
    Output:
        lenth is the average length of a mesh

    :param V:
    :param F:
    :return:

    Copyright (c) 2012 Xiaochao Wang
    '''

    i = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    j = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])

    #A = scipy.sparse.csr_matrix( (i, j))
    #todo Implement sparse and find

    d = np.sqrt(np.square(np.subtract(V[i, :], V[j, :])))

    return np.sum(d) / len(d)


def display_mesh(mesh):
    '''

    :param mesh:
    :return:
    '''
    o3d.visualization.draw_geometries([mesh])


def compute_vertex_face_ring(face):
    '''
    % compute_vertex_face_ring - compute the faces adjacent to each vertex
    ring = compute_vertex_face_ring(face);

    Copyright (c) 2007 Gabriel Peyr?
    :param face:
    :return:
    '''

    nface = len(face)
    x = np.unravel_index(np.array(face).argmax(), face.shape)
    nverts = face[x[0]][x[1]]

    ring = [[] for _ in range(nverts + 1)]

    for i in range(nface):
        for j in range(3):
            ring[face[i][j]].append(i)

    return ring

def  show_feature_vertex(V,F,Sharp_edge_v, Corner_v):
    fig = plt.figure(figsize=(4, 4))

    ax = fig.add_subplot(111, projection='3d')

    for i in Sharp_edge_v:
        ax.scatter(V[i-1][0], V[i-1][1], V[i-1][2],color='blue')  # plot the point (2,3,4) on the figure


    for i in Corner_v:
        ax.scatter(V[i-1][0], V[i-1][1], V[i-1][2],color='red')  # plot the point (2,3,4) on the figure

    plt.show()


def show_vertex_salience(V, F, Cn):
    '''
    show_show_vertex_salience - plot salience of vertex on 3D mesh.

    show_vertex_salience(V,F,Salience)

        - 'V' : a (n x 3) array vertex coordinates
        - 'F' : a (m x 3) array faces
        - 'Salience' : (n x 1) array scalar salience value of each vertex

   Copyright (c) 2012 Xiaochao Wang
    :param V:
    :param F:
    :param Cn:
    :return:
    '''









    return None