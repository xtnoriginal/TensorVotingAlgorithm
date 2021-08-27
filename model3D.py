import numpy as np
import open3d as o3d

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