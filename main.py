import numpy as np
import pymesh
import potpourri3d as pp3d
import model3D
import trimesh

import normal_tensor_voting


class DataStructure:

    def __init__(self, one_ring_vertex, one_ring_face, thgm, normal, nov):
        '''
        :param one_ring_vertex:
        :param one_ring_face:
        :param thgm:
        :param normal:
        :param nov:
        '''
        self.one_ring_vertex = one_ring_vertex
        self.one_ring_face = one_ring_face
        self.thgm = thgm
        self.normal = normal
        self.nov = nov


def run_progarm():
    # read models
    mesh = model3D.read_model('Models/OFF/cube.off')
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)

    # one ring vertices of a vertex
    mesh.compute_adjacency_list()
    one_ring_vertex = np.asarray(mesh.adjacency_list)

    # one ring faces of a vertex
    one_ring_face = model3D.compute_vertex_face_ring(F)
    nov = len(V)


    #Expected value 0.2845
    thgm = model3D.compute_average_length(V,F)  # todo implement method

    # compute vertex and face normals
    mesh.compute_vertex_normals()

    normalf = np.asarray(mesh.triangle_normals)
    normal = np.asarray(mesh.vertex_normals)

    D = DataStructure(one_ring_vertex, one_ring_face, thgm, normal, nov)

    # Display model
    #model3D.display_mesh(mesh)

    # Part1: Detect the initial feature vertex based on normal tensor voting
    # Parameters
    alpha = 0.065
    beta = 0.020

    # Normal tensor voting
    # [Sharp_edge_v, Corner_v, Even, PRIN] = normal_tensor voting(V,F,D,alpha,beta)
    Sharp_edge_v, Corner_v, EVEN, PRIN = normal_tensor_voting.normal_tensor_voting(V, F, D, alpha,
                                                                           beta)  # todo finish of tensor voting
def pymesh_test():
    print(trimesh)


if __name__ == '__main__':
    run_progarm()
    #pymesh_test()
