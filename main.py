import numpy as np

import model3D


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

    # model3D.compute_average_length(V,F)
    thgm = 0.2845  # todo implement method


    #Display model
    model3D.display_mesh(mesh)

if __name__ == '__main__':
    run_progarm()


