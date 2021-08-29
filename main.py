import math

import numpy as np
import pymesh
import potpourri3d as pp3d

import graph_plotter
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
    '''
    # show mesh and feature vertex
    model3D.show_feature_vertex(V, F, Sharp_edge_v, Corner_v)

    #
    #PART2: Salience measure computation
    #

    Salience , EHSalience , LENTH= normal_tensor_voting.compute_enhanced_salience_measure(V,F,D,PRIN,EVEN,Sharp_edge_v, Corner_v)

    # show the salience measure
    TH = []
    Id = []
    Temp = []

    Temp = [Salience[Sharp_edge_v[i]] for i in range(len(Sharp_edge_v))]
    Temp.sort()#getting zeros first
    TH = Temp[math.floor(len(Temp)*0.80):]

    mean = np.mean(TH)
    for i in range(len(Corner_v)):
        Salience[Corner_v[i]] = mean

    Temp = [EHSalience[Sharp_edge_v[i]] for i in range(len(Sharp_edge_v))]
    Temp.sort()  #todo getting zeros first
    TH = Temp[math.floor(len(Temp) * 0.80):]

    mean = np.mean(TH)
    for i in range(len(Corner_v)):
        Salience[Corner_v[i]] = mean

    #show the salience before enhanced
    #show_vertex_salience(V,F,-Salience)

    # show the salience after enhances
    # show_vertex_salience(V,F,-EHSalience);

    #set threshold to filter the false feature points
    temp = [EHSalience[Sharp_edge_v[i]] for i in range(len(Sharp_edge_v))]
    x = np.linspace(0,1,len(temp))
    x.transpose()

    graph_plotter.plot_line(x,temp)
    #interactive select threshold

    while True:

        TH = float(input('Input a threshold based on salience measure:--'))
        Id = []

        for i in range(len(Sharp_edge_v)):
            if EHSalience[Sharp_edge_v[i]]> TH:
                Id.append(i)

        F_R_P = [Sharp_edge_v[i] for i in Id]

        #show feature vertex
        model3D.show_feature_vertex(V,F, F_R_P, Corner_v)

        Door = input('Filter Non Feature Vertex -- The result is OK? Input: y or n:--')
        if Door == 'y' or Door == 'Y':
            break


    #
    #PART 3: Connect the feature point to feature line
    #

    Sharp_edge_v, Corner_v, Edge =  normal_tensor_voting.connect_feature_line(1,F_R_P,Corner_v)
    graph_plotter.show_feature_line(V,F, Edge)

    '''




def pymesh_test():
    print(trimesh)


if __name__ == '__main__':
    run_progarm()
    #pymesh_test()
