import math
import open3d as o3d
from numba import jit, cuda
import numpy
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
    #mesh = model3D.read_model('Models/OFF/cube.off')
    mesh = model3D.read_model('Models/OFF/fandisk_noise.off')
    #mesh = model3D.read_model('Models/PLY/Chapel.ply')
    #mesh = model3D.read_model('Models/PLY/mba1.ply')
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)

    #line_set= o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    halfedge = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(mesh)


    #graph_plotter.show_feature_line(V, F, Edge)
    #Edge = normal_tensor_voting.read_file_u('Test/Edge.txt')

     
    # one ring vertices of a vertex
    mesh.compute_adjacency_list()
    one_ring_vertex = np.asarray(mesh.adjacency_list)

    # one ring faces of a vertex
    one_ring_face = model3D.compute_vertex_face_ring(F)
    nov = len(V)


    #Expected value 0.2845
    thgm = model3D.compute_average_length(V,F)

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
                                                            beta)#plotting wromg answer



    # show mesh and feature vertex
    #graph_plotter.show_feature_vertex(V, F, Sharp_edge_v, Corner_v)

    #
    #PART2: Salience measure computation
    #

    #model3D.show_feature_vertex(V, F, Sharp_edge_v, Corner_v)
    Salience , EHSalience , LENTH = normal_tensor_voting.compute_enhanced_salience_measure(V,F,D,PRIN,EVEN,Sharp_edge_v, Corner_v)


    # show the salience measure
    TH = []
    Id = []
    Temp = []

    Temp = [Salience[Sharp_edge_v[i]] for i in range(len(Sharp_edge_v))]
    Temp.sort()

    TH = Temp[math.floor(len(Temp) * 0.80):]
    mean = np.mean(TH)


    for i in range(len(Corner_v)):
        Salience[Corner_v[i]] = mean

    TH = []
    Id = []
    Temp = []
    Temp = [EHSalience[Sharp_edge_v[i]] for i in range(len(Sharp_edge_v))]
    Temp.sort()
    TH = Temp[math.floor(len(Temp) * 0.80):]

    mean = np.mean(TH)
    for i in range(len(Corner_v)):
        EHSalience[Corner_v[i]] = mean

    #show the salience before enhanced
    #show_vertex_salience(V,F,-Salience)

    # show the salience after enhances
    # show_vertex_salience(V,F,-EHSalience);

    #set threshold to filter the false feature points
    temp = [EHSalience[Sharp_edge_v[i]] for i in range(len(Sharp_edge_v))]
    temp.sort()


    x = np.linspace(0,1,len(temp))


    x = [[i] for i in x]

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
        graph_plotter.show_feature_vertex(V, F, F_R_P, Corner_v)

        Door = input('Filter Non Feature Vertex -- The result is OK? Input: y or n:--')
        if Door == 'y' or Door == 'Y':
            break


    #
    #PART 3: Connect the feature point to feature line
    #

    '''
    Sharp_edge_v = []
    file = open('TEST\\FRP.txt')
    for i in file:
        Sharp_edge_v.append(int(i.strip()))
    file.close()
    '''

    Edge =  normal_tensor_voting.connect_feature_line(halfedge,F_R_P,Corner_v)
    graph_plotter.show_feature_line(V, F, Edge)


    '''
    graph_plotter.show_feature_line(V,F, Edge)

    #PART 4: Filter the feature lines via edge measure (additional prunning process)
    #if we get optimal result through above code, the following processes can be not implemented.
    noe = len(Edge)

    #give privilege to corver
    for i in Corner_v:
        LENTH[i] = 5

    #compute a salience measure measure to each edge
    ELENT = np.zeros(noe)

    for i in noe:
        ELENT[i] = EHSalience[Edge[i][0]]*EHSalience[Edge[i][1]]*LENTH[i][0]*LENTH[i][1]

    temp = ELENT.sort()
    x = np.linspace(0, 1, len(temp))

    while True:
        TH = input('Input a threshold based on edge strength:-- ')

        Id = []

        for i in range(len(Sharp_edge_v)):
            if EHSalience[Sharp_edge_v[i]] > TH:
                Id.append(i)

        Door = input('Filter Non Feature Vertex -- The result is OK? Input: y or n:--')
        if Door == 'y' or Door == 'Y':
            break

    Edge = []
    #Edge = Edge1

    #Further filter extra feature edge via interactive manner
    #deal with joint feature edges

    Edge = normal_tensor_voting.postprocessing_filter_joints(V,F,Edge,D,Corner_v)

    #PART 5: Prolong the existing featuares to get long and closed feature line (not included in artical)
    #parameters

    min_k = 7 # the smallest feature lines we want to preserv
    max_k = 10 # the maxmum steps allowed to prolong

    #old edge information
    O_edge = Edge

    #Iterative adjust parameters to get pleasant result
    while True:
        #get close and prolonged feature edges


        Door = input('Prolong Feature Edge -- The result is OK?  Input: y or n:--')
        if Door.lower() == 'y' :
            break
        else:
            print('The previous max_k is %f: ', max_k);
            max_k = eval(input('Please input a new alpha: '))
            print('The previous min_k is %f: ', min_k)
            min_k = eval(input('Please input a new alpha: '))

            Edge = O_edge

    #
    #PART 6: Delete small circles
    #

    #delte small circles
    Edge = normal_tensor_voting.posprocessing_delete_small_circle(Edge, D)
    '''

def pymesh_test():
    print(trimesh)


if __name__ == '__main__':
    run_progarm()
    #pymesh_test()
