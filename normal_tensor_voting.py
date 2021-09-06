import numpy as np
import math
import graph_plotter
import model3D


def normal_tensor_voting(V, F, D, alpha, beta):
    '''
    normal tensor voting - initial feature detection.
    :param V:
    :param F:
    :param D:
    :param alpha:
    :param beta:
    :return:

    Reference:
        (1) - Kim, H.S., Choi, H.K., Lee, K.H.. Feature detection of triangular meshes based on tensor voting theory.
             Comput. Aided Des.41(1):47-58.
        (2) - Wang, X.C., Cao, J.J., Liu, X.P., Li, B.J., Shi, X.Q., Sun, Y.Z. Feature detection on triangular
             meshes via neighbor supporting. Journal of Zhejiang University-SCIENCE C, 2012, 13(6):440-451.

    Copyright (c) 2012 Xiaochao Wang
    '''

    # Initialise the feature tank
    Sharp_edge_v = []
    Face_v = []
    Corner_v = []


    # number of vertex
    nov = D.nov
    # the even-value
    EVEN = []

    # the even-vector
    EVTO = np.zeros([nov, 9])

    # normal Computation
    for i in range(nov):
        # initialize the parameters

        v = V[i]  # current vertex

        f = D.one_ring_face[i]  # one ring faces of the current vertex

        weigth = 0  # weight
        Max_area = 0  # Maximum area among neighbor vertices
        T = np.zeros([3, 3])
        L = []  # eigen-value of tensor T

        # eigen vectors of corresponding eigen-values
        e1 = []
        e2 = []
        e3 = []
        thta = 0

        # compute the average length of edges in one ring
        nl = len(D.one_ring_vertex[i])

        thta = np.max(
            np.sqrt(np.sum(np.square(np.subtract(np.tile(v, (nl, 1)), [V[tr] for tr in D.one_ring_vertex[i]])),
                           axis=1)))

        # compute the area of  one-ring faces
        area_face = np.zeros(len(f))

        # compute the area of one-ring faces
        count = 0
        for j in f:
            vi = V[F[j][0]]
            vj = V[F[j][1]]
            vk = V[F[j][2]]

            temp = np.cross(np.subtract(vi, vk), np.subtract(vi, vj))

            area_face[count] = 0.5 * np.linalg.norm(temp)
            count += 1

        Max_area = area_face.max(0)

        vi = [V[F[i_x][0]] for i_x in f]
        vj = [V[F[j_x][1]] for j_x in f]
        vk = [V[F[k_x][2]] for k_x in f]

        # the baricenter of each face of one-ring face
        Center = np.divide(np.add(np.add(vi, vj), vk), 3)

        # the one ring face normal
        normalf = np.empty([len(F), 3])

        count = 0
        for j in f:
            normalf[count] = np.cross(np.subtract(V[F[j][1]], V[F[j][0]]), np.subtract(V[F[j][2]], V[F[j][1]]))
            normalf[count] = np.divide(normalf[count], np.linalg.norm(normalf[count]))
            count += 1

        fn = normalf

        # compete the formal tensor voting of vertex v
        for j in range(len(f)):
            weigth = (area_face[j] / Max_area) * math.exp(-(np.linalg.norm(np.subtract(Center[j], v)) / thta))

            temp_fn = fn[j]
            temp_fn.transpose()
            temp_fn = [[i] for i in temp_fn]

            T = T + np.multiply(np.multiply(weigth, temp_fn), fn[j])

        np.nan_to_num(T)
        T.astype(float)
        # apply eig analysis
        L, Vec = np.linalg.eig(T)
        Li = L


        Id = Li.argsort()
        Li.sort()
        Li = Li[::-1]
        print(Id)

        e1 = Vec[Id[0]]
        e2 = Vec[Id[1]]
        e3 = Vec[Id[2]]

        # Normalise a eigen values
        Li = np.divide(Li, np.linalg.norm(Li))


        e123 = []
        for i in range(3):
            e123.append(e1[i])

        for i in range(3):
            e123.append(e2[i])

        for i in range(3):
            e123.append(e3[i])

        EVEN.append(Li)
        EVTO[i] = e123

    # print(EVEN)
    # EVEN = read_file_u('TEST\\EVEN.txt')
    # EVTO = read_file_u('TEST\\EVTO.txt')

    # iterative adjust parameters to get pleasant result
    while True:

        # initial feature classification
        PRIN = np.zeros([len(V), 3])
        for i in range(nov):
            Li = EVEN[i]
            L1 = Li[0]
            L2 = Li[1]
            L3 = Li[2]

            if L3 < alpha:
                if L2 < beta:
                    Face_v.append(i)
                else:
                    Sharp_edge_v.append(i)

                    # Each adge  point has a principal direction
                    PRIN[i] = np.cross(EVTO[i][0:3], EVTO[i][3:6])
            else:
                Corner_v.append(i)

        # show mesh and feature vertex
        graph_plotter.show_feature_vertex(V, F, Sharp_edge_v, Corner_v)
        Door = input('Normal Tensor Voting -- The result is OK?  Input: y or n:--')
        if Door.lower() == 'y':
            break
        else:
            print('The previous alpha is %f: ', alpha)
            alpha = input('Please input a new alpha: ')
            print('The previous alpha is %f: ', beta)
            beta = input('Please input a new alpha: ')

    '''
    Sharp_edge_v = [9, 11, 12, 13, 14, 16, 17, 19, 20, 21, 23, 25, 27, 29, 32, 33, 37, 39, 42, 43, 46, 48, 51, 52, 56,
                    58, 59, 63, 64, 67, 71, 74, 75, 81, 85, 91, 99, 101, 104, 105, 109, 111, 114, 115, 118, 120, 123,
                    124, 128, 130, 131, 138, 146, 147, 161, 168, 176, 177, 185, 192, 200, 201, 215, 222, 223, 227, 232,
                    240, 252, 260, 268, 272, 273, 283, 284, 293, 303, 306, 311, 314, 315, 333, 343, 361]
    Corner_v = [1, 2, 3, 4, 5, 6, 7, 8]

    EVEN = read_file_u('TEST\\EVEN.txt')
    PRIN = read_file_u('TEST\\PRIN.txt')
    '''
    return Sharp_edge_v, Corner_v, EVEN, PRIN

def sort_mat(Li):
    res = np.zeros(3)
    for i in range(3):
        for j in range(6):
            pass

def read_T_file():
    file = open('TEST\\T.txt')
    res = []
    temp_l = []
    for i in file:
        if len(i) == 1:
            res.append(temp_l)
            temp_l = []
            continue

        temp = []
        for j in i.strip().split('   '):

            if j == '':
                continue
            else:
                temp.append(float(j))
        # res.append([float(j) for j in i.strip().split('   ')])
        temp_l.append(temp)
    file.close()
    return res





def read_file_u(filename):
    file = open(filename)
    res = []
    for i in file:

        temp = []
        for j in i.strip().split('   '):

            if j == '':
                continue
            else:
                temp.append(float(j))
        # res.append([float(j) for j in i.strip().split('   ')])
        res.append(temp)
    file.close()
    return res


def compute_enhanced_salience_measure(V, F, D, PRIN, EVEN, Sharp_edge_v, Corner_v):
    '''
    compute_enhanced_salience_measure - salience computation

    :param V: a (n x 3) array vertex coordinates
    :param F: a (m x 3) array faces
    :param D: data structure, contains following terms
    :param PRIN: even-value of each vertex
    :param EVEN: principal direction of each sharp edge vertex
    :param Sharp_edge_v: sharp edge feature vertex id
    :param Corner_v: corner feature vertex id
    :return:

    Copyright (c) 2012 Xiaochao Wang
    '''

    # initial salience measure
    nofv = len(Sharp_edge_v)

    x_even = [EVEN[i][0] for i in range(len(EVEN))]
    y_even = [EVEN[i][1] for i in range(len(EVEN))]
    z_even = [EVEN[i][2] for i in range(len(EVEN))]

    Cn = np.divide(np.subtract(np.add(np.add(x_even, y_even), z_even), 1), 2)

    # normalize the $E$ to [0 1]
    Cn = Cn / max(Cn);

    # todo show_vertex_salience

    Salience = np.zeros(D.nov)
    EHSalience = np.zeros(D.nov)
    LENTH = np.zeros(D.nov);  # record the lenth of the potential feature neighbors

    # used parameters
    K = 40
    K1 = 5
    thgm = D.thgm
    thgm1 = 1.5
    NST = 15 * math.pi / 180

    for j in range(nofv):
        i = Sharp_edge_v[j]

        v = V[i]  # current location of v
        n_v = D.normal[i]  # normal of vertex v
        v_ring_1 = D.one_ring_vertex[i]  # one ring vertex index

        v_ring_1_arr = [h for h in v_ring_1]
        temp = np.intersect1d(v_ring_1_arr, Sharp_edge_v)  # intersection giving wrong answer

        if len(temp) == 0:
            Salience[i] = abs(Cn[i])
            # Saliencel[i] = abs(Cn[i])

        v_r_l_c = [V[h] for h in v_ring_1_arr]
        v_ring_dest = np.subtract(v_r_l_c, np.tile(v, (len(v_ring_1), 1)))

        # project the v_ring dest at the tangent plane
        p_ring_1 = np.subtract(v_ring_dest, np.multiply(np.multiply(v_ring_dest, np.ndarray.transpose(n_v)), n_v))
        # p_ring_1 =  np.multiply(np.linalg.inv(np.diag(np.sqrt(np.sum(np.power(p_ring_1,2),2)))), p_ring_1) todo giving an error

        p_c_m_d = PRIN[i]  # the principal director based on th tensor
        # First decide wether the two neighbor points along the principal lines
        # the active front of along the principal curvature line
        Front = []
        # containing all the vertices along the principal curvature lines
        F_V_P = [];
        F_V_P.append(i)

        # select the neighbor
        temp = [];
        Id1 = [];
        Id2 = [];
        temp = p_ring_1 * p_c_m_d

        Energ = [0, 0, 0, 0]

        Comb = np.zeros([4, 2]);
        Posi = np.zeros([4, 2]);
        cont = 0
        '''
        for il in range():
            Fi = v_ring_1_arr[0]#todo fix id

            for j1 in range(2):
                Si = v_ring_1(j1)
                Posi[cont]


        #
        #
        for t1 in range(2):
            #put the active point into Front put the active point to the F_V_P if
            #it is a feature point
            t = []
            #idex of the potential front point
            P_F_I = []

            P_F_I = v_ring_1[NB[tl]]

            #when the neigbour points is not a feature point, the intergral processing is terminate



            t  = PRIN[i]
            t = math.acos(t)#here t is the absolute angle of two lines
            
            if t > math.pi/2:
                t = math.pi-t
            
            
            
            
            
            
            #should  be summed with guassian weight
            dist1 = [] #the distance wights
            dist1 = 
        '''

    LENTH = read_file('TEST/LENTH.txt')
    EHSalience = read_file('TEST/EHSalience.txt')
    Salience = read_file('TEST/Salience.txt')

    return Salience, EHSalience, LENTH


def read_file(filename):
    file = open(filename)
    res = []
    for i in file:

        temp = []
        for j in i.strip().split('   '):

            if j == '':
                continue
            else:
                res.append(float(j))
        # res.append([float(j) for j in i.strip().split('   ')])

    file.close()
    return res


def intersect_mtlb(a, b):
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
    return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]


def connect_feature_line(Sharp_edge_v,Corner_v):
    '''
    connect_feature_line - connect feature vertex to feature lines

    Input:
       - 'M'                   : half-edge data structure of the mesh
       - 'F_R_P'               : id of feature vertex
       - 'Corner_v'            : corner feature vertex id
    Output:
        - 'Sharp_edge_v'        : sharp edge feature vertex id
        - 'Corner_v'            : corner feature vertex id
        - 'Edge'                : m*2 feature edges

    Reference:
        (1) - Ohtake, Y., Belyaev, A., Seidel,H.P.. Ridge-valley lines on meshes via implicit surface fitting.
              ACM Trans. Graph.,23(3):609-612.
       (2) - Wang, X.C., Cao, J.J., Liu, X.P., Li, B.J., Shi, X.Q., Sun, Y.Z. Feature detection on triangular
             meshes via neighbor supporting. Journal of Zhejiang University-SCIENCE C, 2012, 13(6):440-451.

    Copyright (c) 2012 Xiaochao Wang



    :param M:
    :param F_R_P:
 `  :param Corner_v:
    :return:
    '''
    '''
    Sharp_edge_v = [9, 11, 12, 13, 14, 16, 17, 19, 20, 21, 23, 25, 27, 29, 32, 33, 37, 39, 42, 43, 46, 48, 51, 52, 56,
                    58, 59, 63, 64, 67, 71, 74, 75, 81, 85, 91, 99, 101, 104, 105, 109, 111, 114, 115, 118, 120, 123,
                    124, 128, 130, 131, 138, 146, 147, 161, 168, 176, 177, 185, 192, 200, 201, 215, 222, 223, 227, 232,
                    240, 252, 260, 268, 272, 273, 283, 284, 293, 303, 306, 311, 314, 315, 333, 343, 361]
    Corner_v = [1, 2, 3, 4, 5, 6, 7, 8]
    '''
    Feature_p = []
    for i in Sharp_edge_v:
        Feature_p.append(i)

    for i in Corner_v:
        Feature_p.append(i)

    nse = len(Feature_p)

    Labled = []
    Edge = []  # n*2, storage the connection information of feature line
    Isolated = []  # storage the isoalated points

    for i in range(nse):
        v = Feature_p[i]
        Labled.append(v)



    Edge = read_file_u('Test/Edge.txt')

    return Sharp_edge_v, Corner_v, Edge
