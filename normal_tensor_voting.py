import numpy as np

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
    EVEN = np.zeros([len(V), 3])

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

        thta = np.max(np.sqrt(np.sum(np.square(np.subtract(np.tile(v, (nl, 1)), [V[tr] for tr in D.one_ring_vertex[i]])),
                             axis=1)))

        # compute the area of  one-ring faces
        area_face = np.zeros(len(F))

        # compute the area of one-ring faces
        for j in f:
            vi = V[F[j][0]]
            vj = V[F[j][1]]
            vk = V[F[j][2]]

            temp = np.cross(np.subtract(vi, vk), np.subtract(vi, vj))

            area_face[j] = 0.5 * np.linalg.norm(temp)
        Max_area = area_face.max(0)

        vi = [V[F[i_x][0]] for i_x in f]
        vj = [V[F[j_x][1]] for j_x in f]
        vk = [V[F[k_x][2]] for k_x in f]

        # the baricenter of each face of one-ring face
        Center = np.divide(np.add(np.add(vi, vj), vk), 3)

        # the one ring face normal
        normalf = np.empty([len(F), 3])

        count = 0;
        for j in f:
            normalf[count] = np.cross(np.subtract(V[F[j][1]], V[F[j][0]]), np.subtract(V[F[j][2]], V[F[j][1]]))
            normalf[count] = np.divide(normalf[count], np.linalg.norm(normalf[count]))
            count += 1

        fn = normalf

        # compete the formal tensor voting of vertex v
        for j in range(len(f)):
            weigth = (np.divide(area_face[j], Max_area)) * (
                np.exp(np.negative(np.divide(np.subtract(np.linalg.norm(Center[j]), v), thta))))
            T = T + np.multiply(np.multiply(weigth, fn[j]), fn[j])

        # apply eig analysis
        Vec, L = np.linalg.eig(T)
        Li = np.diag(L)

        e1 = Vec[0]
        e2 = Vec[1]
        e3 = Vec[2]

        # Normalise a eigen values
        Li = np.divide(Li, np.linalg.norm(Li))

        EVEN[i] = Li
        EVTO[i] = e1 + e2 + e3

    # iterative adjust parameters to get pleasant result
    while True:

        # initial feature classification
        PRIN = np.zeros([len(V), len(V)])
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
                    #PRIN[i] = np.cross(EVTO[i][3:5])
            else:
                Corner_v.append(i)

        # show mesh and feature vertex
        model3D.show_feature_vertex(V,F, Sharp_edge_v,Corner_v)
        Door = input('Normal Tensor Voting -- The result is OK?  Input: y or n:--')
        if Door.lower() == 'y':
            break
        else:
            print('The previous alpha is %f: ', alpha)
            alpha = input('Please input a new alpha: ')
            print('The previous alpha is %f: ', beta)
            beta = input('Please input a new alpha: ')

    PRIN = []
    return Sharp_edge_v, Corner_v, EVEN, PRIN