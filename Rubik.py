###############################################################################
# The copyright of this code, including all portions, content, design, text,  #
# output and the selection and arrangement of the subroutines is owned by     #
# the Authors and by CNR, unless otherwise indicated, and is protected by the #
# provisions of the Italian Copyright law.                                    #
#                                                                             #
# All rights reserved. This software may not be reproduced or distributed, in #
# whole or in part, without the prior written permission of the Authors.      #
# However, reproduction and distribution, in whole or in part, by non-profit, #
# research or educational institutions for their own use is permitted if      #
# proper credit is given, with full citation, and copyright is acknowledged.  #
# Any other reproduction or distribution, in whatever form and by whatever    #
# media, is expressly prohibited without the prior written consent of the     #
# Authors. For further information, please contact CNR.                       #
# Contact person:           enrico.prati@cnr.it                               #
#                                                                             #
# Concept and development:  Sebastiano Corli, Lorenzo Moro, Enrico Prati      #
# Year:                     2022                                              #
# Istituto di Fotonica e Nanotecnologie - Consiglio Nazionale delle Ricerche  #
###############################################################################

from baseline import Translation as T, Permutations as Perm
from baseline import *
from functools import reduce
from copy import copy


###########################
## RUBIK'S CUBE (vector) ##
###########################
class RubiksCube:
    def __init__(self, state_vector=None):
        ## define the solved state of the Cube ##
        self.solved = np.concatenate((np.array([Edge() for _ in range(12)]), np.array([Corner() for _ in range(8)])),
                                     axis=0)
        ## allocate the actual state of the Cube ##
        self.Cube = state_vector if state_vector is not None else copy(self.solved)

    ## print function should return the self.Cube, i.e. the state vector ##
    def __repr__(self):
        return str(self.Cube).replace(') ', ')\n ').replace(' (', '(')[1:-1]

    def __eq__(self, other):
        return all(e1 == e2 for e1, e2 in zip(self.Cube, other.Cube))

    ## reset the Cube to its solved state ##
    def reset(self):
        self.Cube = copy(self.solved)

    ## function to determine whether the Cube is solved or not ##
    def is_solved(self):
        return all(self.Cube == self.solved)





###################
## RUBIK'S GROUP ##
###################
class RubiksGroup():
    def __init__(self, *args):
        ## EDGES ##
        ## The indices of the edges are given by args[0]       ##
        ## args[0] is the first list passed in the constructor ##
        ## translations: map from indices to exp operators ##
        self.edge_transl = {key : value for key, value in zip(args[0], args[1])}
        ## flips: map from indices to sigma_x matrices ##
        self.edge_flip = {key : value for key, value in zip(args[0], args[2])}
        ## edge permutations ##
        self.Pe = args[3]

        ## CORNERS ##
        ## The indices of the corners are given by args[4]     ##
        ## args[0] is the fifth list passed in the constructor ##
        ## translations: map from indices to exp operators ##
        self.corner_transl = {key : value for key, value in zip(args[4], args[5])}
        ## rotations: map from indices to sigma matrices ##
        self.corner_rot = {key : value for key, value in zip(args[4], args[6])}
        ## corner permutations ##
        self.Pc = args[7]


    ###################
    ## OPERATOR CORE ##
    ###################

    ###############################
    ## Action on the Cube vector ##
    ###############################
    def __mul__(self, Cube):
        ## single cubie transformations ##
        for te in self.edge_transl.items():
            Cube.Cube[te[0]] *= te[1]
        for fe in self.edge_flip.items():
            Cube.Cube[fe[0]] = fe[1] * Cube.Cube[fe[0]]
        for tc in self.corner_transl.items():
            Cube.Cube[tc[0]] *= tc[1]
            #print(Cube.Cube[tc[0]])
        for rc in self.corner_rot.items():
            Cube.Cube[rc[0]] = rc[1] * Cube.Cube[rc[0]]
        ## permutation of the cubies ##
        Cube.Cube = self.Pe * Cube.Cube
        Cube.Cube = self.Pc * Cube.Cube
        return Cube

    ##############################
    ## Composition of operators ##
    ##############################

    ## Useful functions     ##
    ## compose translations ##
    def compose_translations(self, dic1, dic2, permutation):
        ## newDic will be the dictionary of the composed operator ##
        newDic = {}
        ## permutations of the second operator must be applied on the first one ##
        ## the reason of such operation depends from basics of group theory     ##
        for key in [*dic1]:
            if key in permutation.cycle1: newDic[permutation.convert()[key]] = dic1[key]
        ## find common keys between dic2 and newDic ##
        common_elements = set(newDic.keys()) & set(dic2.keys())
        ## compose common elements in newDic ##
        for elem in common_elements: newDic[elem] *= dic2[elem]
        ## add the items from dic1 and dic2 whose keys are not in newDic ##
        newDic = dict(list(dic2.items()) + list(newDic.items()))
        newDic = dict(list(dic1.items()) + list(newDic.items()))
        ## return keys and values from newDic ##
        return [*newDic], [*newDic.values()]

    ## compose orientations ##
    def compose_orientations(self, dic1, dic2, permutation):
        newDic = {}
        for key in [*dic1]:
            if key in permutation.cycle1: newDic[permutation.convert()[key]] = dic1[key]
        common_elements = set(newDic.keys()) & set(dic2.keys())
        for elem in common_elements: newDic[elem] @= dic2[elem]
        newDic = dict(list(dic2.items()) + list(newDic.items()))
        newDic = dict(list(dic1.items()) + list(newDic.items()))
        return [*newDic.values()]

    ## Composition ##
    def __matmul__(self, other):
        ## COMPOSE PERMUTATIONS ##
        perm_e = self.Pe @ other.Pe
        perm_c = self.Pc @ other.Pc
        ## COMPOSE OPERATORS ##
        ## edge translations ##
        edges, edge_translations = self.compose_translations(self.edge_transl, other.edge_transl, other.Pe)
        ## corner translations ##
        corners, corner_translations = self.compose_translations(self.corner_transl, other.corner_transl, other.Pc)
        ## edge orientation ##
        edge_orientations = self.compose_orientations(self.edge_flip, other.edge_flip, other.Pe)
        ## corner orientation ##
        corner_orientations = self.compose_orientations(self.corner_rot, other.corner_rot, other.Pc)

        return RubiksGroup(edges, edge_translations, edge_orientations, perm_e, corners, corner_translations, corner_orientations, perm_c)


    ######################
    ## GROUP GENERATORS ##
    ######################

    @classmethod
    def U(cls):
        edges = [4,5,6,7]
        corners = [16, 17, 18, 19]
        return cls(edges,
                   [T(x=+1,y=+1), T(x=+1,y=-1), T(x=-1,y=-1), T(x=-1,y=+1)],
                   [Sigma(np.eye(2)), Sigma(np.eye(2)), Sigma(np.eye(2)), Sigma(np.eye(2))],
                   Perm(np.array(edges)),
                   #[16, 17, 18, 19],
                   corners,
                   [T(y=+1), T(x=+1), T(y=-1), T(x=-1)],
                   [Sigma(np.eye(3)), Sigma(np.eye(3)), Sigma(np.eye(3)), Sigma(np.eye(3))],
                   Perm(np.array(corners)))

    @classmethod
    def D(cls):
        edges = [0,1,2,3]
        corners = [12, 13, 14, 15]
        return cls(edges,
                   [T(x=+1,y=+1),T(x=+1,y=-1),T(x=-1,y=-1),T(x=-1,y=+1)],
                   [Sigma(np.eye(2)), Sigma(np.eye(2)), Sigma(np.eye(2)), Sigma(np.eye(2))],
                   Perm(edges),
                   corners,
                   [T(y=+1), T(x=1), T(y=-1), T(x=-1)],
                   [Sigma(np.eye(3)), Sigma(np.eye(3)), Sigma(np.eye(3)), Sigma(np.eye(3))],
                   Perm(corners))

    @classmethod
    def L(cls):
        edges = [1,9,5,8]
        corners = [14,18,17,13]
        return cls(edges,
                   [T(y=-1,z=+1), T(y=-1,z=-1), T(y=+1,z=-1), T(y=+1,z=+1)],
                   [Sigma(np.eye(2)), Sigma(np.eye(2)), Sigma(np.eye(2)), Sigma(np.eye(2))],
                   Perm(np.array(edges)),
                   corners,
                   [T(y=-1),T(z=-1),T(y=+1), T(z=+1)],
                   [Sigma.A(), Sigma.C(), Sigma.A(), Sigma.C()],
                   Perm(np.array(corners)))

    @classmethod
    def R(cls):
        edges = [3,10,7,11]
        corners = [12, 16, 19, 15]
        return cls(edges,
                   [T(y=+1,z=+1), T(y=+1,z=-1), T(y=-1,z=-1), T(y=-1,z=+1)],
                   [Sigma(np.eye(2)), Sigma(np.eye(2)), Sigma(np.eye(2)), Sigma(np.eye(2))],
                   Perm(edges),
                   corners,
                   [T(y=+1), T(z=-1), T(y=-1), T(z=+1)],
                   [Sigma.A(), Sigma.C(), Sigma.A(), Sigma.C()],
                   Perm(corners))

    @classmethod
    def F(cls):
        edges = [0,8,4,10]
        corners = [12, 13, 17, 16]
        return cls(edges,
                   [T(x=+1,z=+1), T(x=+1,z=-1), T(x=-1,z=-1), T(x=-1,z=+1)],
                   [Sigma.X(), Sigma.X(), Sigma.X(), Sigma.X()],
                   Perm(edges),
                   corners,
                   [T(z=+1), T(x=+1), T(z=-1), T(x=-1)],
                   [Sigma.C(), Sigma.A(), Sigma.C(), Sigma.A()],
                   Perm(corners))

    @classmethod
    def B(cls):
        edges = [2,11,6,9]
        corners = [14,15,19,18]
        return cls(edges,
                   [T(x=-1, z=+1), T(x=-1,z=-1), T(x=+1,z=-1), T(x=+1,z=+1)],
                   [Sigma.X(), Sigma.X(), Sigma.X(), Sigma.X()],
                   Perm(edges),
                   corners,
                   [T(z=+1), T(x=-1), T(z=-1), T(x=+1)],
                   [Sigma.C(), Sigma.A(), Sigma.C(), Sigma.A()],
                   Perm(corners))


    ########################
    ## COMPOSED OPERATORS ##
    ########################

    ## From a list of operators, this method returns their composition ##
    ## The argument can be any iterable (lists, tuples, sets etc...)   ##
    @classmethod
    def compose_multipleOperators(cls, operators_to_compose):
        return reduce(lambda x, y: x @ y, operators_to_compose)




