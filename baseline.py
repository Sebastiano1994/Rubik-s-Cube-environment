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

import numpy as np
from numpy import array
from itertools import permutations as p


## Exponential class defines the position of the cubies ##
class Exponential:
    def __init__(self, x=0, y=0, z=0):
        self.x=x
        self.y=y
        self.z=z

    ## multiplicative operation between exponentials objects ##
    def __mul__(self, other):
        return Exponential(self.x+other.x, self.y+other.y, self.z+other.z)

    ## print method for exponentials ##
    def __repr__(self):
        return "(%s, %s, %s)" % (self.x, self.y, self.z)

    def __eq__(self, other):
        ## exponentials are equal when all instance attributes (i.e. self.x,self.y,self.z) match ##
        return all(e1 == e2 for e1, e2 in zip(self.__dict__.values(), other.__dict__.values()))



## Classes for cubies ##
class Cubie(Exponential):
    def __init__(self, vector=None, x=0, y=0, z=0):
        super().__init__(x, y, z)
        self.orientation = vector

    def __mul__(self, other):
        return self.__class__(self.x + other.x, self.y + other.y, self.z + other.z, self.orientation)

    def __repr__(self):
        return "(%s, %s, %s, %s)" % (self.x, self.y, self.z, self.orientation)

    def __eq__(self, other):
        ## Cubies are equal when all instance attributes (i.e. self.x,self.y,self.z and self.orientation) match ##
        return list(self.__dict__.values())[:3] == list(other.__dict__.values())[:3] and np.array_equal(self.orientation, other.orientation)
        #return self.__dict__.values() == other.__dict__.values()


## Define class for corner cubies ##
class Corner(Cubie):
    def __init__(self, x=0, y=0, z=0, vector=np.array([0.,1.,0.])):
        ## the vector param must be a corner state of orientation               ##
        ## [0,1,0], [1,0,0], [0,0,1] numpy arrays are the three possible states ##
        if not tuple(vector) in list(p([0,1,0])):
            raise TypeError(f"Vector {vector} does not match any corner state of orientation")
        super().__init__(vector, x,y,z)


## Define class for edge cubies ##
class Edge(Cubie):
    def __init__(self, x=0, y=0, z=0, vector=np.array([1.,0.])):
        ## the vector param must be an edge state of orientation   ##
        ## [0,1], [1,0] numpy arrays are the three possible states ##
        if not tuple(vector) in list(p([1,0])):
            raise TypeError(f"Vector {vector} does not match any edge state of orientation")
        super().__init__(vector, x, y, z)



###############
## OPERATORS ##
###############
## Translation operators ##
class Translation(Exponential):
    def __init__(self,x=0,y=0,z=0):
        super().__init__(x,y,z)

    def __matmul__(self, cubie):
        return cubie.__class__(self.x+cubie.x, self.y+cubie.y, self.z+cubie.z, cubie.orientation)

## Rotation operators ##
class Sigma:
    def __init__(self, matrix):
        self.matrix = matrix

    def __mul__(self, cubie):
        ## return the same Cubie but with different orientation (@ stands for matmul operation) ##
        return cubie.__class__(cubie.x, cubie.y, cubie.z, self.matrix @ cubie.orientation )

    def __matmul__(self, other):
        ## composition between matrices ##
        return other.__class__(self.matrix@other.matrix)

    def __repr__(self):
        return str(self.matrix)

    ## Define @classmethod to build default constructors ##
    ## Flip matrix for edges ##
    @classmethod
    def X(cls):
        return cls(np.array([[0.,1.],[1.,0.]]) )

    ## Clockwise rotation matrix for corners ##
    @classmethod
    def C(cls):
        return cls(np.array([[0.,0.,1.], [1., 0.,0.], [0.,1.,0.]]))

    ## Anticlockwise rotation matrix for corners ##
    @classmethod
    def A(cls):
        return cls(np.array([[0.,1.,0.], [0., 0.,1.], [1.,0.,0.]]))


##################
## PERMUTATIONS ##
##################

class Permutations:
    def __init__(self, cycle, cycle_2=None):
        ## introduce two-cycle notation ##
        self.cycle1 = cycle
        if cycle_2 is None: self.cycle2 = np.roll(cycle,-1)
        else: self.cycle2 = cycle_2
        if len(self.cycle1)!=len(self.cycle2):
            raise TypeError(f"Permuting lists {self.cycle1} and {self.cycle2} must have the same length: {len(self.cycle1)} is not {len(self.cycle2)}")
        for elem in self.cycle1:
            if elem % 1 != 0: raise TypeError(f"{elem} is not an integer")
            if elem not in self.cycle2: raise TypeError(f"{elem} not in {self.cycle2}")

    ## convert the two cycles in a function f : cycle1[i] -> cycle2[i] ##
    ## such a function can be given by a dictionary                    ##
    def convert(self):
        ## first, decompose the permutation into single exchanges                                          ##
        ## such a purpose is given by splitting the two cycles in a nx2 matrix, each row being an exchange ##
        decoupling = np.array([self.cycle1, self.cycle2]).transpose()
        ## now return a dictionary ##
        return {x[0]: x[1] for x in decoupling}

    ## verify two permutations to be equivalent ##
    def __eq__(self, other):
        return np.array_equal(self.cycle1, other.cycle1) and np.array_equal(self.cycle2, other.cycle2)

    ## action on a vector ##
    def __mul__(self, vector):
        vector[self.cycle1] = vector[self.cycle2]
        return vector

    def __repr__(self):
        return str(self.cycle1) + '\n' + str(self.cycle2)

    def __matmul__(self, other):
        ###################################################
        ##    convert the exchanges into a dictionary    ##
        ## use the self.convert() function defined above ##
        ###################################################
        dic1 = self.convert()
        dic2 = other.convert()
        for key, value in dic1.items():
            try: dic1[key] = dic2[value]
            except: continue
        for key, value in dic2.items():
            if key not in dic1: dic1[key] = value
        for key in dic1.copy():
            if key == dic1[key]: dic1.pop(key)
        A = np.array(list(dic1.items())).transpose()
        try:
            return Permutations(A[0],A[1])
        ## when A == [] no elements are permuted              ##
        ## in such case, return a [0] list                    ##
        ## i.e. and identity permutation on the first element ##
        except:
            return Permutations([0])




















