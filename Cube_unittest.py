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

from baseline import Translation as T, Corner, Edge, Sigma, Permutations as Perm
from Rubik import RubiksCube, RubiksGroup
import numpy as np
import unittest
from copy import copy



## Class test to evaluate the correctness of Exponential properties ##
class TestExponential(unittest.TestCase):
    def setUp(self):
        self.a = T()  ## should be a (0,0,0) Exponential class element ##
        self.b = T(1) ## should be a (1,0,0) Exponential class element ##
        self.c = T(1,-2) ## should be a (1,-2,0) Exponential class element ##
        self.m = T(x=1, y=2, z=3) ## should be a (1,2,3) Exponential class element ##
        self.l = T(z=-1) ## should be a (0,0,-1) Exponential class element ##

    def test_correct_default_initialization(self):
        ## self.a values should be initialized to zero ##
        self.assertEqual(self.a.x,0)
        self.assertEqual(self.a.y,0)
        self.assertEqual(self.a.z,0)
        ## j and k values should be initialized to zero for self.b ##
        self.assertEqual(self.b.y,0)
        self.assertEqual(self.b.z,0)
        ## i value for self.b has to be non-zero ##
        self.assertNotEqual(self.b.x,0)
        ## i and j values are non-zero for self.c ##
        self.assertEqual(self.c.x,1)
        self.assertEqual(self.c.y,-2)
        ## k value is null for self.c ##
        self.assertEqual(self.c.z,0)
        ## check self.m components to be (1,2,3) ##
        self.assertEqual(self.m.x, 1)
        self.assertEqual(self.m.y, 2)
        self.assertEqual(self.m.z, 3)
        ## check self.l components to be (0,0,-1) ##
        self.assertEqual(self.l.x, 0)
        self.assertEqual(self.l.y, 0)
        self.assertEqual(self.l.z, -1)


    def test_sum(self):
        c = T(1, 2, 3)
        d = T(2, 3, 1)
        self.assertEqual( (self.b*self.a).x, 1)
        self.assertEqual( (c*d).x, 3)
        self.assertEqual( (c*d).y, 5)
        self.assertEqual( (c*d).z, 4)

    def test_equals(self):
        h = T()
        ## as h and self.a share the same instance attributes (self.i=0,self.j=0,self.k=0), they should be equal ##
        self.assertEqual(self.a,h)
        ## try with self.b ##
        k = T(1)
        self.assertEqual(self.b,k)


## Check correctness of multiplication between matrices of orientation ##
## (as well as some properties) ##
class TestMatMultiplication(unittest.TestCase):
    def setUp(self):
        self.sX = Sigma.X()
        self.sA = Sigma.A()
        self.sC = Sigma.C()

    def test_sigmaX(self):
        mat = self.sX @ self.sX
        self.assertTrue( np.array_equal(mat.matrix, np.eye(2)) )

    def test_2sigmaA(self):
        mat = self.sA @ self.sA
        self.assertTrue( np.array_equal(mat.matrix, self.sC.matrix ) )

    def test_2sigmaC(self):
        mat = self.sC @ self.sC
        self.assertTrue( np.array_equal(mat.matrix, self.sA.matrix ) )

    def test_sigmaA_sigmaC(self):
        mat = self.sA @ self.sC
        self.assertTrue( np.array_equal(mat.matrix, np.eye(3)) )

    def test_sigmaC_sigmaA(self):
        mat = self.sC @ self.sA
        self.assertTrue( np.array_equal(mat.matrix, np.eye(3)) )

    def test_periodA(self):
        mat = self.sA @ self.sA @ self.sA
        self.assertTrue( np.array_equal(mat.matrix, np.eye(3)) )

    def test_periodC(self):
        mat = self.sC @ self.sC @ self.sC
        self.assertTrue( np.array_equal(mat.matrix, np.eye(3)) )



## Class test for the Edge orientation ##
class TestEdges(unittest.TestCase):
    def setUp(self):
        self.E = Edge(x=1,y=2,z=3)

    def test_orientationError(self):
        ## (x, y and z parameters are by default set null) ##
        with self.assertRaises(TypeError):
            ## [2,0] is not a corner state of orientation ##
            Corner(vector=np.array([2, 0]))
        with self.assertRaises(TypeError):
            ## [0.5,0.5] is not a corner state of orientation ##
            Corner(vector=np.array([.5, .5]))
        with self.assertRaises(TypeError):
            ## [0,0] is not a corner state of orientation ##
            Corner(vector=np.array([0, 0]))

    def test_flip(self):
        sx = Sigma.X()
        ## check if the matrix product returns a flipped vector ##
        self.assertTrue( np.array_equal((sx * self.E).orientation, np.array([0.,1.])) )
        ## check if two flips restore the correct position ##
        ## modify self.E orientation                       ##
        self.E = sx * (self.E)
        ## re-flip self.E orientation in the assert function ##
        self.assertTrue(np.array_equal((sx * self.E).orientation, np.array([1.,0.])))

    def test_expMultiplication(self):
        e = T(x=1, y=2, z=1)
        items = [2,4,4]
        for i,el in enumerate( (e * self.E).__dict__.items() ):
            self.assertEqual( el[1], items[i] )
            if i==2: break


## Class test for the Corner orientation ##
class TestCorners(unittest.TestCase):
    def setUp(self):
        self.C = Corner(x=1,y=2,z=3)

    def test_orientationError(self):
        ## (x, y and z parameters are by default set null) ##
        with self.assertRaises(TypeError):
            ## [2,0,0] is not a corner state of orientation ##
            Corner(vector=np.array([2,0,0]))
        with self.assertRaises(TypeError):
            ## [1,1,0] is not a corner state of orientation ##
            Corner(vector=np.array([1,1,0]))
        with self.assertRaises(TypeError):
            ## [1,.5,0] is not a corner state of orientation ##
            Corner(vector=np.array([1,.5,0]))
        with self.assertRaises(TypeError):
            ## [0,0,0] is not a corner state of orientation ##
            Corner(vector=np.array([0,0,0]))

    def test_clockwise(self):
        sC = Sigma.C()
        self.assertTrue( np.array_equal((sC * self.C).orientation, np.array([0.,0.,1.])) )

    def test_anticlockwise(self):
        sA = Sigma.A()
        self.assertTrue( np.array_equal((sA * self.C).orientation, np.array([1.,0.,0.])) )

    def test_expMultiplication(self):
        e = T(x=1, y=2, z=1)
        items = [2,4,4]
        for i,el in enumerate( (e * self.C).__dict__.items() ):
            self.assertEqual( el[1], items[i] )
            if i==2: break


## CLASS TO TEST PERMUTATIONS ##
class TestPermutations(unittest.TestCase):
    def setUp(self):
        self.perm_base = Perm([0,1,2,3])
        self.perm_double = Perm([1,2,3,4], [2,1,4,3])
        self.vector = np.arange(10)

    ## default initialization of perm_base              ##
    ## self.vector should cycle on elements from 0 to 4 ##
    def test_default_initialization(self):
        v2 = self.perm_base * self.vector
        self.assertTrue( np.array_equal(v2, np.array([1,2,3,0,4,5,6,7,8,9]) ) )

    def test_double_cycle_initialization(self):
        v2 = self.perm_double * self.vector
        self.assertTrue( np.array_equal(v2, np.array([0,2,1,4,3,5,6,7,8,9])) )

    ## check correct composition of two permutations ##
    def test_composition(self):
        P = self.perm_base @ self.perm_double
        ## check the compositions for the cycle, confronting it with lists ##
        self.assertTrue( np.array_equal(P.cycle1, np.array([0,2,3,4]) ) )
        self.assertTrue( np.array_equal(P.cycle2, np.array([2,4,0,3]) ) )
        ## NOW: check the correctness of the permutations on a vector v ##
        ## define P_alt by constructor to be equal to P                 ##
        P_alt = Perm([0,2,3,4], [2,4,0,3])
        ## check on a concrete vector v ##
        v = np.arange(5)
        ## give a shallow copy of v ##
        v_ = copy(v)
        ## check P * v to be correct ##
        self.assertTrue( np.array_equal(P * v, np.array([2,1,4,0,3]) ) )
        ## check the action of P_alt ##
        self.assertTrue(np.array_equal(P_alt * v_, np.array([2, 1, 4, 0, 3])))
        ## NOW confront the action of P and P_alt ##
        self.assertTrue( np.array_equal( P_alt * v_, P * v ) )

    ## check other sets of compositions ##
    def test_composition2(self):
        P1 = Perm([2,4,7], [4,7,2])
        P2 = Perm([1,2,4,8], [8,1,2,4])
        P = P1 @ P2
        P_ = Perm([1,4,7,8], [8,7,1,4])
        self.assertTrue( P, P_ )
        v = np.array([0,3,4,5,6,7,8,1,2])
        v_ = copy(v)
        self.assertTrue( np.array_equal(P * v, np.array([0,2,4,5,1,7,8,3,6]) ))
        self.assertTrue(np.array_equal(P_ * v_, np.array([0, 2, 4, 5, 1, 7, 8, 3, 6])))

    ## check other sets of compositions ##
    def test_composition3(self):
        P1 = Perm([1,3,5,6], [5,1,6,3])
        P2 = Perm([1,2,4,5,6], [2,5,6,1,4])
        P = P1 @ P2
        P_ = Perm([2,3,4,5,6], [5,2,6,4,3])
        self.assertTrue(P, P_)
        v = np.array([0, 3, 4, 5, 6, 7, 8, 1, 2])
        v_ = copy(v)
        self.assertTrue(np.array_equal(P * v, np.array([0,3,7,4,8,6,5,1,2])))
        self.assertTrue(np.array_equal(P_ * v_, np.array([0,3,7,4,8,6,5,1,2])))

    ## first and second lists must have same length ##
    def test_exception1(self):
        with self.assertRaises(TypeError):
            Perm([1,2], [1,2,3])

    ## elements in the first list must be in the second list ##
    def test_exception2(self):
        with self.assertRaises(TypeError):
            Perm([1,2,4], [1,2,3])

    ## not integer elements are not allowed ##
    def test_exception3(self):
        with self.assertRaises(TypeError):
            Perm([1.5, 2, 4])



                                        ##################
                                        ## RUBIK'S CUBE ##
                                        ##################

## class to test the Cube is in solved state or not ##
class TestSolved(unittest.TestCase):
    def setUp(self):
        self.C = RubiksCube()

    def test_isSolved(self):
        ## default state is solved ##
        self.assertTrue(self.C.is_solved())
        ## applying an operator is solved no more ##
        RubiksGroup.U() * self.C
        self.assertFalse(self.C.is_solved())
        ## reset the state: is solved again ##
        self.C.reset()
        self.assertTrue(self.C.is_solved())


## class test to check correct position of cubies after transformations ##
class TestCorrectness(unittest.TestCase):
    def setUp(self):
        self.C = RubiksCube()

    def test_U(self):
        U = RubiksGroup.U()
        self.C = U * self.C
        self.assertEqual(self.C.Cube[4], Edge(x=+1,y=-1))
        self.assertEqual(self.C.Cube[5], Edge(x=-1,y=-1))
        self.assertEqual(self.C.Cube[6], Edge(x=-1,y=+1))
        self.assertEqual(self.C.Cube[7], Edge(x=+1,y=+1))
        self.assertEqual(self.C.Cube[16], Corner(x=+1))
        self.assertEqual(self.C.Cube[17], Corner(y=-1))
        self.assertEqual(self.C.Cube[18], Corner(x=-1))
        self.assertEqual(self.C.Cube[19], Corner(y=+1))
        self.C.reset()

    def test_D(self):
        D = RubiksGroup.D()
        self.C = D * self.C
        self.assertEqual(self.C.Cube[0], Edge(x=+1,y=-1))
        self.assertEqual(self.C.Cube[1], Edge(x=-1,y=-1))
        self.assertEqual(self.C.Cube[2], Edge(x=-1,y=+1))
        self.assertEqual(self.C.Cube[3], Edge(x=+1,y=+1))
        self.assertEqual(self.C.Cube[12], Corner(x=+1))
        self.assertEqual(self.C.Cube[13], Corner(y=-1))
        self.assertEqual(self.C.Cube[14], Corner(x=-1))
        self.assertEqual(self.C.Cube[15], Corner(y=+1))
        self.C.reset()

    def test_L(self):
        L = RubiksGroup.L()
        self.C = L * self.C
        self.assertEqual(self.C.Cube[1], Edge(y=-1, z=-1))
        self.assertEqual(self.C.Cube[8], Edge(y=-1, z=+1))
        self.assertEqual(self.C.Cube[5], Edge(y=+1, z=+1))
        self.assertEqual(self.C.Cube[9], Edge(y=+1, z=-1))
        self.assertEqual(self.C.Cube[14], Corner(z=-1, vector=np.array([0,0,1]) ))
        self.assertEqual(self.C.Cube[13], Corner(y=-1, vector=np.array([1,0,0])))
        self.assertEqual(self.C.Cube[17], Corner(z=+1, vector=np.array([0,0,1])))
        self.assertEqual(self.C.Cube[18], Corner(y=+1, vector=np.array([1,0,0])))
        self.C.reset()

    def test_R(self):
        R = RubiksGroup.R()
        self.C = R * self.C
        self.assertEqual(self.C.Cube[3], Edge(y=+1, z=-1))
        self.assertEqual(self.C.Cube[11], Edge(y=+1, z=+1))
        self.assertEqual(self.C.Cube[7], Edge(y=-1, z=+1))
        self.assertEqual(self.C.Cube[10], Edge(y=-1, z=-1))
        self.assertEqual(self.C.Cube[12], Corner(z=-1 ,vector=np.array([0,0,1])))
        self.assertEqual(self.C.Cube[15], Corner(y=+1, vector=np.array([1,0,0])))
        self.assertEqual(self.C.Cube[19], Corner(z=+1, vector=np.array([0,0,1])))
        self.assertEqual(self.C.Cube[16], Corner(y=-1, vector=np.array([1,0,0])))
        self.C.reset()

    def test_F(self):
        F = RubiksGroup.F()
        self.C = F * self.C
        self.assertEqual(self.C.Cube[0], Edge(x=+1, z=-1, vector=np.array([0,1])))
        self.assertEqual(self.C.Cube[10], Edge(x=+1, z=+1, vector=np.array([0,1])))
        self.assertEqual(self.C.Cube[4], Edge(x=-1, z=+1, vector=np.array([0,1])))
        self.assertEqual(self.C.Cube[8], Edge(x=-1, z=-1, vector=np.array([0,1])))
        self.assertEqual(self.C.Cube[12], Corner(x=+1, vector=np.array([1,0,0])))
        self.assertEqual(self.C.Cube[16], Corner(z=+1, vector=np.array([0,0,1])))
        self.assertEqual(self.C.Cube[17], Corner(x=-1, vector=np.array([1,0,0])))
        self.assertEqual(self.C.Cube[13], Corner(z=-1, vector=np.array([0,0,1])))
        self.C.reset()

    def test_B(self):
        B = RubiksGroup.B()
        self.C = B * self.C
        self.assertEqual(self.C.Cube[2], Edge(x=-1, z=-1, vector=np.array([0, 1])))
        self.assertEqual(self.C.Cube[11], Edge(x=+1, z=-1, vector=np.array([0, 1])))
        self.assertEqual(self.C.Cube[6], Edge(x=+1, z=+1, vector=np.array([0, 1])))
        self.assertEqual(self.C.Cube[9], Edge(x=-1, z=+1, vector=np.array([0, 1])))
        self.assertEqual(self.C.Cube[14], Corner(x=-1, vector=np.array([1,0,0])))
        self.assertEqual(self.C.Cube[15], Corner(z=-1, vector=np.array([0,0,1])))
        self.assertEqual(self.C.Cube[19], Corner(x=+1, vector=np.array([1,0,0])))
        self.assertEqual(self.C.Cube[18], Corner(z=+1, vector=np.array([0,0,1])))
        self.C.reset()




## Class test for the periodicity over 4 of the transformations ##
class TestPeriodicity(unittest.TestCase):
    def setUp(self):
        self.C = RubiksCube()

    def test_U(self):
        U = RubiksGroup.U()
        for _ in range(4): self.C = U * self.C
        self.assertTrue( self.C.is_solved() )
        self.C.reset()

    def test_D(self):
        D = RubiksGroup.D()
        for _ in range(4): self.C = D * self.C
        self.assertTrue(self.C.is_solved())
        self.C.reset()

    def test_L(self):
        L = RubiksGroup.L()
        for _ in range(4): self.C = L * self.C
        self.assertTrue(self.C.is_solved())
        self.C.reset()

    def test_R(self):
        R = RubiksGroup.R()
        for _ in range(4): self.C = R * self.C
        self.assertTrue(self.C.is_solved())
        self.C.reset()

    def test_F(self):
        F = RubiksGroup.F()
        for _ in range(4): self.C = F * self.C
        self.assertTrue(self.C.is_solved())
        self.C.reset()

    def test_B(self):
        B = RubiksGroup.B()
        for _ in range(4): self.C = B * self.C
        self.C.reset()


## class test for the composition of operators ##
class TestComposition(unittest.TestCase):
    def setUp(self):
        self.Ket = RubiksCube()
        self.F = RubiksGroup.F()
        self.L = RubiksGroup.L()
        self.U = RubiksGroup.U()
        self.D = RubiksGroup.D()
        self.R = RubiksGroup.R()
        self.B = RubiksGroup.B()

    def test_L2(self):
        L2 = self.L @ self.L
        for _ in range(2): self.L * self.Ket
        K2 = copy(self.Ket)
        self.Ket.reset()
        K1 = copy(L2 * self.Ket)
        self.Ket.reset()
        self.assertEqual(K1, K2)

    def test_L3(self):
        self.L = RubiksGroup.L()
        L3 = self.L @ self.L @ self.L
        K1 = copy(L3 * self.Ket)
        self.Ket.reset()
        for _ in range(3): self.Ket = self.L * self.Ket
        K2 = copy(self.Ket)
        self.Ket.reset()
        self.assertEqual(K1, K2)

    def test_F2(self):
        F2 = self.F @ self.F
        K1 = copy(F2 * self.Ket)
        self.Ket.reset()
        for _ in range(2): self.F * self.Ket
        K2 = copy(self.Ket)
        self.Ket.reset()
        self.assertEqual(K1, K2)

    def test_F3(self):
        F3 = self.F @ self.F @ self.F
        K1 = copy(F3 * self.Ket)
        self.Ket.reset()
        for _ in range(3): self.Ket = self.F * self.Ket
        K2 = copy(self.Ket)
        self.Ket.reset()
        self.assertEqual(K1, K2)

    def test_UFD(self):
        ## composed operator ##
        operators = [self.U, self.F, self.D]
        ## UFD is given by the composition below ##
        ## UFD = self.U @ self.F @ self.D        ##
        UFD = RubiksGroup.compose_multipleOperators(operators)
        K1 = copy(UFD * self.Ket)
        self.Ket.reset()
        ## apply single operators (from right to left) ##
        for O in reversed(operators): O * self.Ket
        K2 = copy(self.Ket)
        self.Ket.reset()
        self.assertEqual(K1, K2)

    def test_U2F3R3D(self):
        ## list of operators we are going to apply (from the right to the left) ##
        operators = [self.U, self.U, self.F, self.F, self.F, self.R, self.R, self.R, self.D]
        U2F3R4D = RubiksGroup.compose_multipleOperators(operators)
        U2F3R4D * self.Ket
        K1 = copy(self.Ket)
        self.Ket.reset()
        ## apply single operators (from right to left) ##
        for O in reversed(operators): O * self.Ket
        K2 = copy(self.Ket)
        self.Ket.reset()
        self.assertEqual(K1, K2)

    def test_D3F4R(self):
        ## list of operators we are going to apply (from the right to the left) ##
        operators = [self.D, self.D, self.D, self.F, self.F, self.F, self.F, self.R]
        D3F4R = RubiksGroup.compose_multipleOperators(operators)
        D3F4R * self.Ket
        K1 = copy(self.Ket)
        self.Ket.reset()
        ## apply single operators (from right to left) ##
        for O in reversed(operators): O * self.Ket
        K2 = copy(self.Ket)
        self.Ket.reset()
        self.assertEqual(K1, K2)




## class test for the periodicity of composed operators ##
class TestPeriodicComposition(unittest.TestCase):
    def setUp(self):
        self.Ket = RubiksCube()
        self.F = RubiksGroup.F()
        self.L = RubiksGroup.L()
        self.U = RubiksGroup.U()
        self.D = RubiksGroup.D()
        self.R = RubiksGroup.R()
        self.B = RubiksGroup.B()

    ## Apply an operator composed by 4 generators     ##
    ## Check the resulting state to be the solved one ##
    def test_L4(self):
        L4 = self.L @ self.L @ self.L @ self.L
        self.Ket = L4 * self.Ket
        self.assert_(self.Ket.is_solved())

    def test_R4(self):
        R4 = self.R @ self.R @ self.R @ self.R
        self.Ket = R4 * self.Ket
        self.assert_(self.Ket.is_solved())

    def test_U4(self):
        U4 = self.U @ self.U @ self.U @ self.U
        self.Ket = U4 * self.Ket
        self.assert_(self.Ket.is_solved())

    def test_D4(self):
        D4 = self.D @ self.D @ self.D @ self.D
        self.Ket = D4 * self.Ket
        self.assert_(self.Ket.is_solved())

    def test_F4(self):
        F4 = self.F @ self.F @ self.F @ self.F
        self.Ket = F4 * self.Ket
        self.assert_(self.Ket.is_solved())

    def test_B4(self):
        B4 = self.B @ self.B @ self.B @ self.B
        self.Ket = B4 * self.Ket
        self.assert_(self.Ket.is_solved())

    def test_L5(self):
        L5 = self.L @ self.L @ self.L @ self.L @ self.L
        K1 = copy(L5 * self.Ket)
        self.Ket.reset()
        K2 = copy(self.L * self.Ket)
        self.Ket.reset()
        self.assertEqual(K1,K2)







if __name__ == '__main__':
    unittest.main()
