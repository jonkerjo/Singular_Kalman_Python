"""
Creates block bi and tri diagonal matrix structures for Singular Kalman solver
"""

import numpy as np
from scipy.linalg import cholesky

class Blk_Bd_Mat:
    """
    Block bi diagonal matrix class
    """

    def __init__(self, DiagBlks, OffDBlks, transposed = False):
        self.DBlks = DiagBlks   # list of diag blocks
        self.ODBlks = OffDBlks  # list of off diag blocks
        self.T = transposed     # if lower block bi diag set false

    def Blk_Bd_T(self):
        """
        creates the transpose of a block bi diag matrix
        """
        temp = Blk_Bd_Mat(self.DBlks,self.ODBlks,transposed=self.T)
        if temp.T:
            temp.T = False
        else:
            temp.T = True
        return temp

    def Blk_Bd_vm(self,v):
        """
        Block bi diag vector multiplication
        :param v: vector to multiply by
        :return: product as a vector
        """

        D = self.DBlks
        O = self.ODBlks
        N = len(D)
        r = D[0].shape[0]
        c = D[0].shape[1]

        if self.T:
            x_out = np.zeros(N*c)
            if v.shape[0] != N*r:
                print("Vector is the wrong size")
                return None

            for i in range(0,N-1):
                x_out[i*c:(i+1)*c] = np.dot(np.transpose(D[i]),v[i*r:(i+1)*r])+np.dot(np.transpose(O[i]),v[(i+1)*r:(i+2)*r])

            x_out[(N-1)*c:N*c] = np.dot(np.transpose(D[N-1]),v[(N-1)*r:N*r])

        else:
            x_out = np.zeros(N*r)
            if v.shape[0] != N*c:
                print("Vector is the wrong size")
                return None

            x_out[0:r] = np.dot(D[0],v[0:c])

            for i in range(1,N):
                x_out[i*r:(i+1)*r] = np.dot(D[i],v[i*c:(i+1)*c]) + np.dot(O[i-1],v[(i-1)*c:i*c])

        return x_out

    def Blk_Bd_s(self,b):
        """
        Solves Ax = b. Assumes a solution exists
        :param b: vector
        :return: solution x
        """

        D = self.DBlks
        O = self.ODBlks
        N = len(D)

        r = D[0].shape[0]
        c = D[0].shape[1]

        if self.T:
            # need to solve from the bottom up
            x_out = np.zeros(N*r)
            if b.shape[0] != N*c:
                print("Vector is the wrong size")
                return None

            x_out[(N-1)*r:N*r] = np.linalg.lstsq(np.transpose(D[N-1]),b[(N-1)*c:N*c],rcond=None)[0]
            x_lag = x_out[(N-1)*r:N*r]
            for i in range(0,N-1):
                k = N-2-i # what we are iterating over
                x_out[k*r:(k+1)*r] = np.linalg.lstsq(np.transpose(D[k]),b[k*c:(k+1)*c] - np.dot(np.transpose(O[k]),x_lag),rcond=None)[0]
                x_lag = x_out[k*r:(k+1)*r]
        else:
            # now solve forward
            x_out = np.zeros(N*c)
            if b.shape[0] != N*r:
                print("Vector is the wrong size")
                return None
            x_out[0:c] = np.linalg.lstsq(D[0],b[0:r],rcond=None)[0]
            x_lag = x_out[0:c]
            for i in range(1,N):
                x_out[i*c:(i+1)*c] = np.linalg.lstsq(D[i],b[i*r:(i+1)*r] - np.dot(O[i-1],x_lag),rcond=None)[0]
                x_lag = x_out[i*c:(i+1)*c]

        return x_out

class Blk_Td_Mat:
    """
    Block tri diagonal matrix class
    stored as its choleksy factorization of type Blk_BD_Mat (with transposed=false)
    """

    def __init__(self,L):
        self.L = L
        if L.T:
            print("Transposed should be False")

    def Blk_Td_vm(self,v):
        """
        computes the vector product with v
        :param v: vector
        :return: output vector
        """
        L = self.L
        Lt = Blk_Bd_Mat(L.DBlks,L.ODBlks,transposed=True)
        temp = Lt.Blk_Bd_vm(v)
        out = L.Blk_Bd_vm(temp)
        return out

    def Blk_Td_s(self,b):
        """
        Solves Ax = b where A is block tri diagonal
        :param b: right hand side
        :return: solution x
        """

        L = self.L
        Lt = Blk_Bd_Mat(L.DBlks, L.ODBlks, transposed=True)
        temp = L.Blk_Bd_s(b)
        out = Lt.Blk_Bd_s(temp)
        return out

class Kalman_params:
    """
    parameters for kalman solver
    """

    def __init__(self,N,n,m,A,AAT,p_rho1,rho1,p_rho2,rho2,p_rho3,w_hat,y,x0,tol,max_i,init):
        """
        creates kalman parameters
        :param N: number of time points
        :param n: size of state
        :param m: size of measurments
        :param A: constraint matrix of type block bi diag
        :param AAT: chol fact of AAT
        :param p_rho1: prox of rho_1
        :param rho1: process PLQ penalty
        :param p_rho2: prox of rho_2
        :param rho2: measurment PLQ penalty
        :param p_rho3: optional box constrains on x, input projection onto desired box
        :param w_hat: output of kalman setup
        :param y: observed data (in a list)
        :param x0: initial state
        :param tol: solver tolerance
        :param max_i: max iterations
        :param init: initialization of primal variable in optimizatio problem
        """

        self.N = N
        self.n = n
        self.m = m
        self.A = A
        self.AAT = AAT
        self.prox_rho1 = p_rho1
        self.rho1 = rho1
        self.prox_rho2 = p_rho2
        self.rho2 = rho2
        self.prox_rho3 = p_rho3
        self.w_hat = w_hat
        self.y = y
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_i
        self.init = init

class Kalman_history:
    """
    For recording kalman solver history
    """

    def __init__(self,res,loss):
        """
        :param res: residual
        :param loss: objective values
        """

        self.res = res
        self.loss = loss

def Chol_fact_AAT(A):
    """
    Computes the cholesky factorization of AA^T
    Where A is a lower block bi-diagonal matrix
    :param A: lower block bi-diag matrix
    :return: returns L with AA^T = LL^T and L lower block bi-diag and square
    """

    D = A.DBlks
    O = A.ODBlks
    N = len(D)

    r = D[0].shape[0]

    s_lag = np.zeros((r,r))
    b_lag = np.zeros((r,r))

    # Create blocks for output
    D_out = [None]*N
    O_out = [None]*(N-1)

    for k in range(0,N):
        if k == N-1:
            b = np.zeros((r,r))
        else:
            b = np.dot(O[k],np.transpose(D[k]))

        if k == 0:
            a = np.dot(D[k],np.transpose(D[k]))
        else:
            a = np.dot(D[k],np.transpose(D[k])) + np.dot(O[k-1],np.transpose(O[k-1]))

        if k == 0:
            s = a
        else:
            s = a - np.dot(b_lag,np.linalg.lstsq(s_lag,np.transpose(b_lag),rcond=None)[0])

        c = cholesky(s,lower=True)
        d = np.transpose(np.linalg.lstsq(c,np.transpose(b),rcond=None)[0])

        D_out[k] = c
        if k == N-1:
            pass
        else:
            O_out[k] = d

        s_lag = s
        b_lag = b

    L = Blk_Bd_Mat(D_out,O_out)
    return Blk_Td_Mat(L)


def Kalman_Setup(n, m, N, G, sqrtQ, H, sqrtR, meas, x0):
    """
    Returns needed structures for kalman solver
    :param n: size of state space
    :param m: size of measurment space
    :param N: number of data points
    :param G: list of process matrices
    :param sqrtQ: list of sqrt of state covariance
    :param H: list of measurment matrices
    :param sqrtR: list of sqrt of measurment covariance
    :param meas: list of observed measurments
    :param x0: initial state
    :return: A, w_hat, AAT where A is blk-bi diag, w_hat is a vector
    """

    A_DBlks = [None]*N
    A_ODBlks = [None]*(N-1)

    r = n + m  # number of rows in a block
    c = 2*n + m # number of columns in a block

    w_hat = np.zeros(N*(n+m))
    w_hat[0:n] = x0
    for i in range(0,N):
        # create the diag block
        Drow1 = np.concatenate((sqrtQ[i],np.zeros((n,m)),np.identity(n)),axis=1)
        Drow2 = np.concatenate((np.zeros((m,n)),sqrtR[i], H[i]),axis=1)
        D = np.concatenate((Drow1,Drow2),axis=0)

        A_DBlks[i] = D
        if i != 0:
            w_hat[i*r:(i+1)*r] = np.concatenate((np.zeros(n),meas[i]),axis=0)
        if i != N-1:
            Brow1 = np.concatenate((np.zeros((n,r)),-G[i+1]),axis=1)
            Brow2 = np.zeros((m,c))
            B = np.concatenate((Brow1,Brow2),axis=0)
            A_ODBlks[i] = B

    A = Blk_Bd_Mat(A_DBlks,A_ODBlks)
    AAT = Chol_fact_AAT(A)
    return A, w_hat, AAT

def Kalman_extractor(z,N,n,m):
    """
    Extractes the states from solver solution
    :param z: solver solution
    :param N: number of time points
    :param n: size of state
    :param m: size of measurments
    :return: x a n x N matrix with states as columns
    """

    x = np.zeros((n,N))
    c = 2*n+m
    for i in range(0,N):
        x[:,i] = z[i*c+n+m:(i+1)*c]

    return x












