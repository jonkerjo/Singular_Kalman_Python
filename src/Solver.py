"""
Solves the Kalman optimization problem
"""
import classes
import numpy as np

def prox_linear(z,tau,A,AAT,w,zeta):
    """
    solves a linear system
    :param z: primal variable
    :param tau: paramter
    :param A: block bi diag matrix
    :param AAT: A*A^T
    :param w: right hand side vector
    :param zeta: dual variable
    :return:
    """
    c = z-tau*zeta
    Ac = A.Blk_Bd_vm(c)
    v = -AAT.Blk_Td_s(w-Ac)
    AT = classes.Blk_Bd_Mat(A.DBlks, A.ODBlks, transposed=True)
    t = c- AT.Blk_Bd_vm(v)
    return t

def prox_dual(z,sigma,prox_primal):
    """
    :param z: where to evaluate
    :param sigma: parameter
    :param prox_primal:
    :return: the dual prox computed at z
    """
    t = z-sigma*prox_primal((1/sigma)*z,1/sigma)
    return t

def prox_vars(z,N,n,m):
    """
    creates vars u,t,x to apply the appropriate prox functions to
    :param z: interwoven primal variable (vector)
    :param N: number of time points
    :param n: size of state
    :param m: size of measurments
    :return: u,t,x
    """

    u = np.zeros(N*n)
    t = np.zeros(N*m)
    x = np.zeros(N*n)
    c = 2*n+m
    for i in range(0,N):
        u[i*n:(i+1)*n] = z[i*c:i*c+n]
        t[i*m:(i+1)*m] = z[i*c+n:i*c+n+m]
        x[i*n:(i+1)*n] = z[i*c+n+m:(i+1)*c]
    return u,t,x

def woven_var(u,t,x,N,n,m):
    """
    Creates interwoeven var z
    :param u: process var
    :param t: meas var
    :param x: state
    :param N: number of time points
    :param n: size of state
    :param m: size of meas
    :return: var z
    """

    c = 2*n+m
    z = np.zeros(N*(2*n+m))
    for i in range(0,N):
        z[i*c:i*c+n] = u[i*n:(i+1)*n]
        z[i*c+n:i*c+n+m] = t[i*m:(i+1)*m]
        z[i*c+n+m:(i+1)*c] = x[i*n:(i+1)*n]
    return z


def prox_sum(z,sigma,prox_rho1,prox_rho2,prox_rho3,m,n,N):
    """
    applies the prox to the correct components of z
    :param z: single vector primal variable
    :param sigma: prox constant
    :param prox_rho1: prox of rho1
    :param prox_rho2: prox of rho2
    :param prox_rho3: prox of rho3
    :param m: size of measurment
    :param n: size of state
    :param N: number of time points
    :return: application of prox
    """

    u,t,x = prox_vars(z,N,n,m)
    u = prox_rho1(u,sigma)
    t = prox_rho2(t,sigma)
    x = prox_rho3(x,sigma)
    return woven_var(u,t,x,N,n,m)


def solver(params,history):
    """
    Kalman Solver that uses DRS
    :param params: all kalman params
    :param history: for solver data
    :return: the estimated states and history
    """

    N = params.N
    n = params.n
    m = params.m
    A = params.A
    AAT = params.AAT
    prox_rho1 = params.prox_rho1
    rho1 = params.rho1
    prox_rho2 = params.prox_rho2
    rho2 = params.rho2
    prox_rho3 = params.prox_rho3
    w = params.w_hat
    #y = params.y
    #x0 = params.x0
    tol = params.tol
    max_iter = params.max_iter
    init = params.init
    history_res = history.res
    history_loss = history.loss

    z = init
    #zOld = np.zeros(N*(2*n+m))
    zeta = np.zeros(N*(2*n+m))
    tau = 0.9
    sigma = 0.9
    err = 1.
    iter = 0
    def prox_primal(var,const):
       return prox_sum(var,const,prox_rho1,prox_rho2,prox_rho3,m,n,N)

    while err > tol and iter < max_iter:
        iter += 1
        zOld = z
        z = prox_linear(z,tau,A,AAT,w,zeta)
        zeta = prox_dual(zeta+sigma*(2*z-zOld),sigma,prox_primal)
        err = 1/sigma*np.linalg.norm(z-zOld)
        u,t,x = prox_vars(z,N,n,m)
        value = rho1(u) + rho2(t)
        print("iter %3d, obj %7.2e, err %7.2e" % (iter, value,err))
        history_res[iter-1] = err
        history_loss[iter-1] = value

    return z,history



