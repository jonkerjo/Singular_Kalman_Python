"""
This is an example of robustly estimating the state
we will track a particle moving along the path given by
    y = sin(t) + t/10
where we can noisily measusre position at each time point

in addition at randomly selected points with frequency 1/20 we
will add additional (larger) noise to the measurments to make
the robust solver necessary
"""

import classes
import numpy as np
import Solver
import math
import matplotlib.pyplot as plt


n = 2
m = 1
N = 300
dt = .1
sigma = .1 # measurment noise
gamma = 1 # process noise

# construct meaurments and data for plotting later

true_obs = np.zeros(N)
meas = [None]*N
vel_true = np.zeros(N)
meas_vec = np.zeros(N)
x_axis = np.zeros(N)
for i in range(0,N):
    true_obs[i] = math.sin(i*dt)+i*dt/10
    vel_true[i] = math.cos(i*dt)+1/10
    meas[i] = np.array([true_obs[i] + sigma*np.random.normal()])
    meas_vec[i] = meas[i]
    x_axis[i] = i*dt

# now add the extra noise at random points
delta = 2. # scaling of extra noise
for i in range(0,N):
    test = np.random.uniform()
    if test < 0.05:
        meas[i] += delta*np.random.normal()
        meas_vec[i] = meas[i]


# create process and measurment models
Gblock = np.array([[1.,0.],[dt, 1]])
Hblock = np.array([[0.,1.]])
sqrtQblock = gamma*np.array([[dt, 0.],[.5*dt*dt, 0.]])
sqrtRblock = np.array([[sigma]])

G = [None]*N
H = [None]*N
sqrtQ = [None]*N
sqrtR = [None]*N

for i in range(0,N):
    G[i] = Gblock
    H[i] = Hblock
    sqrtQ[i] = sqrtQblock
    sqrtR[i] = sqrtRblock


# create initial state
x0 = np.array([2.,0.])

# create structres to pass into solver
A,w,AAT = classes.Kalman_Setup(n,m,N,G,sqrtQ,H,sqrtR,meas,x0)

# create parameters for solver, both for robust and normal

# first create functions and proxes to pass in
def l2s(x):
    return np.square(np.linalg.norm(x))


def prox_l2s(x,c):
    return (1/(1+c))*x

def id(x,c):
    return x

# need this for prox of Huber
def prox_l1(x,c):
    return np.maximum(np.absolute(x)-c*np.ones(len(x)),0)*np.sign(x)


kappa = 1. # paramter for Huber function
def Huber(x):
    mins = np.minimum(.5*np.square(x) - .5*kappa*kappa*np.ones(len(x)),0)
    maxs = np.maximum(kappa*(np.absolute(x)-kappa*np.ones(len(x))),0)
    return np.sum(mins+maxs + .5*kappa*kappa*np.ones(len(x)))

def prox_Huber(x,c):
    return kappa/(kappa+c)*x + c/(c+kappa)*prox_l1(x,c+kappa)


max_iter = 100
tol = 1e-5
params_l2 = classes.Kalman_params(N,n,m,A,AAT,0,0,0,0,0,w,meas,x0,tol,max_iter,np.zeros(N*(2*n+m)))
params_l2.rho1 = l2s
params_l2.rho2 = l2s
params_l2.prox_rho1 = prox_l2s
params_l2.prox_rho2 = prox_l2s
params_l2.prox_rho3 = id

params_robust = classes.Kalman_params(N,n,m,A,AAT,0,0,0,0,0,w,meas,x0,tol,max_iter,np.zeros(N*(2*n+m)))
params_robust.rho1 = Huber
params_robust.rho2 = Huber
params_robust.prox_rho1 = prox_Huber
params_robust.prox_rho2 = prox_Huber
params_robust.prox_rho3 = id


history_l2 = classes.Kalman_history([None]*max_iter,[None]*max_iter)
history_robust = classes.Kalman_history([None]*max_iter,[None]*max_iter)

# call the solver

z_l2, history_l2 = Solver.solver(params_l2,history_l2)
print("L2 Done!")
z_robust, history_robust = Solver.solver(params_robust,history_robust)

# extract state solution
x_l2 = classes.Kalman_extractor(z_l2,N,n,m)
x_robust = classes.Kalman_extractor(z_robust,N,n,m)


# make plots

# Position plot
plt.figure(1)
plt.plot(x_axis,x_l2[1,:],'b-',label='L2 Estimate')
plt.plot(x_axis,x_robust[1,:],'g',label='Robust Estimate')
plt.plot(x_axis,true_obs,'k',label='Ground Truth')
plt.plot(x_axis,meas_vec,'r+',markersize=2,label='Observed Data')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Robust vs Least Squares Position Estimates')
plt.legend()

# Velocity plot
plt.figure(2)
plt.plot(x_axis,x_l2[0,:],'b-',label='L2 Estimate')
plt.plot(x_axis,x_robust[0,:],'g',label='Robust Estimate')
plt.plot(x_axis,vel_true,'k',label='Ground Truth')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Robust vs Least Squares Velocity Estimates')
plt.legend()

plt.show()