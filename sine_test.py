"""
This is a simple example to illustrate how the solver works by tracking a
particle moving along the path
    y = sin(t)

For this example we use the least squares penalty
"""


import classes
import numpy as np
import Solver
import math
import matplotlib.pyplot as plt


n = 2
m = 1
N = 100
dt = .1
sigma = .1 # measurment noise
gamma = 1 # process noise


# Construct measurments and info for plotting later
true_obs = np.zeros(N)
meas = [None]*N
vel_true = np.zeros(N)
meas_vec = np.zeros(N)
x_axis = np.zeros(N)
for i in range(0,N):
    true_obs[i] = math.sin(i*dt)
    vel_true[i] = math.cos(i*dt)
    meas[i] = np.array([true_obs[i] + sigma*np.random.normal()])
    meas_vec[i] = meas[i]
    x_axis[i] = i*dt


# Create process and measurment model
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
x0 = np.array([1.,0.])


# create structures to pass into the solver
A,w,AAT = classes.Kalman_Setup(n,m,N,G,sqrtQ,H,sqrtR,meas,x0)


# Define parameters for the solver
max_iter = 100
# Create params class Add the functions later
params = classes.Kalman_params(N,n,m,A,AAT,0,0,0,0,0,w,meas,x0,1e-4,max_iter,0)
history = classes.Kalman_history([None]*max_iter,[None]*max_iter)

def l2s(x):
    return np.square(np.linalg.norm(x))


def prox_l2s(x,c):
    return (1/(1+c))*x

def id(x,c):
    return 1.*x


params.rho1 = l2s
params.rho2 = l2s
params.prox_rho1 = prox_l2s
params.prox_rho2 = prox_l2s
params.prox_rho3 = id


params.init = np.zeros(N*(2*n+m)) # Here just choosing to initialize at all zeros


# Call the solver
z,hist = Solver.solver(params,history)
x = classes.Kalman_extractor(z,N,n,m)


# Make plots
plt.figure(1)
plt.plot(x_axis,x[1,:],'b',label='Estimate')
plt.plot(x_axis,true_obs,'k',label='Truth')
plt.plot(x_axis,meas_vec,'r+',label='Observed Data')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Position Estimate')
plt.legend()

plt.figure(2)
plt.plot(x_axis,x[0,:],'b',label='Estimate')
plt.plot(x_axis,vel_true,'k',label='Truth')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Velocity Estimate')
plt.legend()

plt.show()

