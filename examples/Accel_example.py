"""
This is an example of robustly estimating the state (position, velocity, and acceleration)
based off of noisy acceleration data and infrequent position data

We track a particle following the path
        y = sin(t/2) + t/10
with acceleration measurments every .1 seconds
and addtional position measurments every 1 second
"""

import sys
sys.path.append('../src')


import math
import matplotlib.pyplot as plt
import numpy as np

import classes
import Solver


def main():
    n = 3   # size of state contains pos;vel;accel
    m = 2   # size of measurments
    N = 300 # number of time points
    dt = .1 # length of time interval
    sigma = 0.1 #accelration measurment noise
    sigma_pos = 0.1 # position measurment noise
    gamma = 1.  # process noise
    np.random.seed(25)

    #####################################
    # Generate Data
    ######################################
    true_pos = np.zeros(N)
    true_vel = np.zeros(N)
    true_accel = np.zeros(N)
    meas = [None]*N
    meas_pos = np.zeros(N)  # for plotting later
    meas_accel = np.zeros(N)# for plotting later
    x_axis = np.zeros(N)    # for plotting later

    for i in range(0,N):
        true_pos[i] = math.sin(dt*i/2)+ dt*i/10
        true_vel[i] = 1/2*math.cos(dt*i/2) + 1/10
        true_accel[i] = -1/4*math.sin(dt*i/2)
        meas_accel[i] = true_accel[i] + sigma*np.random.normal()
        #meas_pos[i] = true_pos[i] + sigma*np.random.normal()
        meas[i] = np.array([0, meas_accel[i]])
        #meas[i] = np.array([meas_pos[i],meas_accel[i]])
        x_axis[i] = i*dt


    ##################################
    # Create models
    ##################################

    Gamma_mat = np.array([1/6*dt*dt*dt,1/2*dt*dt,dt])


    G = [None]*N
    H = [None]*N
    sqrtQ = [None]*N
    sqrtR = [None]*N

    for i in range(0,N):
        G[i] = np.array([[1,dt,.5*dt*dt],[0,1,dt],[0,0,1]])
        H[i] = np.array([[0,0,0],[0,0,1.]])
        sqrtQ[i] = gamma*np.outer(Gamma_mat,Gamma_mat)
        sqrtR[i] = np.array([[.000001,0.],[0,sigma]])


    # Create initial state
    x0 = np.array([0,1/2+1/10,0])

    #####################################
    # Additional position measurments
    #####################################
    delta = 2 # for additional noise
    count = 0
    for i in range(0,N):
        if count == 2:
            H[i][0,:] = np.array([1.,0,0])
            sqrtR[i][0,:] = np.array([sigma_pos,0.])
            meas[i][0] = true_pos[i] + sigma_pos*np.random.normal()
            meas_pos[i] = meas[i][0]
            count = 0
            test = np.random.uniform()
            # add additional noise in position
            if test < 0.3:
                meas[i][0] = meas[i][0] + delta*np.random.normal()
                meas_pos[i] = meas[i][0]
        else:
            count += 1
            meas_pos[i] = None




    A, w, AAT = classes.Kalman_Setup(n, m, N, G, sqrtQ, H, sqrtR, meas, x0)

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
    tol = 1e-4
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

    #call the solver

    z_l2, history_l2 = Solver.solver(params_l2,history_l2)
    print("L2 Done!")
    z_robust, history_robust = Solver.solver(params_robust,history_robust)

    # extract state solution
    x_l2 = classes.Kalman_extractor(z_l2,N,n,m)
    x_robust = classes.Kalman_extractor(z_robust,N,n,m)


    ####################################
    # Make plots
    ####################################

    #Position plot
    plt.figure(1)
    plt.plot(x_axis,x_l2[0,:],'b-',label='L2 Estimate')
    plt.plot(x_axis,x_robust[0,:],'g',label='Robust Estimate')
    plt.plot(x_axis,true_pos,'k',label='Ground Truth')
    plt.plot(x_axis,meas_pos,'r+',markersize=2,label='Observed Data')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Robust vs Least Squares Position Estimates')
    plt.ylim((-2,5))
    plt.legend()
    plt.savefig('Pos_plot.pdf')


    # Velocity plot
    plt.figure(2)
    plt.plot(x_axis,x_l2[1,:],'b-',label='L2 Estimate')
    plt.plot(x_axis,x_robust[1,:],'g',label='Robust Estimate')
    plt.plot(x_axis,true_vel,'k',label='Ground Truth')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Robust vs Least Squares Velocity Estimates')
    plt.legend()
    plt.savefig('Vel_plot.pdf')


    # Acceleration plot
    plt.figure(3)
    plt.plot(x_axis,x_l2[2,:],'b-',label='L2 Esteimate')
    plt.plot(x_axis,x_robust[2,:],'g',label='Robust Estimate')
    plt.plot(x_axis,true_accel,'k',label='Ground Truth')
    plt.plot(x_axis,meas_accel,'r+',markersize=2,label='Observed Data')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Robust vs Least Squares Acceleration Estimates')
    plt.legend()
    plt.savefig('Accel_plot.pdf')


    plt.show()
main()



