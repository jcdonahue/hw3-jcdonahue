from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pickle
import eulerExact as eul
import leastsquares as ls
#import highorder as high

from highorder import apam, uinterpol, hLu, hgetDt, higherevolve

import onedim as one

a = 0
b = 1.0
cfl = 0.5
gamma = 1.4
x0 = .5
PL = 1
rhoL = 1 
rhoR = .1
PR = .125
vR = 0
vL = 0

def varsodshock(x,x0,N,PL,rhoL,vL,PR,rhoR,vR):
    gamma = 1.4
    momenR = rhoR*vR
    momenL = rhoL*vL
    ER = (PR/(gamma-1))+0.5*rhoR*np.power(vR, 2.0)
    EL = (PL/(gamma-1))+0.5*rhoL*vL*np.power(vL,2.0)
    u = np.zeros([N,3])
    for i in range(len(x)):
        if x[i] < x0:
            u[i,0] = rhoL
            u[i,1] = momenL
            u[i,2] = EL
        else:
            u[i,0] = rhoR
            u[i,1] = momenR
            u[i,2] = ER
    return u

ai = 0.0
bi = 2.0
x0i = 0.5
sigma = 0.4
alpha = 0.2
gammai = 5./3
rho0 = 1
P0 = 0.6
initial = eul.isentropicWave(ai, bi, 40, 0, x0i, sigma, alpha, gammai, rho0, P0,
                    TOL=1.0e-10)



def errorcomp(Pap,Pex,dx):
    integral = 0
    N = len(Pap)
    for i in range(N):
        integral += dx*np.fabs(Pap[i]-Pex[i])
    return integral

def specentr(P,rho,dx,P0,rho0):
    integral = 0
    N = len(P)
    for i in range(N):
        integral += dx*np.fabs(np.log((P[i]/P0)*np.power((rho[i]/rho0),-gammai))/(gammai-1.0))
    return integral


test = eul.isentropicWave(ai, bi, 100, 0, x0i, sigma, alpha, gammai, rho0, P0,
                    TOL=1.0e-10)
init = np.zeros([100,3])
for i in range(1,4):
    init[:,i-1]=test[i]
final = higherevolve(init,.01,.35)

def convergence(T,N,q,r):
    dx = (b-a)/N
    x = a + dx*(np.arange(N)+0.5)
    if q == 0:
        Xex, rhoex, vex, Pex  = eul.riemann(a, b, x0, N, T, rhoL, vL, PL, rhoR, vR, PR, gamma, 
                TOL=1.0e-14, MAX=100)
        init = varsodshock(x, x0, N, PL, rhoL, vL, PR, rhoR, vR)
    elif q == 1:
        Xex, rhoex, vex, Pex  = eul.isentropicWave(ai, bi, N, T, x0i, sigma, alpha, gammai, rho0, P0,
                    TOL=1.0e-10)
        listic = eul.isentropicWave(ai, bi, N, 0, x0i, sigma, alpha, gammai, rho0, P0,
                    TOL=1.0e-10)
        init = np.zeros([N,3])
        for i in range(1,4):
            init[:,i-1]=listic[i]
    if r == 0:
        u = one.evolve(init,dx,T)
    elif r == 1:
        u = higherevolve(init,dx,T)
    rhoap =  u[:,0]
    vap =  u[:,1]/u[:,0]
    rhov2 = .5*vap[:]*u[:,1]
    Pap = (gamma-1.0)*(u[:,2]-rhov2[:])
    error  = np.zeros(3)
    if q == 0:
        error[0] = errorcomp(rhoap,rhoex,dx)
        error[1] = errorcomp(vap,vex,dx)
        error[2] = errorcomp(Pap,Pex,dx)
    elif q ==1:
        error[:]= specentr(Pap,rhoap,dx,P0,rho0)
    # fig1,ax1 = plt.subplots(1,1)
    # ax1.plot(x,rhoap,'r-')
    # ax1.plot(x,rhoex,'b-')
    return error


def errorplot(T,N,q,r, filename=None):
    terms = len(N)
    a = np.zeros([terms, 3])
    fits = np.zeros([3,2])
    for i in range(terms):
        a[i] = convergence(T,N[i],q,r)
    lN = np.log10(N)
    la = np.log10(a)
    for i in range (3):
        least = ls.leastsquares(lN,la[:,i])
        fits[i,0] = least[0]
        fits[i,1] = least[1]
    if q == 0:
        fig1, ax1 = plt.subplots(3,1)
        ax1[0].plot(lN, la[:,0],'ko')
        ax1[0].plot(lN,fits[0,0]*lN+fits[0,1],'k-')
        ax1[0].set_title("t = "+str(T))
        ax1[0].set_ylabel(r'$\rho$ Log Error')
        ax1[1].plot(lN, la[:,1],'ko')
        ax1[1].plot(lN,fits[1,0]*lN+fits[1,1],'k-')
        ax1[1].set_ylabel(r'$V$ Log Error')
        ax1[2].plot(lN, la[:,2], 'ko')
        ax1[2].plot(lN,fits[2,0]*lN+fits[2,1],'k-')
        ax1[2].set_xlabel(r'Log $N$')
        ax1[2].set_ylabel(r'$P$ Log Error')
    elif q ==1:
        fig1,ax1 = plt.subplots(1,1)
        ax1.plot(np.log(N),np.log(a[:,0]),'ko')
        ax1.set_title("t = "+str(T))
        ax1.set_xlabel(r'Log $N$')              
        ax1.set_ylabel(r'$s$ Log Error')
        
    # if ax is None:
    #     fig, ax = plt.subplots(1,1)
    # else:
    #     fig = ax.get_figure()
        
    if filename is not None:
        fig.savefig(filename)
            
    return ax1



