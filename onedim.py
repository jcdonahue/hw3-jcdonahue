from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pickle

Nx = 2000
a = 0
b = 1.0
dx  = 0.1
cfl = 0.5
#dx = (b-a)/Nx
x = a + dx*(np.arange(Nx)+0.5)


def sodshock(x0,PL,rhoL,vL,PR,rhoR,vR):
    gamma = 1.4
    momenR = rhoR*vR
    momenL = rhoL*vL
    ER = (PR/(gamma-1))+0.5*rhoR*np.power(vR, 2.0)
    EL = (PL/(gamma-1))+0.5*rhoL*vL*np.power(vL,2.0)
    u = np.zeros([Nx,3])
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

def evolve(u,dx, tfinal):
    t = 0.0
    while t < tfinal:
        # Get dt for this timestep.
        # Don't go past tfinal!
        dt = getDt(u,dx)
        if t + dt > tfinal:
            dt = tfinal - t
        
        #Calculate fluxes
        udot = Lu(u,dx)
        
        #update u
        u[:] += dt*udot
        t += dt
    return u
            
def getDt(u,dx):
    Nx = len(u)
    ap = np.empty(Nx-1)
    am = np.empty(Nx-1)
    gamma = 1.4
    v = u[:,1]/u[:,0]
    rhov2 = .5*v[:]*u[:,1]
    cs = gamma*(gamma-1.0)*(u[:,2]-rhov2[:])/u[:,0]
    for i in range(Nx):
        if cs[i]<0:
            cs[i] =0
    cs = np.sqrt(cs)

    for i in range(Nx-1):
        ap[i] = max(0, v[i]+cs[i],v[i+1]+cs[i+1])
        am[i] = max(0,-(v[i]-cs[i]),-(v[i+1]-cs[i+1]))
         
    maxalphaplus = np.fabs(ap).max()
    maxalphamin = np.fabs(am).max()
    maxalpha =  max(maxalphamin,maxalphaplus)
    DT = cfl*dx/maxalpha
    return DT

def Lu(u,dx):
    gamma = 1.4
    Nx = len(u)
    ap = np.empty(Nx-1)
    am = np.empty(Nx-1)
    v = u[:,1]/u[:,0]
    rhov2 = .5*v[:]*u[:,1]
    cs = gamma*(gamma-1.0)*(u[:,2]-rhov2[:])/u[:,0]
    for i in range(Nx):
        if cs[i]<0:
            cs[i] =0
    cs = np.sqrt(cs)
        
    for i in range(Nx-1):
         ap[i] = max(0, v[i]+cs[i],v[i+1]+cs[i+1])
         am[i] = max(0,-(v[i]-cs[i]),-(v[i+1]-cs[i+1]))
            
    F = np.zeros([Nx,3])
    F[:,0] = u[:,1]
    F[:,1] = (gamma-1.0)*u[:,2] + (3.0-gamma)*rhov2[:]
    F[:,2] =  v[:]*(gamma*u[:,2]-(gamma-1.0)*rhov2[:])
    FL = F[:-1]
    FR = F[1:]
    uL = u[:-1]
    uR = u[1:]
    ap = ap[:,np.newaxis]
    am = am[:,np.newaxis]
    
    FHLL = (ap*FL + am*FR - ap*am*(uR-uL)) / (ap+am)

    LU = np.zeros([Nx,3])
    LU[1:-1] = -(FHLL[1:] - FHLL[:-1]) / dx

    return LU


def plot(t, x, ax=None, filename=None):
    gamma = 1.4
    u = sodshock(.5,1,1,0,.125,.1,0)
    u = evolve(u, t)
    v = u[:,1]/u[:,0]
    rhov2 = .5*v[:]*u[:,1]
    P = (gamma-1.0)*(u[:,2]-rhov2[:])
    if ax is None:
        fig, ax = plt.subplots(1,1)
    else:
        fig = ax.get_figure()
            
    ax.plot(x, u[:,0], '-')
    ax.set_xlabel(r'$X$')
    ax.set_ylabel(r'$U$')
    ax.set_title("t = "+str(t))
        
    if filename is not None:
        fig.savefig(filename)
            
    return ax
    

# def saveTxt(self, filename):
#         f = open(filename, "w")
        
#         f.write(str(self.Nx)+"\n")
#         f.write(str(self.a)+"\n")
#         f.write(str(self.b)+"\n")
#         f.write(str(self.t)+"\n")
#         f.write(str(self.cfl)+"\n")
#         f.write(" ".join([str(x) for x in self.x]) + "\n")
#         f.write(" ".join([str(u) for u in self.u]) + "\n")
    
#         f.close()
        
 # def loadTxt(self, filename):
 #        f = open(filename, "r")
        
 #        self.Nx = int(f.readline())
 #        self.a = float(f.readline())
 #        self.b = float(f.readline())
 #        self.t = float(f.readline())
 #        self.cfl = float(f.readline())
 #        self.dx = (self.b-self.a) / (float(self.Nx))
        
 #        x_str = f.readline()
 #        u_str = f.readline()
            
 #        f.close()
        
 #        self.x = np.array([float(x) for x in x_str.split()])
 #        self.u = np.array([float(u) for u in u_str.split()])
    
 # def savePickle(self, filename):
 #        f = open(filename, "w")
 #        pickle.dump(self, f, protocol=-1)
 #        f.close()
