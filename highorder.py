from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import eulerExact as eul


a = 0.0
b = 1.0
cfl = 0.5


def minmod(x,y,z):
    return .25*np.fabs(np.sign(x)+np.sign(y))*(np.sign(x)+np.sign(z))*min(np.fabs(x),np.fabs(y),np.fabs(z))


def sodshock(Nx,x0,PL,rhoL,vL,PR,rhoR,vR):
    dx = (b-a)/Nx
    x = a + dx*(np.arange(Nx)+0.5)
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

def higherevolve(u,dx,tfinal):
    t = 0.0
    while t < tfinal:
        # Get dt for this timestep.
        # Don't go past tfinal!
        uinter0  = uinterpol(u)
        alpha = apam(uinter0)
        dt = hgetDt(alpha[2],alpha[3],dx)
        if t + dt > tfinal:
            dt = tfinal - t
        
        #Calculate fluxes
        Lu0 = hLu(alpha[0],alpha[1],alpha[2],alpha[3],uinter0,dx)
        u1 = u + dt*Lu0
        uinter1 = uinterpol(u1)
        alpha = apam(uinter1)
        Lu1 = hLu(alpha[0],alpha[1],alpha[2],alpha[3],uinter1,dx)
        u2 = 0.75*u+0.25*u1+.25*dt*Lu1
        uinter2 = uinterpol(u2)
        alpha = apam(uinter2)
        Lu2 = hLu(alpha[0],alpha[1],alpha[2],alpha[3],uinter2,dx)
        u = (u + 2.0*u2 + 2.0*dt*Lu2)/3.0
        t += dt
    return u

def uinterpol(u):
    Nx = len(u)
    uinter = np.zeros([Nx-3,3,2])
    for i in range(1,Nx-2):
        for j in range(3):
            uinter[i-1,j,0] = u[i,j] + 0.5*minmod(1.5*(u[i,j]-u[i-1,j]),0.5*(u[i+1,j]-u[i-1,j]),1.5*(u[i+1,j]-u[i,j]))
            uinter[i-1,j,1] = u[i+1,j] - 0.5*minmod(1.5*(u[i+1,j]-u[i,j]),0.5*(u[i+2,j]-u[i,j]),1.5*(u[i+2,j]-u[i+1,j]))
    return uinter

def apam(u):
    Nx = len(u)
    ap = np.zeros([Nx])
    am = np.zeros([Nx])
    gamma =1.4
    v = u[:,1,:]/u[:,0,:]
    rhov2 = .5*v[:,:]*u[:,1,:]
    cs = gamma*(gamma-1.0)*(u[:,2,:]-rhov2[:,:])/u[:,0,:]
    for i in range(Nx):
        for j in range(2):
            if cs[i,j]<0:
                cs[i,j] =0
    cs = np.sqrt(cs)
        
    for i in range(Nx):
            ap[i] = max(0, v[i,0]+cs[i,0],v[i,1]+cs[i,1])
            am[i] = max(0,-(v[i,0]-cs[i,0]),-(v[i,1]-cs[i,1]))
    return (v, rhov2, ap, am,cs)
            
def hgetDt(ap,am,dx):
    maxalphaplus = np.fabs(ap).max()
    maxalphamin = np.fabs(am).max()
    maxalpha =  max(maxalphamin,maxalphaplus)
    DT = cfl*dx/maxalpha
    return DT

def hLu(v,rhov2,ap,am,u,dx):
    gamma = 1.4
    Nx = len(u)
            
    F = np.zeros([Nx,3,2])
    F[:,0,:] = u[:,1,:]
    F[:,1,:] = (gamma-1.0)*u[:,2,:] + (3.0-gamma)*rhov2[:,:]
    F[:,2,:] =  v[:,:]*(gamma*u[:,2,:]-(gamma-1.0)*rhov2[:,:])
    FL = F[:,:,0]
    FR = F[:,:,1]
    uL = u[:,:,0]
    uR = u[:,:,1]
    ap = ap[:,np.newaxis]
    am = am[:,np.newaxis]
    
    FHLL = (ap*FL + am*FR - ap*am*(uR-uL)) / (ap+am)

    LU = np.zeros([Nx+3,3])
    LU[2:-2] = -(FHLL[1:] - FHLL[:-1]) / dx

    return LU


def hplot(N,t,ax=None, filename=None):
    dx = (b-a)/N
    x = a + dx*(np.arange(N)+0.5)
    gamma = 1.4
    #u = sodshock(N,.5,1,1,0,.125,.1,0)
    init = eul.isentropicWave(0.0, 2.0, N, 0, 1, 0.4, 0.2, 5.0/3.0, 1.0, 0.6,TOL=1.0e-10)
    u = np.zeros([N,3])
    for i in range(1,4):
        u[:,i-1]=init[i]
    u = higherevolve(u,dx,t)
    v = u[:,1]/u[:,0]
    rhov2 = .5*v[:]*u[:,1]
    P = (gamma-1.0)*(u[:,2]-rhov2[:])
    fig1, ax1 = plt.subplots(3,1)
    ax1[0].plot(x, u[:,0],'b-')
    ax1[0].set_title("t = "+str(t))
    ax1[0].set_ylabel(r'$rho$')
    ax1[1].plot(x, v,'b-')
    ax1[1].set_ylabel(r'$V$')
    ax1[2].plot(x, P, 'b-')
    ax1[2].set_xlabel(r'$X$')
    ax1[2].set_ylabel(r'$P$')
    # if ax is None:
    #     fig, ax = plt.subplots(1,1)
    # else:
    #     fig = ax.get_figure()
        
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

