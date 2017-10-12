from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import eulerExact as eul


a = 0.0
b = 2.0
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

def higherevolve(u,dx,gamma,tfinal):
    t = 0.0
    while t < tfinal:
        # Get dt for this timestep.
        # Don't go past tfinal!
        uinter0  = uinterpol(u)
        alpha0 = apam(uinter0,gamma)
        dt = hgetDt(alpha0[2],alpha0[3],dx)
        if t + dt > tfinal:
            dt = tfinal - t
        
        #Calculate fluxes
        Lu0 = hLu(alpha0[0],alpha0[1],alpha0[2],alpha0[3],uinter0,dx,gamma)

        u1 = u + dt*Lu0
        uinter1 = uinterpol(u1)
        alpha1 = apam(uinter1,gamma)
        Lu1 = hLu(alpha1[0],alpha1[1],alpha1[2],alpha1[3],uinter1,dx,gamma)

        u2 = 0.75*u+0.25*u1+.25*dt*Lu1
        uinter2 = uinterpol(u2)
        alpha2 = apam(uinter2,gamma)
        Lu2 = hLu(alpha2[0],alpha2[1],alpha2[2],alpha2[3],uinter2,dx,gamma)

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

def apam(u,gamma):
    Nx = len(u)
    ap = np.zeros(Nx)
    am = np.zeros(Nx)
    v = np.zeros([Nx,2])
    rhov2 = np.zeros([Nx,2])
    cs = np.zeros([Nx,2])
    for i in range(2):
        v[:,i] = u[:,1,i]/u[:,0,i]
        rhov2[:,i] = .5*v[:,i]*u[:,1,i]
        cs[:,i] = gamma*(gamma-1.0)*(u[:,2,i]-rhov2[:,i])/u[:,0,i]
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

def hLu(v,rhov2,ap,am,u,dx,gamma):
    Nx = len(u)
            
    F = np.zeros([Nx,3,2])
    FL = np.zeros([Nx,3])
    FR = np.zeros([Nx,3])
    uL = np.zeros([Nx,3])
    uR = np.zeros([Nx,3])
    for i in range(2):
        F[:,0,i] = u[:,1,i]
        F[:,1,i] = (gamma-1.0)*u[:,2,i] + (3.0-gamma)*rhov2[:,i]
        F[:,2,i] =  v[:,i]*(gamma*u[:,2,i]-(gamma-1.0)*rhov2[:,i])
    for j in range(3):
        FL[:,j] = F[:,j,0]
        FR[:,j] = F[:,j,1]
        uL[:,j] = u[:,j,0]
        uR[:,j] = u[:,j,1]
    ap = ap[:,np.newaxis]
    am = am[:,np.newaxis]
    
    FHLL = (ap*FL + am*FR - ap*am*(uR-uL)) / (ap+am)

    LU = np.zeros([Nx+3,3])
    LU[2:-2] = -(FHLL[1:] - FHLL[:-1]) / dx

    return LU


def hplot(N,t,ax=None, filename=None):
    dx = (b-a)/N
    x = a + dx*(np.arange(N)+0.5)
    gamma = 5.0/3.0
    #u = sodshock(N,.5,1,1,0,.125,.1,0)
    init = eul.isentropicWave(0.0, 2.0, N, 0, 0.5, 0.4, 0.2, gamma, 1.0, 0.6,TOL=1.0e-10)
    exact = eul.isentropicWave(0.0, 2.0, N, t, 0.5, 0.4, 0.2, gamma, 1.0, 0.6,TOL=1.0e-10)
    u = np.zeros([N,3])
    u[:,0]=init[1]
    u[:,1]=init[1]*init[2]
    u[:,2]=0.5*init[1]*init[2]*init[2]+(init[3]/(gamma-1))
    u = higherevolve(u,dx,gamma,t)
    v = u[:,1]/u[:,0]
    rhov2 = .5*v[:]*u[:,1]
    P = (gamma-1.0)*(u[:,2]-rhov2[:])
    fig1, ax1 = plt.subplots(3,1)
    ax1[0].plot(x, u[:,0],'k--')
    ax1[0].plot(x, exact[1],'k-')
    ax1[0].set_title("t = "+str(t))
    ax1[0].set_ylabel(r'$rho$')
    ax1[1].plot(x, v,'k--')
    ax1[1].plot(x, exact[2],'k-')
    ax1[1].set_ylabel(r'$V$')
    ax1[2].plot(x, P, 'k--')
    ax1[2].plot(x, exact[3],'k-')
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

