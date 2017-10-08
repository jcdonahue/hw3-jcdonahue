import math
import numpy as np
import matplotlib.pyplot as plt

def riemann(a, b, x0, N, T, rhoL, vL, PL, rhoR, vR, PR, gamma, 
                TOL=1.0e-14, MAX=100):
    # Returns the solution to the Riemann problem with left state (rhoL,vL,PL),
    # and right state (rhoR,vR,PR) after a time 'T' on a grid of N cells on
    # [a,b]. The initial discontinuity is placed at x0.
    #
    # Returns: X, rho, v, P

    AL = 2.0/((gamma+1.0)*rhoL)
    AR = 2.0/((gamma+1.0)*rhoR)
    BL = (gamma-1.0) / (gamma+1.0) * PL
    BR = (gamma-1.0) / (gamma+1.0) * PR
    csL = math.sqrt(gamma*PL/rhoL)
    csR = math.sqrt(gamma*PR/rhoR)

    p1 = 0.5*(PL + PR)
    p = p1
    i = 0
    dp = np.inf
    while abs(dp) > abs(p*TOL) and i < MAX:
        p = p1
        f1, df1 = riemann_f(p, rhoL, vL, PL, gamma, AL, BL, csL)
        f2, df2 = riemann_f(p, rhoR, vR, PR, gamma, AR, BR, csR)
        f = f1 + f2 + vR-vL
        df = df1 + df2

        dp = -f/df
        p1 = p + dp
        i += 1

    p = p1
    u = 0.5*(vL+vR) + 0.5*(riemann_f(p, rhoR, vR, PR, gamma, AR, BR, csR)[0]
                - riemann_f(p, rhoL, vL, PL, gamma, AL, BL, csL)[0])

    X = a + (b-a)/float(N) * (np.arange(N) + 0.5)
    xi = (X-x0)/float(T)

    rho = np.empty(X.shape)
    v = np.empty(X.shape)
    P = np.empty(X.shape)

    if p > PL:
        # Left Shock
        rhoLS = rhoL * (p/PL + (gamma-1.0)/(gamma+1.0)) / (
                (gamma-1.0)/(gamma+1.0) * p/PL + 1.0)
        SL = vL - csL*math.sqrt(((gamma+1) * p/PL + (gamma-1))/(2*gamma))

        iL = xi < SL
        iLS = (xi >= SL) * (xi < u)
        rho[iL] = rhoL
        v[iL] = vL
        P[iL] = PL
        rho[iLS] = rhoLS
        v[iLS] = u
        P[iLS] = p
    else:
        # Left Rarefaction
        rhoLS = rhoL * math.pow(p/PL, 1.0/gamma)
        csLS = csL * math.pow(p/PL, (gamma-1.0) / (2*gamma))
        SHL = vL - csL
        STL = u - csLS
        
        iL = xi < SHL
        ifan = (xi >= SHL) * (xi < STL)
        iLS = (xi >= STL)*(xi < u) 

        rho[iL] = rhoL
        v[iL] = vL
        P[iL] = PL
        rho[ifan] = rhoL * np.power(2.0/(gamma+1) + (gamma-1)/(gamma+1) 
                                    * (vL - xi[ifan]) / csL, 2.0/(gamma-1.0))
        v[ifan] = 2.0/(gamma+1) * (csL + 0.5*(gamma-1)*vL + xi[ifan])
        P[ifan] = PL * np.power(2.0/(gamma+1) + (gamma-1)/(gamma+1) 
                                * (vL - xi[ifan]) / csL, 2.0*gamma/(gamma-1.0))
        rho[iLS] = rhoLS
        v[iLS] = u
        P[iLS] = p

    if p > PR:
        # Right Shock
        rhoRS = rhoR * (p/PR + (gamma-1.0)/(gamma+1.0)) / (
                (gamma-1.0)/(gamma+1.0) * p/PR + 1.0)
        SR = vR + csR*math.sqrt(((gamma+1) * p/PR + (gamma-1))/(2*gamma))

        iR = xi >= SR
        iRS = (xi < SR) * (xi >= u)
        rho[iR] = rhoR
        v[iR] = vR
        P[iR] = PR
        rho[iRS] = rhoRS
        v[iRS] = u
        P[iRS] = p
    else:
        # Right Rarefaction
        rhoRS = rhoR * math.pow(p/PR, 1.0/gamma)
        csRS = csR * math.pow(p/PR, (gamma-1.0) / (2*gamma))
        SHR = vR + csR
        STR = u + csRS
        
        iR = xi >= SHR
        ifan = (xi < SHR) * (xi >= STR)
        iRS = (xi < STR)*(x >= u) 

        rho[iR] = rhoR
        v[iR] = vR
        P[iR] = PR
        rho[ifan] = rhoR * np.power(2.0/(gamma+1) - (gamma-1)/(gamma+1) 
                                    * (vR - xi[ifan]) / csR, 2.0/(gamma-1.0))
        v[ifan] = 2.0/(gamma+1) * (-csR + 0.5*(gamma-1)*vR + xi[ifan])
        P[ifan] = PR * np.power(2.0/(gamma+1) - (gamma-1)/(gamma+1) 
                                * (vR - xi[ifan]) / csR, 2.0*gamma/(gamma-1.0))
        rho[iRS] = rhoRS
        v[iRS] = u
        P[iRS] = p

    return X, rho, v, P

def isentropicWave(a, b, N, t, x0, sigma, alpha, gamma, rho0=1.0, P0=1.0,
                    TOL=1.0e-10):
    # Returns an isentropic wave, evolved for a time t on a grid of N cells on
    # [a,b]. The initial wave is centered at x0, has width sigma, and strength
    # alpha.
    #
    # Returns: X, rho, v, P

    dx = (b-a) / float(N)
    X = a + dx * (np.arange(N) + 0.5)

    def rhoProfile(x, x0, alpha, sigma, rho0):
        rho = np.empty(x.shape)
        rho[:] = rho0
        pulse = np.fabs(x-x0) < 1.0*sigma
        rho[pulse] += alpha*rho0*np.power(1-np.power((x[pulse]-x0)/sigma,2),2)
        return rho

    cs0 = math.sqrt(gamma*P0/float(rho0))

    rhoMax = rho0*(1.0+alpha)
    PMax = P0*math.pow(rhoMax/rho0, gamma)
    csMax = math.sqrt(gamma*PMax/rhoMax)
    vMax = 2 * (csMax-cs0) / (gamma-1.0)

    xL = X - (csMax + vMax)*t 
    xR = X - cs0*t

    while np.fabs(xR-xL).mean() > dx*TOL:
        x = 0.5*(xL + xR)
        rho = rhoProfile(x, x0, alpha, sigma, rho0)
        P = P0 * np.power(rho/rho0, gamma)
        cs = np.sqrt(gamma*P/rho)
        v = 2*(cs - cs0) / (gamma-1.0)
        xt = x + (v+cs)*t

        over = xt > X
        under = xt <= X
        xR[over] = x[over]
        xL[under] = x[under]

    x = 0.5*(xL + xR)
    rho = rhoProfile(x, x0, alpha, sigma, rho0)
    P = P0 * np.power(rho/rho0, gamma)
    cs = np.sqrt(gamma*P/rho)
    v = 2*(cs - cs0) / (gamma-1.0)

    return X, rho, v, P

def riemann_f(p, rho, v, P, gamma, A, B, cs):
    if p <= P:
        f = 2*cs*(math.pow(p/P,(gamma-1)/(2*gamma))-1.0) / (gamma-1.0)
        df = 2*cs*math.pow(p/P,-(gamma+1)/(2*gamma)) / (2*gamma*P)
    else:
        f = (p-P)*math.sqrt(A / (p+B))
        df = (1.0 - 0.5*(p-P)/(p+B)) * math.sqrt(A / (p+B))
    return f, df

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    a = -1.0
    b = 1.0
    x0 = 0.0
    N = 1000
    t = 0.25

    rhoL = 1.0
    PL = 1.0
    vL = 0.0
    rhoR = 0.1
    PR = 0.125
    vR = 0.0
    gamma = 1.4
    
    X, rho, v, P = riemann(a, b, x0, N, t, rhoL, vL, PL, rhoR, vR, PR, gamma)

    fig1, ax1 = plt.subplots(3,1)
    ax1[0].plot(X, rho)
    ax1[1].plot(X, v)
    ax1[2].plot(X, P)

    a = 0.0
    b = 2.0
    x0 = 0.5
    sigma = 0.3
    N = 1000
    t = 0.7

    rho0 = 1.0
    P0 = 1.0
    alpha = 0.1
    gamma = 5.0/3.0
    
    X, rho, v, P = isentropicWave(a, b, N, t, x0, sigma, alpha, gamma, 
                                    rho0, P0)
    fig2, ax2 = plt.subplots(3,1)
    ax2[0].plot(X, rho)
    ax2[1].plot(X, v)
    ax2[2].plot(X, P)

    plt.show()
