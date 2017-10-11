from __future__ import division
import numpy as np
from pylab import scatter,imshow,show,plot,xlabel,ylabel,xlim,ylim,suptitle

def leastsquares(x,y):
  N = len(x)                                                              
  Ex = np.sum(x)/N                                                       
  Ey = np.sum(y)/N
  Exx = np.sum(x*x)/N
  Exy = np.sum(x*y)/N
  m = (Exy-Ex*Ey)/(Exx-Ex*Ex)                                             
  c = (Exx*Ey-Ex*Exy)/(Exx-Ex*Ex)
  return m,c

