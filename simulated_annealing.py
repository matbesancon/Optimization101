# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:33:38 2016

@author: MATHIEU
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D


def rastringin(x):
    """
    defines the n-dimensional Rastringin function
    """
    A = 10
    s = A*len(x)+(np.power(x,2)-A*np.cos(2*np.pi*x)).sum()
    return s
    

def rastrin_grad(x):
    """
    returns the exact gradient of the rastringin function
    """
    return 2*x+2*np.pi*np.sin(2*np.pi*x)

x = np.array([0,0])


# Objective function visualization

xmesh, ymesh = np.meshgrid(np.arange(-5,5,.1),\
    np.arange(-5,5,.1))

pmap = np.array([xmesh.ravel(),ymesh.ravel()]).T
p = np.array([])
for line in pmap:
    p = np.append(p,(rastringin(line)))
    
p = p.reshape(xmesh.shape)

f1 = plt.figure()
ax1 = f1.gca()
cp = ax1.contourf(xmesh, ymesh, p, cmap= matplotlib.cm.seismic_r,alpha=1)
plt.colorbar(cp,shrink=.5,aspect=5)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
s = ax.plot_surface(xmesh, ymesh, p, rstride=4, cstride=4, cmap = matplotlib.cm.seismic_r,\
    linewidth=0,alpha=.5)
fig.colorbar(s, shrink=0.5, aspect=5)
plt.show()


def fixed_step(x0,alpha):
    """
    iterates the fixed step gradient descent
    from a starting point and with a step size
    returns the number of iterations, 
    the final value of the function and of the gradient,
    """
    x1 = x0+1
    niter = 0
    while niter<1000 and np.sum(np.abs(rastrin_grad(x0)))>10**(-5):
        x1 = x0        
        x0 = x0 - alpha*rastrin_grad(x0)
        niter+=1
    return niter,x0,rastringin(x0),rastrin_grad(x0)        

fvalue = []
niter = []
alpha = []
xn = []
for al in np.arange(0.01,0.8,0.01):
    for x in np.random.randn(100,2)*5:
        alpha.append(al)
        a, b, c, d = fixed_step(x,al)
        fvalue.append(c)
        niter.append(a)
        xn.append(np.power(x,2).sum())
        

def temperature(n, t0=50, cooling_rate=.95):
    """
    coefficient determining the probability of acceptance
    for a move highering the objective function
    """
    return t0*cooling_rate**n
    
def acceptance(x0,x1,t):
    """
    returns the probability of accepting
    a move, given the initial, final values
    and the current temperature
    """
    return min(1,np.exp(-(rastringin(x1)-rastringin(x0))/t))
    
def candidate(x0,n,slowing_rate=.99):
    """
    takes a starting point and generates
    a new candidate from the specific distribution
    """
    theta = np.random.rand(1)[0]*2*np.pi
    rad = np.random.rand(1)[0]*3*(slowing_rate)**n
    return np.array([rad*np.cos(theta),
                     rad*np.sin(theta)])

def sa_iterate(t0,cooling_rate,slowing_rate,n,x0):
    """
    Given all simulation parameters and initial
    position, returns a final position
    """
    x1 = candidate(x0,n,slowing_rate)
    if acceptance(x0,x1,temperature(n,t0,cooling_rate))>np.random.rand():
        return x1
    else:
        return x0
    
def sa_global(t0,cooling_rate,slowing_rate,x0=np.random.rand(2,)):
    """
    iterates until convergence or too many iterations
    returns the path taken during the exploration
    and number of iterations
    """
    x1 = x0
    niter = 0
    path = []
    converge = False
    while niter < 5000 and converge == False:
        path.append(x0)
        if len(path)>=5:
            merror = 0
            for i in range(-5,-1):
                merror+=sum((path[i]-path[i+1])**2)
            converge = (merror < 10**(-9))
        if not converge:
            x1 = x0
            while (x1==x0).sum()/x0.size == 1:
                x1 = sa_iterate(t0,cooling_rate,slowing_rate,niter,x0)
            niter+=1
            x0 = x1
    return path, niter


path, niter = sa_global(10,.2,.6,np.random.rand(2,)*5)
path = np.array(path)

# Algorithm exploration visualization
xmesh, ymesh = np.meshgrid(np.arange(-3,3,.1),\
    np.arange(-3,3,.1))

pmap = np.array([xmesh.ravel(),ymesh.ravel()]).T
p = np.array([])
for line in pmap:
    p = np.append(p,(rastringin(line)))
    
p = p.reshape(xmesh.shape)

f1 = plt.figure()
ax1 = f1.gca()
cp = ax1.contourf(xmesh, ymesh, p, cmap= matplotlib.cm.seismic_r,alpha=1)
plt.colorbar(cp,shrink=.5,aspect=5)
plt.plot(path[:,0],path[:,1],'g')
plt.show()

plt.grid()
plt.plot(path[range(niter-70,niter),0],path[range(niter-70,niter),1])
