import numpy
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interpn
import math

import matplotlib.pyplot as plt
from dask import delayed
import typing
import time
import datetime
from numba import prange, jit, guvectorize, float64, void, int32

class DataField:
    '''Vector field based on provided data'''
    
    def __init__(self, uX, uY, vX, vY, U, V, gridded = False):
        self.uVals = U
        self.vVals = V
        self.uGrid = dict()
        self.vGrid = dict()
        
        if gridded:
            self.uGrid["x"] = uX
            self.uGrid["y"] = uY
            self.vGrid["x"] = vX
            self.vGrid["y"] = vY
        else:
            self.uGrid["x"], self.uGrid["y"] = numpy.meshgrid(uX, uY)
            self.vGrid["x"], self.vGrid["y"] = numpy.meshGrid(vX, vY)
    
    def interpolate(self, comp, x, y, t=0):
        cInt = None
        if comp=="x" or comp=="u":
            cInt = RectBivariateSpline(self.uGrid["x"][0], self.uGrid["y"][:, 0], self.uVals[t].T, kx=1, ky=1)
        else:
            cInt = RectBivariateSpline(self.vGrid["x"][0], self.vGrid["y"][:, 0], self.vVals[t].T, kx=1, ky=1)
        #uv = (uInt.ev(x, y), vInt.ev(x, y))
        #print(f"({x}, {y})->({uv[0]}, {uv[1]})")

        return cInt.ev(x, y)
    
class Particle:
    '''Particle advected on a field'''    
    
    def __init__(self, x0, y0, func:callable, field):
        self.trajx = [x0]
        self.trajy = [y0]
        self.adFunc = func
        self.adField = field
        
    def advect(self, t0, tf, dt, path=True):
        t = t0
        cx = self.trajx[0]
        cy = self.trajy[0]
        while t < tf:
            cx, cy, t = self.adFunc(self.adField, cx, cy, t, dt)
            
            self.trajx.append(cx)
            self.trajy.append(cy)
            #print(self.trajx[0], self.trajy[0], cx, cy)
        if path:
            return self.trajx, self.trajy
        else:
            return self.trajx[-1], self.trajy[-1]

@jit(nopython=True, cache=True)
def uFunc(coords, sep):
    '''Update number of contour particles based on separation'''
    
    xs = coords[0]
    ys = coords[1]
    
    if len(xs) > 20000:
        return coords
    
    res = numpy.array([[xs[0]], [ys[0]]])
                
    for i in range(1, len(xs)):
        xdiff = xs[i] - xs[i-1]
        ydiff = ys[i] - ys[i-1]
        dist = math.sqrt(xdiff**2+ydiff**2)
        
        if dist > sep:
            res = numpy.append(res, numpy.array([[xs[i-1]+xdiff/2], [ys[i-1]+ydiff/2]]), 1)
        
        res = numpy.append(res, numpy.array([[xs[i]], [ys[i]]]), 1)
        
    #distance from 1st to last particle
    xdiff = xs[0] - xs[-1]
    ydiff = ys[0] - ys[-1]
    dist = math.sqrt(xdiff**2+ydiff**2)
        
    if dist > sep:
        res = numpy.append(res, numpy.array([[xs[i-1]+xdiff/2], [ys[i-1]+ydiff/2]]), 1)
        
    return res        

class ParticleContour:
    '''Circular contour defined by center and radius
    Boundary is defined by some number of particles'''
   
    def __init__(self, cx, cy, r, particles, func:callable, field, create=True):
        '''
        :param cx: center x coordinate
        :param cy: center y coordinate
        :param particles: number of particles
        :param func: advection function to pass to Particle
        :param field: vector field used for advection
        :param create: whether to create particles or use a list of particles'''
        self.adFunc = func
        self.adField = field
        self.particles = []
        
        if create == True:
            dtheta = 2*numpy.pi/particles #angle separation of particles in radians
            theta = 0
            self.separation = dtheta*r #angle separation -> distance separation (it is only a cirlce at the beginning)
            p0 = Particle(cx+r, cy, func, field) #1st particle at theta=0
            self.particles.append(p0)

            for i in range(particles-1):
                theta+=dtheta
                x = r*numpy.cos(theta)
                y = r*numpy.sin(theta)
                self.particles.append(Particle(cx+x, cy+y, func, field))
        else:
            self.particles = particles
            self.separation = math.sqrt((particles[1].trajx[-1] - particles[0].trajx[-1])**2+(particles[1].trajy[-1] - particles[0].trajy[-1])**2)
            
    @jit(forceobj=True)
    def advect(self, t0, tf, dt):
        '''advect the constituent particles
        Uncomment yield statement in order to use in animation'''
        
        t = t0
        xs = [p.trajx[-1] for p in self.particles]
        ys = [p.trajy[-1] for p in self.particles]
            
        while t < tf: 
            tf2 = t+1 if t+1 < tf else tf
            #print("hour", tf2, len(xs), flush=True)
            
            while t < tf2:
                xs, ys, t = self.adFunc(self.adField, xs, ys, t, dt)
                
            xs, ys = uFunc(numpy.asarray([xs, ys]), self.separation)
            #yield (xs, ys)
            
        parts = numpy.vectorize(Particle, excluded={2,3})(xs, ys, self.adFunc, self.adField)
        self.particles = parts