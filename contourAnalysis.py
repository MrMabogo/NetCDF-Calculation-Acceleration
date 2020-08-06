from advection import WCOFS_RK2, geodist_to_xy
from wcofs_lonlat_2_xy import wcofs_lonlat_2_xy
import numpy as np
import math
from scipy.interpolate import RectBivariateSpline as rbv

from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from profiler import SProfiler
from numba import jit, float64, int32
import dask
import genfield_data as gd
import xarray as xr

tend = 24*7
R = 6.368e6

def calcCM(xs, ys):
    #simple mass with rho=1
    #mass = curve length = len(xs)
    mass = len(xs)
    
    #moments about each axis
    xM = sum(ys)
    yM = sum(xs)
    
    return yM/mass, xM/mass

@jit(nopython=True, cache=True)
def calcArea(lons, lats):
    '''return km^2 area'''
    area = 0
    lonsr = lons*np.pi/180
    latsr = lats*np.pi/180
    
    #line integral (area via Green's Theorem)
    for i in range(len(lons)):
        area +=(lonsr[i]-lonsr[0])*R*(latsr[(i+1)%len(latsr)]-latsr[i])*R*np.cos(latsr[i])
    return area/1000**2

@jit(nopython=True, cache=True)
def calcPeri(lons, lats):
    per = 0
    lonsr = lons*np.pi/180
    latsr = lats*np.pi/180
    
    for i in range(len(lons)):
        per += math.sqrt((R*(latsr[(i+1)%len(latsr)]-latsr[i]))**2+((lonsr[(i+1)%len(lonsr)]-lonsr[i])*R*np.cos(latsr[i]))**2)
        
    return per/1000

dprof = SProfiler()
grd = xr.open_dataset("grd_wcofs_large_visc200.nc")
dat = xr.open_mfdataset("zuvt_qck_Exp41_35*.nc", combine="by_coords", data_vars="minimal", chunks={"ocean_time":1})

rsl = slice(-1180, -400)
csl = slice(-690, -120)
tsl = slice(0, tend+1)

lon_u, lat_u = grd["lon_u"], grd["lat_u"]
lon_v, lat_v = grd["lon_v"].loc[rsl, csl], grd["lat_v"].loc[rsl, csl]
lonrs = grd.coords["lon_rho"].loc[rsl, csl]
latrs = grd.coords["lat_rho"].loc[rsl, csl]
xv, yv = wcofs_lonlat_2_xy(lon_v, lat_v, True)
rx, ry = wcofs_lonlat_2_xy(lonrs, latrs, True)

mu  = grd["mask_u"]
us = dat["u_sur"][tsl, rsl, csl]
vs = dat["v_sur"][tsl, rsl, csl]
dprof.add("loading")

xc = np.cos(xv*np.pi/180)
u = us/R #dx/dt
v = vs/R/xc #dy/dt

fig = plt.figure()
ax = fig.add_subplot(111)
dprof.add("setup")

u = u.fillna(0.0)
v = v.fillna(0.0)
dprof.add("fill-nans")

ax.quiver(lonrs, latrs, u[1], v[1], color="cyan", scale=1e-6, scale_units="xy")
ax.contour(lon_u, lat_u, mu, colors="red")
dprof.add("initial-plot")

colors = list(mcolors.CSS4_COLORS)[::-1]

lonInt = rbv(rx[0], ry[:, 0], lonrs.transpose(transpose_coords=False), kx=1, ky=1)
latInt = rbv(rx[0], ry[:, 0], latrs.transpose(transpose_coords=False), kx=1, ky=1)

cSet = xr.open_dataset("geostrophic_contour.nc")
x1 = cSet["T0"].loc["x"].values 
y1 = cSet["T0"].loc["y"].values
plons1 = lonInt.ev(x1, y1)
plats1 = latInt.ev(x1, y1)

cmx, cmy = calcCM(x1, y1)
cmlon = lonInt.ev(cmx, cmy)
cmlat = latInt.ev(cmx, cmy)
area1 = calcArea(plons1, plats1)
pmtr1 = calcPeri(plons1, plats1)
print("==Before==")
print(f"CM ({cmlon}, {cmlat})", flush=True)
print(f"Area {area1} km^2")
print(f"Perimeter {pmtr1} km")

x2 = cSet["TF"].loc["x"].values #[p.trajx[-1] for p in circle.particles]
y2 = cSet["TF"].loc["y"].values #[p.trajy[-1] for p in circle.particles]
plons = lonInt.ev(x2, y2)
plats = latInt.ev(x2, y2)

cmx, cmy = calcCM(x2, y2)
cmlon = lonInt.ev(cmx, cmy)
cmlat = latInt.ev(cmx, cmy)
area2 = calcArea(plons, plats)
pmtr2 = calcPeri(plons, plats)
print("==After==")
print(f"CM ({cmlon}, {cmlat})", flush=True)
print(f"Area {area2} km^2")
print(f"Perimeter {pmtr2} km")

print("==Change==")
print(f"Area/Area0 {area2/area1}")
print(f"Perimeter/Perimiter0 {pmtr2/pmtr1}\n")

pathVs = np.asarray([plons, plats])
pathVs = pathVs.T
pp = Path(pathVs, closed=True)
patch = PathPatch(pp, linewidth=0)
ax.add_artist(patch)
dprof.add("contour")

dprof.show()
cSet.close()
ax.set_aspect(1.3)
plt.show()
