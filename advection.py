import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.colors as mcolors
import time
from profiler import SProfiler
import numpy as np
import math
from scipy.interpolate import RectBivariateSpline as rbv
from wcofs_lonlat_2_xy import wcofs_lonlat_2_xy
from numba import jit, float64, int32, void, vectorize
import genfield_data as gd
from dask import delayed
import dask
from dask import array as darr
import xarray as xr

R = 6.368e6 #earth radius (m)
tend = 24*10

def geodist_to_xy(x, lon0, lat0):
    '''convert geographical distance to distance on WCOFS grid'''
    #get lon and lat difference for x distance
    #dlat = x/R
    #dlon = x/R/np.cos(lat0*np.pi/180)
    
    #lonlat to tangent plane distance
    #center at (lon0, lat0)
    #y = R*(lat-lat0)
    #x = R*cos(lat0)*(lon-lon0)
    
    #tangent plane to lonlat
    #lat = lat0+y/R
    #lon = lon0+x/(R*cos(lat0))
    rE = 6.368e6
    
    lat = lat0*np.pi/180+x/rE
    lon = lon0*np.pi/180+(x/rE/np.cos(lat0*np.pi/180))
    lat*=180/np.pi
    lon*=180/np.pi
    
    xs, ys = wcofs_lonlat_2_xy(np.asarray([lon0, lon]), np.asarray([lat0, lat]), False)
    
    return xs[1]-xs[0], ys[1]-ys[0]

@jit(cache=True, forceobj=True)
def WCOFS_RK4(field, xn, yn, tn, dt):#, ux, uy, vx, vy, us, vs):
    k1x = field.interpolate("u", xn, yn, round(tn))*3600*dt*100
    k1y = field.interpolate("v", xn, yn, round(tn))*3600*dt*100
    #print(f"{k1x}, {k1y}")*3600*dt*100
    #print(f"{k1x}, {k1y}")
    #print(f"({xn}, {yn})->({k1x}, {k1y})")
    xi = xn+.5*k1x#*dt*3600*1000
    yi = yn+.5*k1y#*dt*3600*1000
    k2x = field.interpolate("u", xi, yi, round(tn+dt/2))*3600*dt*100
    k2y = field.interpolate("v", xi, yi, round(tn+dt/2))*3600*dt*100
    xi = xn+.5*k2x#*dt*3600*1000
    yi = yn+.5*k2y#*dt*3600*1000
    k3x = field.interpolate("u", xi, yi, round(tn+dt/2))*3600*dt*100
    k3y = field.interpolate("v", xi, yi, round(tn+dt/2))*3600*dt*100
    xi = xn+k3x#*dt*3600*1000
    yi = yn+k3y#*dt*3600*1000
    k4x = field.interpolate("u", xi, yi, round(tn+dt))*3600*dt*100
    k4y = field.interpolate("v", xi, yi, round(tn+dt))*3600*dt*100
    xi = xn+1/6*(k1x+2*k2x+2*k3x+k4x)
    yi = yn+1/6*(k1y+2*k2y+2*k3y+k4y)
    return xi, yi, tn+dt

@jit(cache=True, forceobj=True)
def WCOFS_RK2(field, xn, yn, tn, dt):
    k1x = field.interpolate("u", xn, yn, round(tn))*3600*dt*100
    k1y = field.interpolate("v", xn, yn, round(tn))*3600*dt*100
    xi = xn+.5*k1x
    yi = yn+.5*k1y
    k2x = field.interpolate("u", xi, yi, round(tn+dt/2))*3600*dt*100
    k2y = field.interpolate("v", xi, yi, round(tn+dt/2))*3600*dt*100
    xi = xi+k2x
    yi = yi+k2y
    return xi, yi, tn+dt

def animate(coords, aax, lons, lats, u, v, loni, lati):
    aax.clear()
    aax.quiver(lons, lats, u, v, color="cyan")
    plons = loni.ev(coords[0], coords[1])
    plats = lati.ev(coords[0], coords[1])
    pathVs = np.asarray([plons, plats])
    pathVs = pathVs.T
    pp = Path(pathVs, closed=True)
    patch = PathPatch(pp, edgecolor=None)
    ax.add_artist(patch)

if __name__ == "__main__": #prevent errors when I import from advection.py
    mprof = SProfiler() #crude timing mechanism
    grd = xr.open_dataset("grd_wcofs_large_visc200.nc")
    dat = xr.open_mfdataset("zuvt_qck_Exp41_35*.nc", combine="by_coords", parallel=True, data_vars="minimal", chunks={"ocean_time":1}) #open all datasets beginning with "zuvt_qck_Exp41_35"

    rsl = slice(-1180, -700)
    csl = slice(-690, -120)
    tsl = slice(0, tend+1)
    lon0 = -125.5
    lat0 = 40.25
    x0, y0 = wcofs_lonlat_2_xy(lon0, lat0, False)

    lon_u, lat_u = grd["lon_u"], grd["lat_u"]
    lon_v, lat_v = grd["lon_v"].loc[rsl, csl], grd["lat_v"].loc[rsl, csl]
    lonrs = grd.coords["lon_rho"].loc[rsl, csl]
    latrs = grd.coords["lat_rho"].loc[rsl, csl]
    xu, yu = wcofs_lonlat_2_xy(lon_u, lat_u, True)
    xv, yv = wcofs_lonlat_2_xy(lon_v, lat_v, True)
    rx, ry = wcofs_lonlat_2_xy(lonrs, latrs, True)

    mu  = grd["mask_u"]
    us = dat["u_sur"][tsl, rsl, csl]
    vs = dat["v_sur"][tsl, rsl, csl]
    mprof.add("loading")

    xc = np.cos(xv*np.pi/180)
    u = us/R #dx/dt
    v = vs/R/xc #dy/dt

    mprof.add("setup")

    u = u.fillna(0.0)
    v = v.fillna(0.0)
    mprof.add("fill-nans")
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    Q = ax.quiver(lonrs, latrs, u[1], v[1], color="cyan", scale=1e-6, scale_units="xy")
    ax.contour(lon_u, lat_u, mu, colors="red")
    mprof.add("initial-plot")
    
    mfield = gd.DataField(xu[rsl, csl], yu[rsl, csl], xv, yv, u, v, True)

    vx, vy = [], []
    px = x0
    py = y0
    t = 0

    #particle cloud
    n = 20
    lons = np.random.uniform(-131.4, -127.8, n)
    lats = np.random.uniform(38, 42.5, n)
    xs, ys = wcofs_lonlat_2_xy(lons, lats, False)
    colors = list(mcolors.CSS4_COLORS)[::-1]

    lonInt = rbv(rx[0], ry[:, 0], lonrs.transpose(transpose_coords=False), kx=1, ky=1)
    latInt = rbv(rx[0], ry[:, 0], latrs.transpose(transpose_coords=False), kx=1, ky=1)

    conR = 50000 #contour radius (m)
    conRxy = geodist_to_xy(conR, lons[0], lats[0])
    circle = gd.ParticleContour(xs[0], ys[0], conRxy[0], 100, WCOFS_RK2, mfield)
    x1 = [p.trajx[-1] for p in circle.particles]
    y1 = [p.trajy[-1] for p in circle.particles]
    plons = list(lonInt.ev(x1, y1))
    plons.append(plons[0])
    plats = list(latInt.ev(x1, y1))
    plats.append(plats[0])
    ax.plot(plons, plats, marker="X", color=colors[-7])
    #anim = FuncAnimation(fig, animate, circle.advect(0, tend, .3), save_count=24*5, fargs=(ax, lonrs, latrs, u[1], v[1], lonInt, latInt))
    #anim.save("5dcontourPatchT.gif", PillowWriter(10))
    #plt.close(fig)
    circle.advect(0,tend,.3)
    
    pxl = [p.trajx[-1] for p in circle.particles]
    pyl = [p.trajy[-1] for p in circle.particles]
    
    contourDS = xr.Dataset(data_vars={"T0":(["var", "p0"], [x1, y1]), "TF":(["var", "pf"], [pxl, pyl])}, coords={"var": ["x", "y"]})
    contourDS.to_netcdf("total_contour.nc", format="NETCDF4") #save particle coordinates for later calculations

    plons = lonInt.ev(pxl, pyl)
    #plons = np.append(plons, plons[0])
    plats = latInt.ev(pxl, pyl)
    #plats = np.append(plats, plats[0])
    #ax.plot(plons, plats) #, marker="o", markersize="2")
    pathVs = np.asarray([plons, plats])
    pathVs = pathVs.T
    pp = Path(pathVs, closed=True)
    patch = PathPatch(pp, edgecolor=None)
    ax.add_artist(patch)
    mprof.add("contour-object")

    mprof.show()

    ax.set_aspect(1.3)
    plt.show()