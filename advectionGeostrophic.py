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
from advection import WCOFS_RK2, geodist_to_xy
from numba import jit, float64, int32, void, vectorize
import genfield_data as gd
from dask import delayed
import dask
import xarray as xr

#total current is good for high turbulence areas
#geostrophic destroys the contour there

R = 6.368e6 #earth radius (m)
tend = 24*10

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

if __name__ == "__main__":
    mprof = SProfiler()
    grd = xr.open_dataset("grd_wcofs_large_visc200.nc")
    dat = xr.open_mfdataset("zuvt_qck_Exp41_35*.nc", combine="by_coords", parallel=True, data_vars="minimal", chunks={"ocean_time":1})
    
    #to subset the displayed & interpolated vector field for speed
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
    
    gdat = xr.open_mfdataset("geostrophic_uv*.nc", combine="by_coords", parallel=True, chunks={"ocean_time":1})
    gu = gdat["ug"][tsl, rsl, csl]
    gv = gdat["vg"][tsl, rsl, csl]
    gu = gu/R
    gv = gv/R/xc
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    Q = ax.quiver(lonrs, latrs, gu[1], gv[1], color="cyan", scale=1e-6, scale_units="xy")
    ax.contour(lon_u, lat_u, mu, colors="red")
    mprof.add("initial-plot")
    
    mfield = gd.DataField(xu[rsl, csl], yu[rsl, csl], xv, yv, gu, gv, True)

    pxl, pyl = [], []
    x1, y1 = [], []
    px = x0
    py = y0
    t = 0

    n = 5
    lons = np.random.uniform(-131.4, -127.8, n)
    lats = np.random.uniform(38, 42.5, n)
    xs, ys = wcofs_lonlat_2_xy(lons, lats, False)
    colors = list(mcolors.CSS4_COLORS)[::-1]
    conR = 50000 #contour radius (m)
    
    lonInt = rbv(rx[0], ry[:, 0], lonrs.transpose(transpose_coords=False), kx=1, ky=1)
    latInt = rbv(rx[0], ry[:, 0], latrs.transpose(transpose_coords=False), kx=1, ky=1)
    
    #this block creates multiple contours
    '''for i in range(n):
        conRxy = geodist_to_xy(conR, lons[i], lats[i])
        circle = gd.ParticleContour(xs[i], ys[i], conRxy[0], 100, WCOFS_RK2, mfield)
        x1 = [p.trajx[-1] for p in circle.particles]
        y1 = [p.trajy[-1] for p in circle.particles]
        #plons = [lonInt.ev(p.trajx[-1], p.trajy[-1]) for p in circle.particles]
        #plons.append(plons[0])
        #plats = [latInt.ev(p.trajx[-1], p.trajy[-1]) for p in circle.particles]
        #plats.append(plats[0])
        ax.plot(lons[i], lats[i], marker="X", color=colors[i])
        #anim = FuncAnimation(fig, animate, circle.advect(0, tend, .3), save_count=24*15, fargs=(ax, lonrs, latrs, u[1], v[1], lonInt, latInt))
        #anim.save("20dcontourPatchG.gif", PillowWriter(10))
        circle.advect(0,tend,.3)

        pxl = [p.trajx[-1] for p in circle.particles]
        pyl = [p.trajy[-1] for p in circle.particles]

        plons = lonInt.ev(pxl, pyl)
        #plons = np.append(plons, plons[0])
        plats = latInt.ev(pxl, pyl)
        #plats = np.append(plats, plats[0])
        #ax.plot(plons, plats, color=colors[i]) #, marker="o", markersize="2")
        pathVs = np.asarray([plons, plats])
        pathVs = pathVs.T
        pp = Path(pathVs, closed=True)
        patch = PathPatch(pp, color=colors[i], edgecolor=None)
        ax.add_artist(patch)'''
       
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
    #anim.save("5dcontourPatchG.gif", PillowWriter(10))
    circle.advect(0,tend,.3)
    pxl = [p.trajx[-1] for p in circle.particles]
    pyl = [p.trajy[-1] for p in circle.particles]
    
    plons = lonInt.ev(pxl, pyl)
    plats = latInt.ev(pxl, pyl)
    pathVs = np.asarray([plons, plats])
    pathVs = pathVs.T
    pp = Path(pathVs, closed=True)
    patch = PathPatch(pp, color=colors[0], edgecolor=None)
    ax.add_artist(patch)
    
    contourDS = xr.Dataset(data_vars={"T0":(["var", "p0"], [x1, y1]), "TF":(["var", "pf"], [pxl, pyl])}, coords={"var": ["x", "y"]})
    contourDS.to_netcdf("geostrophic_contour.nc", format="NETCDF4") #save 1 contour for later calculations
        
    mprof.add("contours")

    mprof.show()

    ax.set_aspect(1.3)
    plt.show()