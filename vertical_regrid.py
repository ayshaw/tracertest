import numpy as np
import xarray as xr
import matplotlib.pyplot as plt # general plotting
import cartopy.crs as ccrs # plot on maps, better than the Basemap module
import re
import scipy.interpolate
import glob

filenames = glob.glob("AEIC.72L.gen.1x1.nc")
ds = xr.open_dataset(filenames[0])

#get list of molecules
molec = list(ds.keys())[4:]

#initialize NetCDF-4
dataset = xr.Dataset()
dataset.coords['time'] = ds.coords['time']
lev = np.linspace(1,132,num =132,dtype = 'f')
dataset.coords['lev'] = (('lev'),lev)
dataset.coords['lev'].attrs = ds.coords['lev'].attrs
dataset.coords['lat'] = ds.coords['lat'] 
dataset.coords['lon'] = ds.coords['lon']
dataset.attrs['title'] = filenames[0] + '@ 132 lvl'
dataset.attrs['history'] = 'created by Ada Shaw with' +filenames[0]
dataset.attrs['format'] = "NetCDF-4"
dataset.attrs['conventions'] = 'COARDS'

#find dimensions of model output
lon_dim = len(ds['lon'].values)
lat_dim = len(ds['lat'].values)
time_vector = ds['time'].values
time_dim = len(time_vector)

#initialize variables for interpolation
lon_i =np.linspace(1,lon_dim,num= lon_dim,dtype ='float64')
lev72 = np.linspace(1,72, num=72,dtype = 'float64')
lev132 = np.linspace(1,72, num=132,dtype = 'float64')

#loop to save interpolated concentrations of species
for tick_molec in range(len(molec)):
    conc_interp = np.ndarray(shape=(time_dim,132,lat_dim,lon_dim), dtype='f', order='F')
    dr = ds[molec[tick_molec]].values
    for tick_lat in range(lat_dim):
        for tick_time in range(time_dim):
            interp = scipy.interpolate.interp2d(lon_i,lev72 ,dr[tick_time,:,tick_lat,:], kind='linear')
            conc_interp[tick_time,:,tick_lat,:] = interp(lon_i,lev132)
    dataset[molec[tick_molec]] = (('time','lev','lat','lon'),conc_interp)
    dataset.data_vars[molec[tick_molec]].attrs = ds.data_vars[molec[tick_molec]].attrs

#save to netCDF
dataset.to_netcdf('132_lvl'+filenames[0],'w')
