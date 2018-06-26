import numpy as np
import xarray as xr
import matplotlib.pyplot as plt # general plotting
plt.switch_backend('agg')
import re
import re
import glob
import matplotlib.lines as mlines
filenames132 = glob.glob('c90_RnPbBe_L132/monthlyholding/*.nc4')
filenames132.sort
filenames132
filenames72 = glob.glob('c90_RnPbBe_L72/monthlyholding/*.nc4')
filenames72.sort
filenames72
latit = np.linspace(-90,90,num  = 181)
longit = np.linspace(-180,180,num = 360)
Area = np.ones([181,360])
for tick_lat in range(len(latit)):
    for tick_lon in range(len(longit)):
        latit_low = (latit[tick_lat]-0.5)*np.pi/180
        latit_high = (latit[tick_lat]+0.5)*np.pi/180
        longit_low = longit[tick_lon]-0.5
        longit_high = longit[tick_lon]+0.5
        Area[tick_lat,tick_lon] = np.pi/180*6378.1e3**2*abs(np.sin(latit_high)-np.sin(latit_low))*abs(longit_high-longit_low)
df = ['DryDep_Be7','DryDep_Pb','WetLossConv_Be7','WetLossConv_Pb' ]
lat = ds['lat'].values
plt.figure(figsize = [15,15])
plt.hold(True)
for tick_df in range(len(df)):
    tot_flux132 = np.zeros([181])
    tot_flux72 = np.zeros([181])
    for tick_f in range(len(filenames132)):
        ds = xr.open_dataset(filenames132[tick_f])
        a = np.sum(np.multiply(ds[df[tick_df]].sel(time = ds['time'].values[0]),Area),axis=1)
        tot_flux132 += a
        dr = xr.open_dataset(filenames72[tick_f])
        b = np.sum(np.multiply(dr[df[tick_df]].sel(time = dr['time'].values[0]),Area),axis=1)
        tot_flux72 += b
    plt.subplot(2,2,tick_df+1)
    red = linelabel132 = df[tick_df] + 'L132'
    black = linelabel72 = df[tick_df] + 'L72'
    plt.plot(lat,tot_flux132,color = 'red')
    plt.plot(lat,tot_flux72,color='black')
    plt.title('Total flux '+df[tick_df] +' [kg/s] summed over 6 months',fontsize = 15)
red_line = mlines.Line2D([], [], color='red', label='L132')
black_line = mlines.Line2D([], [], color='black', label='L72')   
plt.legend(handles=[red_line,black_line],frameon=False,fontsize = 15,loc = 'best')
