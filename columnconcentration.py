import numpy as np
import xarray as xr
import matplotlib.pyplot as plt # general plotting
plt.switch_backend('agg')
import re
import re
import glob
import cartopy.crs as ccrs
import matplotlib.lines as mlines
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

def add_latlon_ticks(ax):
    '''Add latlon label ticks and gridlines to ax

    Adapted from
    http://scitools.org.uk/cartopy/docs/v0.13/matplotlib/gridliner.html
    '''
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylocator = mticker.FixedLocator(np.arange(-90,91,30))
fig0 = plt.figure(num = 0,figsize = [80,20])
#dates = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14']
dates = ['15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
for tick_d in range(len(dates)):
	filenames132 = glob.glob('c90_RnPbBe_L132/holding/geosgcm_gcc_lev/c90_L132.geosgcm_gcc_lev.201303'+dates[tick_d]+'*z.nc4')
	filenames72 = glob.glob('c90_RnPbBe_L72/holding/geosgcm_gcc_lev/c90_L72.geosgcm_gcc_lev.201303'+dates[tick_d]+'*z.nc4')
	num_tick_f = len(filenames72)
	if len(filenames132)<len(filenames72):
		num_tick_f=len(filenames132)
	acccolmean_L132 = np.zeros([181,360])
	acccolmean_L72 = np.zeros([181,360])
	for tick_f in range(num_tick_f):
		ds = xr.open_dataset(filenames132[tick_f])
		dr = xr.open_dataset(filenames72[tick_f])
		acccolmean_L132 += np.squeeze(ds['TRC_PASV'].sum(dim = 'lev').values)
		acccolmean_L72 += np.squeeze(dr['TRC_PASV'].sum(dim = 'lev').values)		
	lat = np.linspace(-90,90,num  = 181)
	lon = np.linspace(-180,180,num = 360)
	colmean_L132 = acccolmean_L132/num_tick_f
	colmean_L72 = acccolmean_L72/num_tick_f
	# L72 plot
	ax = plt.subplot(3,17,3*tick_d+1,projection=ccrs.PlateCarree())
	ax.coastlines()
	plt.title('L72 day'+dates[tick_d])
	pc = ax.pcolormesh(lon,lat,colmean_L72,vmin =0, vmax = 1800)
	plt.colorbar(pc,orientation='horizontal',shrink = .65,label = 'average column mixing ratio kg/kg')
	add_latlon_ticks(ax)
	# L132 plot
	ax1 = plt.subplot(3,17,3*tick_d+2,projection=ccrs.PlateCarree())
	plt.title('L132 day'+dates[tick_d])
	ax1.coastlines()
	pc = ax1.pcolormesh(lon,lat,colmean_L132,vmin = 0,vmax = 1800)
	plt.colorbar(pc,orientation='horizontal',shrink = .65,label = 'average column mixing ratio kg/kg')
	add_latlon_ticks(ax1)
	# L72 versus L132
	perc_avg = np.divide((colmean_L72-colmean_L132),colmean_L72,out=np.zeros_like(colmean_L72-colmean_L132),where=colmean_L72!=0)*100
        ax2 =plt.subplot(3,17,3*tick_d+3,projection=ccrs.PlateCarree())
        plt.title('day'+dates[tick_d])
        ax2.coastlines()
        pc = ax2.pcolormesh(lon,lat,colmean_L132,vmin = -100,vmax = 100)
        plt.colorbar(pc,orientation='horizontal',shrink = .65,label = 'red:more L132   percent average   blue:more L72')
        add_latlon_ticks(ax1)
fig0.savefig('plumeimages/L72_plume05.2.png')
