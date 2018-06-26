import numpy as np
import xarray as xr
import matplotlib.pyplot as plt # general plotting
plt.switch_backend('agg')
import re
import re
import glob
import cartopy.crs as ccrs
import scipy.interpolate

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
mycmap = discrete_cmap(21,'spectral')
divcmap = discrete_cmap(21,'RdBu')

fig0 = plt.figure(num = 0,figsize = [30,150])
#dates = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14']
dates = ['15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
for tick_d in range(len(dates)):
	filenames132 = glob.glob('c90_RnPbBe_L132/holding/geosgcm_gcc_lev/c90_L132.geosgcm_gcc_lev.201303'+dates[tick_d]+'*z.nc4')
	filenames72 = glob.glob('c90_RnPbBe_L72/holding/geosgcm_gcc_lev/c90_L72.geosgcm_gcc_lev.201303'+dates[tick_d]+'*z.nc4')
	num_tick_f = len(filenames72)
	lev132 = np.zeros([132])
	lev72 = np.zeros([72])
	print dates[tick_d]
	if len(filenames132)<len(filenames72):
		num_tick_f=len(filenames132)
	acccolmean_L132 = np.zeros([132,360])
	acccolmean_L72 = np.zeros([72,360])
	for tick_f in range(num_tick_f):
		ds = xr.open_dataset(filenames132[tick_f])
		dr = xr.open_dataset(filenames72[tick_f])
		lev72 = np.reshape(dr['PL'].mean(dim=['lat','lon']).values,[72])
		lev132 = np.reshape(ds['PL'].mean(dim=['lat','lon']).values,[132])
		acccolmean_L132 += np.squeeze(ds['TRC_PASV'].sum(dim = 'lat').values)
		acccolmean_L72 += np.squeeze(dr['TRC_PASV'].sum(dim = 'lat').values)		
	lon = np.linspace(-180,180,num = 360)
	colmean_L132 = acccolmean_L132/num_tick_f/181
	colmean_L72_lowres = acccolmean_L72/num_tick_f/181

	#interpolate 72 layers to 132 layers
        interp = scipy.interpolate.interp2d(lon,lev72,colmean_L72_lowres,kind = 'linear')
        colmean_L72 = interp(lon,lev132)

	# L72 plot
	ax = plt.subplot(17,3,1+3*tick_d)
	pc = ax.pcolormesh(lon,lev132,colmean_L72,vmin=0,vmax=2)
	plt.xlabel('longitude')
	plt.xlim([-180,180])
	plt.gca().invert_yaxis()
	plt.title('L72 day'+dates[tick_d])
	plt.colorbar(pc,orientation='horizontal',shrink = .65,label = 'average mixing ratio kg/kg')

	# L132 plot
	ax1 =plt.subplot(17,3,2+3*tick_d)
	plt.xlabel('longitude')
	plt.xlim([-180,180])
	plt.title('L132 day'+dates[tick_d])
	pc = ax1.pcolormesh(lon,lev132,colmean_L132,vmin = 0,vmax =2)
	plt.gca().invert_yaxis()
	plt.colorbar(pc,orientation='horizontal',shrink = .65,label = 'average mixing ratio kg/kg')

	# L72 v L132 plot
	perc_avg = np.divide((colmean_L72-colmean_L132),colmean_L72,out=np.zeros_like(colmean_L72-colmean_L132),where=colmean_L72!=0)*100
	ax2 =plt.subplot(17,3,3+3*tick_d)
	plt.xlabel('longitude')
	plt.xlim([-180,180])
	plt.title('day'+dates[tick_d])
	pc = ax2.pcolormesh(lon,lev132,perc_avg,vmin = -100,vmax =100,cmap = divcmap)
	plt.gca().invert_yaxis()
	plt.colorbar(pc,orientation='horizontal',shrink = .65,label = 'red:more L132   percent average   blue:more L72')

	
fig0.savefig('plumeimages/meridional_mean_plume05.2.png')
