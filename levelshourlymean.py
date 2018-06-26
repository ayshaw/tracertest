# load required packages
import numpy as np	
import xarray as xr
import matplotlib.pyplot as plt # general plotting
plt.switch_backend('agg')
import re
import glob
import cartopy.crs as ccrs
import scipy.interpolate
# make my own discrete cmap
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
mycmap = discrete_cmap(21,'RdBu')
fig = plt.figure(figsize=[30,10])
# vector of months 
t = ['00','03','06','09','12','15','18','21']
#t=['03','04','05'] 
# loop through each month to get monthly mean of Rn, Pb
for tick_t in range(len(t)):
	
	# import filenames from that month and make sure they are the same length
	filenames132 = glob.glob('c90_RnPbBe_L132/holding/geosgcm_gcc_lev/c90_L132.geosgcm_gcc_lev.2013*'+t[tick_t]+'00z.nc4')
	filenames72 = glob.glob('c90_RnPbBe_L72/holding/geosgcm_gcc_lev/c90_L72.geosgcm_gcc_lev.2013*'+t[tick_t]+'00z.nc4')
	filenames132.sort
	filenames72.sort
	print filenames72
	print filenames132
	index_f = len(filenames132)	
	if index_f>len(filenames72):
		index_f = len(filenames72)
	print '\n'+'num_files'+str(index_f) 	
	sum132 = np.zeros([132,181])#initialize monthly sum matrix
	sum72 = np.zeros([72,181])

	lev132 = np.zeros([132])#initalize level matrix
	lev72 = np.zeros([72])
	# loop through the month here
	for tick_f in range(index_f):
		ds = xr.open_dataset(filenames132[tick_f])
		dr = xr.open_dataset(filenames72[tick_f])
		lev132 = np.reshape(ds['PL'].sel(lat = 35,lon=-121).values,[132]) # get level data
		lev72 = np.reshape(dr['PL'].sel(lat=35,lon=-121).values,[72])
		print  np.shape(lev132)	
		time_stamp = ds['time'].values[0]
		#tracer data
		data132 = np.reshape(ds['TRC_Rn'].values,[132,181,360])
		data72 = np.reshape(dr['TRC_Rn'].values,[72,181,360])

		#zonal mean
		avg132 = np.average(data132,axis=2)
		avg72 = np.average(data72,axis=2)
		print avg72		

		#aggregate monthly sum
		sum132 += avg132 
		sum72 += avg72

	# divide to find monthly average	
	monthlyavg132 = sum132/index_f
	monthlyavg72_lowres = sum72/index_f
        
	#interpolate 72 layers to 132
        lat = np.linspace(-90,90,181)
        print np.shape(lat)
        interp = scipy.interpolate.interp2d(lat,lev72,monthlyavg72_lowres,kind = 'linear')
        monthlyavg72 = interp(lat,lev132)
	
	#find percent differences
	perc_avg = np.divide((monthlyavg72-monthlyavg132),monthlyavg72,out=np.zeros_like(monthlyavg72-monthlyavg132),where=monthlyavg72!=0)*100	
	#perc_avg = monthlyavg132###<--CHANGE---|###
	eps=1e-38
	perc_avg[np.abs(perc_avg)<eps]=0
	print perc_avg

	#plot monthly average
	plt.hold(True)
	plt.subplot(2,4,tick_t+1)
	lat = np.linspace(-90,90,181)
	pc = plt.pcolormesh(lat,lev132,perc_avg,cmap=mycmap,vmin=100,vmax=-100)###<--CHANGE---|###
	plt.title(t[tick_t]+"percent difference (L72-L132)/L72 *100%")
        #plt.title(t[tick_t] + "hourly average")
        #if t[tick_t]=='08':
	#	cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
	#	fig.colorbar(pc, cax=cax)
	plt.ylim([0,100000])
	plt.gca().invert_yaxis()
	plt.colorbar(pc,cmap=mycmap,extend='both',label='$\Delta$%')
plt.savefig('images/132v72hourlyavgRn.png')###<--CHANGE---|###
	
