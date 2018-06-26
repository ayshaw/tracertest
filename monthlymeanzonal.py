# load required packages
import numpy as np	
import xarray as xr
import matplotlib.pyplot as plt # general plotting
plt.switch_backend('agg')
import re
import glob
import cartopy.crs as ccrs

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
mycmap = discrete_cmap(21,'spectral')
plt.figure(figsize=[30,10])
# vector of months 
#t = ['03','04','05','06','07', '08']
t=['03','04','05'] 
# loop through each month to get monthly mean of Rn, Pb
for tick_t in range(len(t)):
	
	# import filenames from that month and make sure they are the same length
	filenames132 = glob.glob('c90_RnPbBe_L132/holding/geosgcm_gcc_p/c90_L132.geosgcm_gcc_p.2013'+t[tick_t]+'*0000z.nc4')
	filenames72 = glob.glob('c90_RnPbBe_L72/holding/geosgcm_gcc_p/c90_L72.geosgcm_gcc_p.2013'+t[tick_t]+'*0000z.nc4')
	filenames132.sort
	filenames72.sort
	print filenames72
	print filenames132
	index_f = len(filenames132)	
	if index_f>len(filenames72):
		index_f = len(filenames72)
	print '\n'+'num_files'+str(index_f) 	
	sum132 = np.zeros([48,181])#initialize monthly sum matrix
	sum72 = np.zeros([48,181])

	lev = np.zeros(48)#initalize level matrix
	# loop through the month here
	for tick_f in range(index_f):
		ds = xr.open_dataset(filenames132[tick_f])
		dr = xr.open_dataset(filenames72[tick_f])
		lev = ds['lev'].values # get level data
		
		time_stamp = ds['time'].values[0]
		#tracer data
		data132 = np.reshape(ds['TRC_Rn'].values,[48,181,360])
		data72 = np.reshape(dr['TRC_Rn'].values,[48,181,360])

		#zonal mean
		avg132 = np.average(data132,axis=2)
		avg72 = np.average(data72,axis=2)
		print avg72		

		#aggregate monthly sum
		sum132 += avg132 
		sum72 += avg72

	# divide to find monthly average	
	monthlyavg132 = sum132/index_f
	monthlyavg72 = sum72/index_f
	eps =1e-38
	monthlyavg132[np.abs(monthlyavg132)<eps]=0
	monthlyavg72[np.abs(monthlyavg72)<eps]=0
	print sum132

	#find percent differences
	#perc_avg = np.divide((monthlyavg72-monthlyavg132),monthlyavg72,out=np.zeros_like(monthlyavg72-monthlyavg132),where=monthlyavg72!=0)*100	
	perc_avg = monthlyavg72
	print perc_avg

	#plot monthly average
	plt.hold(True)
	plt.subplot(2,3,tick_t+1)
	lat = np.linspace(-90,90,181)
	#plt.title(t[tick_t]+"percent difference (L72-L132)/L72 *100%")
        plt.title(t[tick_t] + "monthly average")
	pc=plt.pcolormesh(lat,lev[0:21],perc_avg[0:21,:],vmin=0,vmax=2.4e-19,cmap = mycmap)
        #if t[tick_t]=='08':
	#	plt.colorbar(pc,cmap=mycmap,extend='both',label='kg/kg')
	plt.colorbar(pc,cmap=mycmap,extend='both',label='$\Delta$%')
	plt.gca().invert_yaxis()
plt.savefig('images/L72_p_mz_Rnavg.png')
	
