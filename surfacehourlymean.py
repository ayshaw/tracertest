# load required packages
import numpy as np	
import xarray as xr
import matplotlib.pyplot as plt # general plotting
plt.switch_backend('agg')
import re
import glob
import cartopy.crs as ccrs
import scipy.interpolate
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
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
# vector of hours
t = ['00','03','06','09','12','15','18','21']
# vector of lat lons
loc = np.array([[39.0,-33.0,39.0,-26.0],[-121.0,-77.0,116.0,28.0]]) #([lat],[lon]) example  loc[0,0],loc[1,0]: lat = 35,lon=-121
#t=['03','04','05'] 
# loop through each month to get monthly mean of Rn, Pb
mean132=np.zeros([4,8])
mean72 = np.zeros([4,8])	
utc = np.array([-8,-4,8,2])
colors = ['r','g','b','m']
for tick_t in range(len(t)):
	
	# import filenames from that hour and make sure they are the same length
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

	data132 = np.zeros([4]) # initialize matrix to store all the data from each hour
	data72 = np.zeros([4])
	temp132 = np.zeros([4])
	temp72 = np.zeros([4])
	# loop through all the files at that time here and grab all the data from each location
	for tick_f in range(index_f):
		ds = xr.open_dataset(filenames132[tick_f])
		dr = xr.open_dataset(filenames72[tick_f])
		time_stamp = ds['time']
		# loop through all locations and sum up all concentrations
		for tick_l in range(4):
			#tracer data
			data132[tick_l] += np.reshape(ds['TRC_Rn'].sel(lat=loc[0,tick_l],lon=loc[1,tick_l],lev=132).values,[1])
			data72[tick_l] += np.reshape(dr['TRC_Rn'].sel(lat=loc[0,tick_l],lon=loc[1,tick_l],lev=72).values,[1])
			temp72[tick_l] += np.reshape(dr['T'].sel(lat=loc[0,tick_l],lon=loc[1,tick_l],lev=72).values,[1])
			temp132[tick_l] += np.reshape(ds['T'].sel(lat=loc[0,tick_l],lon=loc[1,tick_l],lev=72).values,[1])
	# find average value for time and convert values to becquerel
	#mean132[:,tick_t] = (data132/index_f)/(temp132/index_f)*101325/(8.314/.029)*3.7e16
	#mean72[:,tick_t] = (data72/index_f)/(temp72/index_f)*101325/(8.314/.029)*3.7e16	
	mean132[:,tick_t] = (data132/index_f)
	mean72[:,tick_t] = (data72/index_f)	
	label = list()
for tick_loc in range(4):
	time = np.arange(0,24,3)+utc[tick_loc]
	time[time>24]=time[time>24]-24
	time[time<0]=time[time<0]+24
	d132 = zip( *sorted( zip(time, mean132[tick_loc,:]) ) )	
	d72 = zip( *sorted( zip(time, mean72[tick_loc,:]) ) )		
	plt.hold(True)
	plt.subplot(2,2,tick_loc+1)
	plt.plot(d132[0],d132[1],linewidth=2,linestyle = 'solid', marker = '.',c=colors[tick_loc])
	plt.plot(d72[0],d72[1],linewidth=2,linestyle = 'dashed',marker = 'o',c=colors[tick_loc])
	plt.xlim=([0,24])
	plt.ylabel('mBq/sqm Rn222')
	plt.xlabel("local time [h]")
	plt.xticks(np.arange(0,24,1))
	plt.title('(lat,lon) ='+ '('+ str(loc[0,tick_loc])+ ','+ str(loc[1,tick_loc])+')')
L132 = mlines.Line2D([], [], color='k', label='L132',linestyle='solid')
L72 = mlines.Line2D([], [],color='k', label='L72',linestyle='dashed')
plt.legend(handles=[L132,L72])
plt.savefig('images/debugsantiagohourly_values.png')	
