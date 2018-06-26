import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
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
dates = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14']
colmean_L132 = np.zeros([len(dates)])
colmean_L72 = np.zeros([len(dates)])

#calculate the area of each lat/lon grid:
latit = np.linspace(-90,90,num  = 181)
longit = np.linspace(-180,180,num = 360)
Area = np.ones([181,360])
for tick_lat in range(len(latit)):
    for tick_lon in range(len(longit)):
        latit_low = (latit[tick_lat]-0.5)*np.pi/180
        latit_high = (latit[tick_lat]+0.5)*np.pi/180
        longit_low = longit[tick_lon]-0.5
        longit_high = longit[tick_lon]+0.5
        #square meters:
        Area[tick_lat,tick_lon] = np.pi/180*6378.1e3**2*abs(np.sin(latit_high)-np.sin(latit_low))*abs(longit_high-longit_low)#dim: 181,360
def entropy(ZL_da, PL_da, T_da, mr_da,vertres):
	#zedge = 0.5*(ZL_da.values[0,np.arange(0,vertres,2),:,:]+ZL_da.values[0,np.arange(1,vertres+1,2),:,:])#dim: 71,181,360
	zedge = (ZL_da.values[0,1:,:,:] + ZL_da.values[0,:-1,:,:])/2
	Area3d = np.repeat(Area[np.newaxis,:,:], vertres-2,axis=0)
	dz = zedge[np.arange(0,vertres-2,1),:,:]-zedge[np.arange(1,vertres-1,1),:,:] #dim: 70,181,360
	dv = Area3d * np.squeeze(dz)
	air_mass = dv * 100 * PL_da.values[0,np.arange(1,vertres-1,1),:,:] /8.314/T_da.values[0,np.arange(1,vertres-1,1),:,:]*29
	mixingrat = mr_da.values[0,np.arange(1,vertres-1,1),:,:]*29
	print np.log(mixingrat*29, where=mixingrat>0 )
	entropy = np.sum(air_mass*mixingrat*np.log(mixingrat*29, where=mixingrat>0),axis = None)
	return(entropy)
def easyent(airmass_da, mr_da):
	mixing_ratio=mr_da.values * 29
	entropy = np.sum(airmass_da.values*mixing_ratio*np.log(mixing_ratio, where=mixing_ratio>0),axis=None)
	return(entropy)
	
for tick_d in range(len(dates)):
	filenames132 = glob.glob('c90_RnPbBe_L132/holding_plume03/geosgcm_gcc_lev/c90_L132.geosgcm_gcc_lev.201303'+dates[tick_d]+'*z.nc4')
	filenames72 = glob.glob('c90_RnPbBe_L72/holding_plume03/geosgcm_gcc_lev/c90_L72.geosgcm_gcc_lev.201303'+dates[tick_d]+'*z.nc4')
	num_tick_f = len(filenames72)
	lev132 = np.zeros([132])
	lev72 = np.zeros([72])
	print dates[tick_d]
	if len(filenames132)<len(filenames72):
		num_tick_f=len(filenames132)
	acccolmean_L132 = np.zeros([1])
	acccolmean_L72 = np.zeros([1])
	for tick_f in range(num_tick_f):
		ds = xr.open_dataset(filenames132[tick_f])
		dr = xr.open_dataset(filenames72[tick_f])
		#lev72 = np.reshape(dr['PL'].mean(dim=['lat','lon']).values,[72])
		#lev132 = np.reshape(ds['PL'].mean(dim=['lat','lon']).values,[132])
		if tick_d == 0 and tick_f == 0:
			k_132 = entropy(ds['ZL'],ds['PL'],ds['T'],ds['TRC_PASV'],132)
			k_72 = entropy(dr['ZL'],dr['PL'],dr['T'],dr['TRC_PASV'],72)
			entropy_L132 = 1
			entropy_L72 = 1
		else:
			entropy_L132 =entropy(ds['ZL'],ds['PL'],ds['T'],ds['TRC_PASV'],132)/k_132
			entropy_L72 =entropy(dr['ZL'],dr['PL'],dr['T'],dr['TRC_PASV'],72)/k_72
		#entropy_L132 = easyent(ds['AIRMASS'],ds['TRC_PASV'])	
		#entropy_L72 = easyent(dr['AIRMASS'],dr['TRC_PASV'])
		print entropy_L72 
		acccolmean_L132 += entropy_L132
		acccolmean_L72 += entropy_L72
#		acccolmean_L132 += np.squeeze(ds['TRC_PASV'].sum(dim = 'lat').values)
#		acccolmean_L72 += np.squeeze(dr['TRC_PASV'].sum(dim = 'lat').values)		
	lon = np.linspace(-180,180,num = 360)
	print np.shape(acccolmean_L132)
	colmean_L132[tick_d] = acccolmean_L132/num_tick_f/181/360
	colmean_L72[tick_d] = acccolmean_L72/num_tick_f/181/360
	print colmean_L132
	print colmean_L72
	
	#interpolate 72 layers to 132 layers
        #interp = scipy.interpolate.interp2d(lon,lev72,colmean_L72_lowres,kind = 'linear')
        #colmean_L72 = interp(lon,lev132)

	# L72 plot
	#ax = plt.subplot(14,3,1+3*tick_d)
	#pc = ax.pcolormesh(lon,lev132,colmean_L72,vmin=0,vmax=20)
	#plt.xlabel('longitude')
	#plt.xlim([-180,180])
	#plt.gca().invert_yaxis()
	#plt.title('L72 day'+dates[tick_d])
	#plt.colorbar(pc,orientation='horizontal',shrink = .65,label = 'average mixing ratio kg/kg')

	# L132 plot
	#ax1 =plt.subplot(14,3,2+3*tick_d)
	#plt.xlabel('longitude')
	#plt.xlim([-180,180])
	#plt.title('L132 day'+dates[tick_d])
	#pc = ax1.pcolormesh(lon,lev132,colmean_L132,vmin = 0,vmax =20)
	#plt.gca().invert_yaxis()
	#plt.colorbar(pc,orientation='horizontal',shrink = .65,label = 'average mixing ratio kg/kg')

	# L72 v L132 plot
	#perc_avg = np.divide((colmean_L72-colmean_L132),colmean_L72,out=np.zeros_like(colmean_L72-colmean_L132),where=colmean_L72!=0)*100
	#ax2 =plt.subplot(14,3,3+3*tick_d)
	#plt.xlabel('longitude')
	#plt.xlim([-180,180])
	#plt.title('day'+dates[tick_d])
	#pc = ax2.pcolormesh(lon,lev132,perc_avg,vmin = -100,vmax =100,cmap = divcmap)
	#plt.gca().invert_yaxis()
	#plt.colorbar(pc,orientation='horizontal',shrink = .65,label = 'red:more L132   percent average   blue:more L72')
plt.subplot(111)
L132 = plt.plot(np.arange(1,15,1), colmean_L132,'bo--',linewidth=2,linestyle = 'dashed',markersize =4, label = 'L132')
plt.hold(True)
plt.xlabel('day')
plt.ylabel('plume entropy')
L72 = plt.plot(np.arange(1,15,1), colmean_L72,'ko--',linewidth=2,label ='L72',markersize = 4)
plt.legend(loc = 'best')
plt.savefig('plumeimages/entropy_plume05.2.png')
