import numpy as np
import xarray as xr
import matplotlib.pyplot as plt # general plotting
plt.switch_backend('agg')
import re
import glob

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
# species data
filenames132 = glob.glob('c90_RnPbBe_L132/monthlyholding/*.nc4')
filenames132.sort
filenames72 = glob.glob('c90_RnPbBe_L72/monthlyholding/*.nc4')
filenames72.sort
# temperature data
filenamesT132 = glob.glob('c90_RnPbBe_L132/holding/geosgcm_prog/*.nc4')
filenamesT132.sort

species = ['TRC_Rn','TRC_Pb']

for tick_s in range(len(species)):
    num_tick_f = len(filenames72)

    # make sure filenames vectors are same length
    if len(filenames132)<len(filenames72):
        num_tick_f = len(filenames132)

    # iterate through filenames to plot
    for tick_f in range(num_tick_f):
        #individual plot for 132 levels
        time = re.search(r'(\d\d\d\d)(\d\d)(\d\d)_(\d\d)(\d\d)',filenames132[tick_f])
        t = [time.group(1) , time.group(2), time.group(3), time.group(4), time.group(5)]
        fig=plt.figure(figsize=[25,8])
	plt.subplot(1,2,1)
        ds = xr.open_dataset(filenames132[tick_f])
	lat = ds['lat'].values
        lev = ds['lev'].values
	lon = ds['lon'].values
	Pres = np.zeros([48,181,360])
	for tick_lat in range(len(lat)):
		for tick_lon in range(len(lon)):
			Pres[:,tick_lat,tick_lon]=ds['lev'].values*100#Pa

        # open the dataarray for temperature
	dt132=xr.open_dataset(filenamesT132[tick_f])

	# plotting Rn,Pb species with the right units:
	if species[tick_s]=='TRC_Rn':
		Temp = np.reshape(dt132['T'].values,[48,181,360])#K
		coef = 0.029/8.314*3.7e16# K mBq kg_Rn /sqm Pa kg_air
		becRn = Pres*coef/Temp*ds[species[tick_s]].sel(time = ds['time'].values[0]).values
		lon_avg132 = np.average(becRn,axis = 2)
        	pc = plt.pcolormesh(lat,lev[0:21],lon_avg132[0:21,:],cmap = mycmap,vmin=0,vmax=1e-30)
        	plt.colorbar(pc,label='mBq/sqm',shrink=0.80,pad=0.20,extend='both',orientation='horizontal')
        else:
		shitstain =np.reshape (ds[species[tick_s]].values,[48,181,360])#reshaping the data vector to not have that time dim
        	lon_avg132=np.average(shitstain,axis=2)#taking that average
		pc = plt.pcolormesh(lat,lev[0:21],lon_avg132[0:21,:],cmap = mycmap,vmin=0,vmax=1e-40)
        	plt.colorbar(pc,label='kg/kg',shrink=0.80,pad=0.20,extend='both',orientation='horizontal')
        plt.title(t[0]+'-'+t[1]+'-'+t[2]+ ' '+t[3]+':'+t[4]+' '+species[tick_s]+' zonally averaged 132 level') 
        plt.xlim([-90,90])
	plt.gca().invert_yaxis()
        plt.xlabel('latitude')
	plt.xlabel('latitude')
        plt.ylabel('Pressure [hPa]')
        #plt.savefig('images/L132/'+species[tick_s] + '_132lvl_'+t[0]+t[1]+t[2]+'_'+t[3]+t[4]+'.png')
        #plt.close(fig)
	#individual plot for 72 levels
        #fig1=plt.figure(figsize=[16,8])
	plt.subplot(1,2,2)
        dr = xr.open_dataset(filenames72[tick_f])

	# plotting crap with the right units
	if species[tick_s]=='TRC_Rn':
		Temp = np.reshape(dt132['T'].values,[48,181,360])#K
		coef = 0.029/8.314*3.7e16# K mBq kg_Rn /sqm Pa kg_air
		becRn = Pres*coef/Temp*dr[species[tick_s]].sel(time = dr['time'].values[0]).values
        	lon_avg72=np.average(becRn,axis=2)
		pc = plt.pcolormesh(lat,lev[0:21],lon_avg72[0:21,:],cmap = mycmap,vmin=0,vmax=1e-30)
        	plt.colorbar(pc,label='mBq/sqm',shrink=0.80,pad=0.20,extend='both',orientation='horizontal')
		
        else:
		shitstain =np.reshape(dr[species[tick_s]].values,[48,181,360])#reshaping the data vector to not have that time dim
        	lon_avg72=np.average(shitstain,axis=2)
        	pc = plt.pcolormesh(lat,lev[0:21],lon_avg72[0:21,:],cmap = mycmap,vmin=0,vmax=1e-40)
        	plt.colorbar(pc,label='kg/kg',shrink=0.80,pad=0.20,extend='both',orientation='horizontal')
        plt.title(t[0]+'-'+t[1]+'-'+t[2]+ ' '+t[3]+':'+t[4]+' '+ species[tick_s]+' zonally averaged 72 level')
	plt.xlim([-90,90])
	plt.gca().invert_yaxis()
        plt.xlabel('latitude')
        plt.ylabel('Pressure [hPa]') 
        plt.savefig('images/L132vL72/'+species[tick_s] + '_side_by_side_'+t[0]+t[1]+t[2]+'_'+t[3]+t[4]+ '.png')
	plt.close(fig)
	
	#percent difference plot for mixing ratio
        fig2=plt.figure(figsize=[16,8])
        perc_avg = np.divide((lon_avg72-lon_avg132),lon_avg72,out=np.zeros_like(lon_avg72-lon_avg132),where=lon_avg72!=0)*100
        pc = plt.pcolormesh(lat,lev[0:21],perc_avg[0:21,:],cmap = mycmap,vmin = -200, vmax = 200)
	plt.xlim([-90,90])
	plt.gca().invert_yaxis()
        plt.colorbar(pc,label='$\Delta $%',shrink=0.80,pad=0.20,extend='both',orientation='horizontal')
        plt.title(t[0]+'-'+t[1]+'-'+t[2]+ ' '+t[3]+':'+t[4]+' '+species[tick_s]+' % difference with 72 levels')
        plt.xlabel('latitude')
        plt.ylabel('Pressure [hPa]')
        plt.savefig('images/L132vL72/'+species[tick_s] + '_L72vL132_'+t[0]+t[1]+t[2]+'_'+t[3]+t[4]+ '.png')
