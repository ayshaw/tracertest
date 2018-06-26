import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
plt.switch_backend('agg')
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
mycmap = discrete_cmap(13,'spectral')
t = ['03','04','05','06','07','08']
plt.figure(figsize = [40,20])
for tick_t in range(len(t)):
	sum_emis=np.zeros([181,360])
	filestring=['c90_RnPbBe_L132/holding/geosgcm_gcc_p/c90_L132.geosgcm_gcc_p.2013' + t[tick_t]  + '*.nc4']
	files_L132=glob.glob('c90_RnPbBe_L132/72timestep/c90_L132.geosgcm_gcc_p.2013' + t[tick_t]  + '*.nc4')
	files_L72=glob.glob('c90_RnPbBe_L72/holding/geosgcm_gcc_p/c90_L72.geosgcm_gcc_p.2013' + t[tick_t]  + '*.nc4')
	print files_L132
	print files_L72 
	for tick_f in range(len(files_L132)):
		dr = xr.open_dataset(files_L72[tick_f])
		ds = xr.open_dataset(files_L132[tick_f])
		emis_L132 = np.reshape(ds['EMIS_Be7'].values,[181, 360])
		emis_L72 = np.reshape(dr['EMIS_Be7'].values,[181,360])
		Demis = (emis_L132-emis_L72)/emis_L72*100
		sum_emis += emis_L132
	lat = np.linspace(-90,90,181)
	lon = np.linspace(-180,179,360)
	plt.hold(True)
	ax1=plt.subplot(2,3,tick_t+1,projection=ccrs.PlateCarree())
	ax1.coastlines()
	aemis=sum_emis/len(files_L72)
	print aemis
	pc=ax1.pcolormesh(lon,lat,aemis,vmin=1e-23,vmax=1.5e-23)
	if t[tick_t]=='08':
		plt.colorbar(pc,cmap=mycmap,shrink=0.50,extend='both')

plt.savefig('images/132emismonthly__.png')
