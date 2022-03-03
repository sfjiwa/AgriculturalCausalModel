import rasterio
import numpy as np
import os
import torch
import glob
rootdir = "data_latest_future/kbdifuture"

for subdir, dirs, files in os.walk(rootdir):
	for file in files:
		#print(file[:len(file)-4])#remove extension
		"""b = os.path.join(subdir,file)
		with rasterio.open(b, 'r') as ds:
			arr = ds.read()  # read all raster values
			a = np.array(arr)
			#t = torch.from_numpy(a)
			print(t)
			torch.save(t, file[:len(file)-4]+".pt")
			"""
list_of_files = sorted( filter( os.path.isfile,
                        glob.glob(rootdir + '/**/*', recursive=True) ) )
# Iterate over sorted list of files and print the file paths 
# one by one.
aa = 0
for file in list_of_files:
    if aa == 0:
        print(0)
        print(file) 
        with rasterio.open(file, 'r') as ds:
            arr = ds.read()  # read all raster values
            c = np.array(arr)
        aa = 1
    else:
        with rasterio.open(file, 'r') as ds:
             arr = ds.read()  # read all raster values
             a = np.array(arr)
             c = np.append(c,a,0)
t = torch.from_numpy(c)
print(t.size())
torch.save(t, "kbdif.pt")
