#introducion methods 
import torch
import matplotlib.pyplot as plt;
import torch.nn as nn
import numpy as np
from PIL import Image
# getting the data and trimming it
cddm = torch.load("cddm61_99.pt")
cddm = torch.narrow(cddm,1,0,164)
cddm = torch.narrow(cddm,2,0,151)
kdbi = torch.load("kdbi.pt")
kdbi = torch.narrow(kdbi,1,0,164)
kdbi = torch.narrow(kdbi,2,0,151)
extr = torch.load("extr61_99.pt")
extr = torch.narrow(extr,1,0,164)
extr = torch.narrow(extr,2,0,151)
crop = torch.load("croplandtiff_grown.pt")
crop = torch.swapaxes(crop,1,2)
ndvi = torch.load("ndvi.pt")
ndvi = torch.swapaxes(ndvi,0,1)
ndvi = nn.functional.normalize(ndvi)
pr = torch.load("pr61_99.pt")
pr = torch.narrow(pr,1,0,164)
pr = torch.narrow(pr,2,0,151)
print(crop.size())
print(pr[2]/pr[0])
#in the following line, cddm[1] represents rcp4 cddm future estimate, [0] is historic, [2] is rcp8
riskGauge = crop[0]*ndvi*((cddm[1]/cddm[0])+kdbi[1]/kdbi[0]+extr[1]/extr[0]+pr[1]/pr[0])/3
#the next line adds precipitation only if the normalized value is above a certain threshold
riskGauge = torch.where(abs(pr[1]/pr[0])>1.3,abs(pr[1]/pr[0])*1/4+riskGauge*3/4,riskGauge)
#these are just showing the plots
plt.imshow(riskGauge,vmin=0,vmax=1)
plt.title('Risk')
plt.colorbar(shrink=0.9)
plt.set_cmap("bwr")
plt.show()
riskGauge = riskGauge.detach().numpy()
im = Image.fromarray(riskGauge)
im.save("RCP4TotalRisk.tiff","TIFF")
"""
plt.imshow(crop[0],vmin=0,vmax=10)
plt.title('Risk')
plt.colorbar(shrink=0.9)
plt.set_cmap("bwr")
plt.show()
plt.imshow((riskGauge * crop[0]),vmin=0,vmax=10)
plt.title('Risk')
plt.colorbar(shrink=0.9)
plt.set_cmap("bwr")
plt.show()
"""