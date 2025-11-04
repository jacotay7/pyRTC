#%% 

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
# @jit(nopython=True, nogil=True, cache=True)
def computeSlopesSHWFSOptimNumba(image:np.ndarray, 
                                 slopes:np.ndarray, 
                                 unaberratedSlopes:np.ndarray, 
                                 threshold:np.float32, 
                                 spacing:np.float32,
                                 xvals:np.ndarray,
                                 offsetX:int, 
                                 offsetY:int,
                                 intN:int,
                                 ):
    
    # Convert image to the same dtype as unaberratedSlopes
    image = image.astype(np.float32)
    
    # Compute the number of sub-apertures
    numRegions = unaberratedSlopes.shape[1]

    # Loop over all regions
    for i in range(numRegions):
        for j in range(numRegions):
            # Compute where to start
            start_i = int(round(spacing * i)) + offsetY
            start_j = int(round(spacing * j)) + offsetX
            
            # Ensure we stay within the bounds of the image
            if start_j + intN <= image.shape[1] and start_i + intN <= image.shape[0]:
                #Create a local subimage around the lenslet spot
                sub_im = image[start_i:start_i + intN, start_j:start_j + intN]

                #loop through the sub image
                norm = np.float32(0)
                weightX = np.float32(0)
                weightY = np.float32(0)
                for m in range(intN):
                    for n in range(intN):
                        #If we are counting the pixel
                        if sub_im[m,n] > threshold:
                            #Add it to the normalization
                            norm += sub_im[m,n]
                            #Compute the X and Y centroids (before normalization)
                            weightX += xvals[m,n] * sub_im[m,n]
                            weightY += xvals[n,m] * sub_im[m,n]

                #If we have flux in the sub aperture
                if norm > 0:
                    #Normalize the centroids and remove the reference slope
                    slopes[i, j] = weightX/norm - unaberratedSlopes[i, j]
                    slopes[i + numRegions, j] = weightY/norm - \
                        unaberratedSlopes[i + numRegions, j]
                # if i == 5:    
                #     #If we have no flux slopes should be zero
                #     plt.imshow(sub_im)
                #     plt.plot(slopes[i, j] + intN/2, slopes[i + numRegions, j]+ intN/2, 'o', color = 'r')
                #     plt.show()
                #     if norm > 0:
                #         print("Value Counted", slopes[i, j], slopes[i+ numRegions, j])

    return slopes
#%%
OOPAO_files = ["4mag_raw.npy", "5mag_raw.npy", "4mag_raw_atm.npy", "5mag_raw_atm.npy"]
OOPAO_img = [np.load(OOPAO_files[i]) for i in range(len(OOPAO_files))]

for img in OOPAO_img:
    plotImg = img.copy().astype(float)
    # plotImg[plotImg < np.min(img)*-1] = np.nan
    plt.imshow(plotImg, vmin = 0)
    plt.show()
    plt.hist(img.flatten(), bins=np.arange(np.min(img), np.max(img)+2))
    plt.yscale('log')
    plt.show()
mask = np.load("slopemask.npy")
# %%


spacing = 16
num = 0
for img in OOPAO_img:
    threshold =  np.min(img)*-1/2
    # Compute the closest integer size of the sub-apertures
    intN = int(round(spacing))
    numRegions = img.shape[1]//spacing
    # Pre-compute the array to bias our centroid by
    xvals = np.arange(intN).astype(int) - intN // 2
    xvals, _ = np.meshgrid(xvals, xvals)
    xvals = xvals.astype(np.float32)
    image = img
    slopes = np.zeros((2*numRegions,numRegions), dtype=np.float32) 
    unaberratedSlopes = np.zeros_like(slopes) 
    offsetX = 0 
    offsetY = 0

    slopes =  computeSlopesSHWFSOptimNumba(image, 
                                 slopes, 
                                 unaberratedSlopes, 
                                 threshold, 
                                 spacing,
                                 xvals,
                                 offsetX, 
                                 offsetY,
                                 intN,
                                 )
    slopes[mask] = 0
    plt.imshow(slopes)
    plt.colorbar()
    plt.show()
    plt.hist(slopes.flatten())
    plt.show()
    num += 1
#%%