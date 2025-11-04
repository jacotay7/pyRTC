
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm

# Parameters
signal_mean = 20 # Mean number of photons (signal)
read_noise_std = 1.33 # Read noise standard deviation

#Approximate effect of spots
N = 400*400  # Number of pixels total in image
pixelFractionAtMax = 0.01/2 #Tune these
numBrightnessInModel = 50  #Tune these
fracs = np.linspace(pixelFractionAtMax,
                    pixelFractionAtMax*3,
                    numBrightnessInModel)
#Amount to reduce spot brightness per fraction decrease
reductionRatio = 2 #Tune this

scale = 1

dark_subtracted_image = np.array([])
for frac in fracs:
    signal = np.round(np.random.poisson(signal_mean*scale, size=int(N*frac))).astype(int)
    signal += np.round(np.random.normal(0, read_noise_std, size=signal.size)).astype(int)
    scale /= reductionRatio
    dark_subtracted_image = np.concatenate([dark_subtracted_image,signal])
read_noise = np.round(np.random.normal(0, read_noise_std, size=N - dark_subtracted_image.size)).astype(int)  # Gaussian read noise
dark_subtracted_image = np.concatenate([dark_subtracted_image,read_noise])# Plot histogram of the pixel values
plt.hist(dark_subtracted_image,
          bins=np.arange(np.min(dark_subtracted_image),np.max(dark_subtracted_image)),
          density=True, alpha=0.7, label="Dark-subtracted image")

plt.xlabel("Pixel Value")
plt.ylabel("Probability Density")
plt.legend()
plt.yscale('log')
plt.axvline(x = np.min(dark_subtracted_image), color = 'r')
plt.axvline(x = np.min(dark_subtracted_image)*-1, color = 'r')
plt.show()
print(np.std(dark_subtracted_image[dark_subtracted_image < -1*np.min(dark_subtracted_image)]))
# %%
