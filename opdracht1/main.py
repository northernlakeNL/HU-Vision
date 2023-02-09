import numpy as np
import matplotlib.pyplot as plt
from skimage import io,color
from skimage.viewer import ImageViewer

image = io.imread('./image.jpg')
image_copy = image.copy()
image_hsv = color.rgb2hsv(image)
image_filter = image[:,:,0] > 150

mask_down = image_hsv[:,:,0] > 0.05
mask_up = image_hsv[:,:,0] < 0.5
satu = image_hsv[:,:,1] > 0.3
mask = mask_up*mask_down*satu

red = image_copy[:,:,0]*mask
green = image_copy[:,:,1]*mask
blue = image_copy[:,:,2]*mask

stacked = np.dstack((red,green,blue))

viewer = ImageViewer(stacked)
viewer.show()

fig, (ax1,ax2) = plt.subplots(1,2, sharex=True, sharey = True, figsize=(10,5))


ax1.hist(image.ravel(), bins = 256, color = 'orange')
ax1.hist(image[:,:,0].ravel(), bins = 256, color = 'red', alpha = 0.5)
ax1.hist(image[:,:,1].ravel(), bins = 256, color = 'green', alpha = 0.5)
ax1.hist(image[:,:,2].ravel(), bins = 256, color = 'blue', alpha = 0.5)

ax2.hist(stacked.ravel(), bins = 256, color = 'orange')
ax2.hist(stacked[:,:,0].ravel(), bins = 256, color = 'red', alpha = 0.5)
ax2.hist(stacked[:,:,1].ravel(), bins = 256, color = 'green', alpha = 0.5)
ax2.hist(stacked[:,:,2].ravel(), bins = 256, color = 'blue', alpha = 0.5)

for ax in ax1, ax2:
    ax.grid(True)
    ax.label_outer()
    ax.set_ylim([0, 50000])
    ax.set_xlim([0,256])

ax1.set_xlabel('Intensity')
ax1.set_ylabel('Count')
ax2.legend(['Total', 'Red', 'Green', 'Blue'])

fig.suptitle('Color Usage of image')
ax1.set_title('Full Image')
ax2.set_title('Filtered Image')

plt.show()