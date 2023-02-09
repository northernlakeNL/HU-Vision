import numpy as np
import matplotlib.pyplot as plt
from skimage import data,io,color
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

plt.hist(image.ravel(), bins = 256, color = 'orange')
plt.hist(image[:,:,0].ravel(), bins = 256, color = 'red', alpha = 0.5)
plt.hist(image[:,:,1].ravel(), bins = 256, color = 'green', alpha = 0.5)
plt.hist(image[:,:,2].ravel(), bins = 256, color = 'blue', alpha = 0.5)

plt.xlabel('Intensity')
plt.ylabel('Count')
plt.legend(['Total', 'Red', 'Green', 'Blue'])

plt.hist(stacked.ravel(), bins = 256, color = 'orange')
plt.hist(stacked[:,:,0].ravel(), bins = 256, color = 'red', alpha = 0.5)
plt.hist(stacked[:,:,1].ravel(), bins = 256, color = 'green', alpha = 0.5)
plt.hist(stacked[:,:,2].ravel(), bins = 256, color = 'blue', alpha = 0.5)

plt.xlabel('Intensity')
plt.ylabel('Count')
plt.legend(['Total', 'Red', 'Green', 'Blue'])

plt.show()