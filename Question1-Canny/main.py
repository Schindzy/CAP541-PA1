import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from .canny_edge_detector import CannyEdgeDetector as ced
from skimage.color import rgb2gray



for i in range(1, 4):
    img = mpimg.imread("Sturdy_Lizard_Colored.jpg")
    img_g = rgb2gray(img)
    detector = ced(sigma=i, kernel_size=3, low_threshold=0.1, high_threshold=0.25)
    path = "Testing/lizard/testing_sigma"+str(i)
    if(~os.path.isdir(path)):
        os.mkdir(path)
    img_final = detector.detect(img_g, True, path+"/")

# Plot it
# plt.imshow(img_final, 'gray')
# plt.show()
img