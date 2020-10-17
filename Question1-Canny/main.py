
import matplotlib.image as im
from .canny_edge_detector import CannyEdgeDetector
from skimage.color import rgb2gray

img = im.imread("Sturdy_Lizard_Colored.jpg")
img_g = rgb2gray(img)

detector = CannyEdgeDetector(sigma=1, kernel_size=3, low_threshold=0.05, high_threshold=0.15)
img_final = detector.detect(img_g, True, "")
