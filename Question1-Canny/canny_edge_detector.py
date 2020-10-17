import math
import numpy as np
import matplotlib.pyplot as plt

# This class implements a simple canny edged detector algorithm
class CannyEdgeDetector:

    # Value used to mark a pixel as an edge
    EDGE_PIXEL_VALUE = 255

    # Value used to mark a pixel that COULD be part of an edge, if connected to another edge pixel
    IN_BETWEEN_PIXEL_VALUE = 50

    # Required constructor inputs are the sigma level for the gaussian kernel, the kernel length,
    # and the low and high thresholds expressed as ratios of the max gradient response
    def __init__(self, sigma, kernel_size, low_threshold, high_threshold):
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        return

    # Main entry point of the function
    def detect(self, img, save=False, output_path=""):

        # Compute 1d Gaussian mask with the specified specified size and sigma level
        g = self.gaussian_kernel_1d(self.kernel_size, self.sigma)

        # Compute the derivative of the gaussian via convolution
        g_prime = np.convolve(g, [1, 0, -1])

        # Apply the mask in the x and y directions
        ix = self.convolve_1_by_row(img, g)
        iy = self.convolve_1_by_col(img, np.transpose(g))

        # Use derivative of gaussian to compute gradients
        ix_prime = self.convolve_1_by_row(ix, g_prime)
        iy_prime = self.convolve_1_by_col(iy, np.transpose(g_prime))

        # Compute the magnitude and orientation of the gradients
        magnitude = np.hypot(ix_prime, iy_prime)
        magnitude = magnitude / magnitude.max() * 255  # Gotta normalize it for plotting
        orientation = np.arctan2(iy_prime, ix_prime)

        # Apply non maximum suppression
        non_max_img = self.non_max_suppression(magnitude, orientation)

        # Apply hysteresis thresholding to the image
        final = self.hysteresis_threshold(non_max_img, self.low_threshold, self.high_threshold)

        if save:
            plt.imsave(output_path+"ix.jpg", ix, cmap='gray')
            plt.imsave(output_path+"iy.jpg", iy, cmap='gray')
            plt.imsave(output_path+"ix_prime.jpg", ix_prime, cmap='gray')
            plt.imsave(output_path+"iy_prime.jpg", iy_prime, cmap='gray')
            plt.imsave(output_path+"magnitude.jpg", magnitude, cmap='gray')
            plt.imsave(output_path+"final.jpg", final, cmap='gray')

        # Apply hysteresis thresholding to the image
        return final

    # Compute a 1d gaussian mask of the given length and sigma level
    # Source: https://en.wikipedia.org/wiki/Gaussian_blur
    @staticmethod
    def gaussian_kernel_1d(size, sigma):
        size = int(size) // 2
        x = np.mgrid[-size:size + 1]

        normal = 1 / math.sqrt((2.0 * np.pi * sigma ** 2))
        g = np.exp(-((x ** 2) / (2.0 * sigma ** 2))) * normal
        g = g / np.sum(g)

        return g

    # Helper function that performs row and column wise convolution of a given 1d mask across a provided image
    # Input: image matrix, 1d mask
    # Outputs: matrices representing row and column-wise convolution
    @staticmethod
    def convolve_1_by_row(img, kernel):

        # Convolve in the x direction
        x = np.zeros(img.shape, dtype=img.dtype)
        for row in range(img.shape[0]):
            x[row, :] = np.convolve(img[row, :], kernel, mode='same')

        return x

    # Helper function that performs row and column wise convolution of a given 1d mask across a provided image
    # Input: image matrix, 1d mask
    # Outputs: matrices representing row and column-wise convolution
    @staticmethod
    def convolve_1_by_col(img, kernel):

        # Convolve in the y direction
        y = np.zeros(img.shape, dtype=img.dtype)
        for col in range(img.shape[1]):
            y[:, col] = np.convolve(img[:, col], kernel, mode='same')

        return y

    # Implementation of non maximum suppression algorithm from class
    # Inputs: a matrix representing the magnitude of the gradient response
    #        a matrix representing the orientation of the gradient response
    # Output: Matrix with containing only local maximum edges
    @staticmethod
    def non_max_suppression(magnitude, orientation):
        m, n = magnitude.shape
        output = np.zeros((m, n), dtype=np.int32)
        angle = orientation * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, m - 1):
            for j in range(1, n - 1):
                q = 255
                r = 255

                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]

                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]

                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]

                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                    output[i, j] = magnitude[i, j]
                else:
                    output[i, j] = 0
        return output

    @staticmethod
    def threshold(image, low, high, weak):
        output = np.zeros(image.shape)

        strong = 255

        strong_row, strong_col = np.where(image >= high)
        weak_row, weak_col = np.where((image <= high) & (image >= low))

        output[strong_row, strong_col] = strong
        output[weak_row, weak_col] = weak

        return output

    @staticmethod
    def hysteresis_threshold(img, low_threshold_ratio, high_threshold_ratio):

        # First categorize all the responses as edge, non edge, or in between
        high_threshold = img.max() * high_threshold_ratio
        low_threshold = high_threshold * low_threshold_ratio

        m, n = img.shape
        response = np.zeros((m, n), dtype=np.int32)

        weak = np.int32(CannyEdgeDetector.IN_BETWEEN_PIXEL_VALUE)
        strong = np.int32(CannyEdgeDetector.EDGE_PIXEL_VALUE)

        # Response is strong, mark as edge
        strong_i, strong_j = np.where(img >= high_threshold)

        # Response is in the middle, mark as in between
        weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

        response[strong_i, strong_j] = strong
        response[weak_i, weak_j] = weak

        # Finally loop, overall the in between pixels, and mark any connected, directly or indirectly as edges
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if response[i, j] == CannyEdgeDetector.IN_BETWEEN_PIXEL_VALUE:
                    try:
                        if ((response[i + 1, j - 1] == CannyEdgeDetector.EDGE_PIXEL_VALUE) or (response[i + 1, j] == CannyEdgeDetector.EDGE_PIXEL_VALUE) or (response[i + 1, j + 1] == CannyEdgeDetector.EDGE_PIXEL_VALUE)
                                or (response[i, j - 1] == CannyEdgeDetector.EDGE_PIXEL_VALUE) or (response[i, j + 1] == CannyEdgeDetector.EDGE_PIXEL_VALUE)
                                or (response[i - 1, j - 1] == CannyEdgeDetector.EDGE_PIXEL_VALUE) or (response[i - 1, j] == CannyEdgeDetector.EDGE_PIXEL_VALUE) or (
                                        response[i - 1, j + 1] == CannyEdgeDetector.EDGE_PIXEL_VALUE)):
                            response[i, j] = CannyEdgeDetector.EDGE_PIXEL_VALUE
                        else:
                            response[i, j] = 0
                    except IndexError:
                        pass

        return response
