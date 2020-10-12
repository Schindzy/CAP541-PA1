import math
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters

#This class implements a simple canny edged detector algorithm
class CannyEdgeDetector:

    #Value used to mark a pixel as an edge
    EDGE_PIXEL_VALUE = 255

    #Value used to mark a pixel that COULD be part of an edge, if connected to another edge pixel
    IN_BETWEEN_PIXEL_VALUE=50

    #Required constructor inputs are the sigma level for the gaussian kernel, the kernel length,
    # and the low and high thresholds expressed as ratios of the max gradient resonse
    def __init__(self, sigma, kernel_size, low_threshold, high_threshold):
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        return

    #Main entry point of the function
    def detect(self, img, save=False, output_path=""):

        #Compute 1d Guassian mask with the specified specified size and sigma level
        g = self.gaussian_kernel_1d(self.kernel_size, self.sigma)

        #Compute the derivative of the gaussian via convolution
        g_prime = np.convolve(g, [1, -1])

        #Apply the mask in the x and y directions
        ix = self.convolve_1_by_row(img, g)
        iy = self.convolve_1_by_col(img, np.transpose(g))

        #Use derivative of gaussian to compute gradients
        ix_prime = self.convolve_1_by_row(ix, g_prime)
        iy_prime = self.convolve_1_by_col(iy, np.transpose(g_prime))

        #Compute the magnitude and orientation of the gradients
        magnitude = np.hypot(ix_prime, iy_prime)
        magnitude = magnitude / magnitude.max() * 255 #Gotta normalize it for plotting
        orientation = np.arctan2(iy_prime, ix_prime)

        # #Apply non maximum supression
        non_max_img = self.non_max_suppression(magnitude, orientation)

        new_image = self.threshold(non_max_img, 5, 20, 50)

        final = self.hysteresis(new_image, 50)

        # Apply hysteresis thresholding to the image
        # final = filters.apply_hysteresis_threshold(non_max_img, self.low_threshold, self.high_threshold)
        # final = self.hysteresis_threshold(non_max_img, self.low_threshold, self.high_threshold)

        if(save):
            plt.imsave(output_path+"ix.jpg", ix, cmap='gray')
            plt.imsave(output_path+"iy.jpg", iy, cmap='gray')
            plt.imsave(output_path+"ix_prime.jpg", ix_prime, cmap='gray')
            plt.imsave(output_path+"iy_prime.jpg", iy_prime, cmap='gray')
            plt.imsave(output_path+"magnitude.jpg", magnitude, cmap='gray')
            plt.imsave(output_path+"final.jpg", final, cmap='gray')

        #Apply hysteresis thresholding to the image
        return final

    #Compute a 1d gaussian mask of the given length and sigma level
    #Source: https://en.wikipedia.org/wiki/Gaussian_blur
    @staticmethod
    def gaussian_kernel_1d(size, sigma):
        size = int(size) // 2
        x = np.mgrid[-size:size + 1]

        normal = 1 / math.sqrt((2.0 * np.pi * sigma ** 2))
        g = np.exp(-((x ** 2) / (2.0 * sigma ** 2))) * normal
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

        #Convolve in the y direction
        y = np.zeros(img.shape, dtype=img.dtype)
        for col in range(img.shape[1]):
            y[:, col] = np.convolve(img[:, col], kernel, mode='same')

        return y

    #Implementation of non maximum supression algorithm from class
    #Inputs: a matrix reprsenting the magnitude of the gradient response
    #        a matrix representing the orientation of the gradient response
    #Output: Matrix with containing only local maximum edges
    # @staticmethod
    # def non_max_suppression(magnitude, orientation):
    #     m, n = magnitude.shape
    #     output = np.zeros((m, n), dtype=np.int32)
    #     angle = orientation * 180. / np.pi
    #     angle[angle < 0] += 180
    #
    #     for i in range(1, m - 1):
    #         for j in range(1, n - 1):
    #             q = 255
    #             r = 255
    #
    #             # angle 0
    #             if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
    #                 q = magnitude[i, j + 1]
    #                 r = magnitude[i, j - 1]
    #             # angle 45
    #             elif 22.5 <= angle[i, j] < 67.5:
    #                 q = magnitude[i + 1, j - 1]
    #                 r = magnitude[i - 1, j + 1]
    #             # angle 90
    #             elif 67.5 <= angle[i, j] < 112.5:
    #                 q = magnitude[i + 1, j]
    #                 r = magnitude[i - 1, j]
    #             # angle 135
    #             elif 112.5 <= angle[i, j] < 157.5:
    #                 q = magnitude[i - 1, j - 1]
    #                 r = magnitude[i + 1, j + 1]
    #
    #             if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
    #                 output[i, j] = magnitude[i, j]
    #             else:
    #                 output[i, j] = 0
    #     return output

    @staticmethod
    def non_max_suppression(gradient_magnitude, gradient_direction):
        image_row, image_col = gradient_magnitude.shape

        output = np.zeros(gradient_magnitude.shape)

        PI = 180

        for row in range(1, image_row - 1):
            for col in range(1, image_col - 1):
                direction = gradient_direction[row, col]

                # (0 - PI/8 and 15PI/8 - 2PI)
                if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                    before_pixel = gradient_magnitude[row, col - 1]
                    after_pixel = gradient_magnitude[row, col + 1]

                elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                    before_pixel = gradient_magnitude[row + 1, col - 1]
                    after_pixel = gradient_magnitude[row - 1, col + 1]

                elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                    before_pixel = gradient_magnitude[row - 1, col]
                    after_pixel = gradient_magnitude[row + 1, col]

                else:
                    before_pixel = gradient_magnitude[row - 1, col - 1]
                    after_pixel = gradient_magnitude[row + 1, col + 1]

                if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                    output[row, col] = gradient_magnitude[row, col]

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
    def hysteresis(image, weak):
        image_row, image_col = image.shape

        top_to_bottom = image.copy()

        for row in range(1, image_row):
            for col in range(1, image_col):
                if top_to_bottom[row, col] == weak:
                    if top_to_bottom[row, col + 1] == 255 or top_to_bottom[row, col - 1] == 255 or top_to_bottom[
                        row - 1, col] == 255 or top_to_bottom[
                        row + 1, col] == 255 or top_to_bottom[
                        row - 1, col - 1] == 255 or top_to_bottom[row + 1, col - 1] == 255 or top_to_bottom[
                        row - 1, col + 1] == 255 or top_to_bottom[
                        row + 1, col + 1] == 255:
                        top_to_bottom[row, col] = 255
                    else:
                        top_to_bottom[row, col] = 0

        bottom_to_top = image.copy()

        for row in range(image_row - 1, 0, -1):
            for col in range(image_col - 1, 0, -1):
                if bottom_to_top[row, col] == weak:
                    if bottom_to_top[row, col + 1] == 255 or bottom_to_top[row, col - 1] == 255 or bottom_to_top[
                        row - 1, col] == 255 or bottom_to_top[
                        row + 1, col] == 255 or bottom_to_top[
                        row - 1, col - 1] == 255 or bottom_to_top[row + 1, col - 1] == 255 or bottom_to_top[
                        row - 1, col + 1] == 255 or bottom_to_top[
                        row + 1, col + 1] == 255:
                        bottom_to_top[row, col] = 255
                    else:
                        bottom_to_top[row, col] = 0

        right_to_left = image.copy()

        for row in range(1, image_row):
            for col in range(image_col - 1, 0, -1):
                if right_to_left[row, col] == weak:
                    if right_to_left[row, col + 1] == 255 or right_to_left[row, col - 1] == 255 or right_to_left[
                        row - 1, col] == 255 or right_to_left[
                        row + 1, col] == 255 or right_to_left[
                        row - 1, col - 1] == 255 or right_to_left[row + 1, col - 1] == 255 or right_to_left[
                        row - 1, col + 1] == 255 or right_to_left[
                        row + 1, col + 1] == 255:
                        right_to_left[row, col] = 255
                    else:
                        right_to_left[row, col] = 0

        left_to_right = image.copy()

        for row in range(image_row - 1, 0, -1):
            for col in range(1, image_col):
                if left_to_right[row, col] == weak:
                    if left_to_right[row, col + 1] == 255 or left_to_right[row, col - 1] == 255 or left_to_right[
                        row - 1, col] == 255 or left_to_right[
                        row + 1, col] == 255 or left_to_right[
                        row - 1, col - 1] == 255 or left_to_right[row + 1, col - 1] == 255 or left_to_right[
                        row - 1, col + 1] == 255 or left_to_right[
                        row + 1, col + 1] == 255:
                        left_to_right[row, col] = 255
                    else:
                        left_to_right[row, col] = 0

        final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right

        final_image[final_image > 255] = 255

        return final_image

    # @staticmethod
    # def hysteresis_threshold(img, low_threshold_ratio, high_threshold_ratio):
    #
    #     #First categorize all the responses as edge, non edge, or in between
    #     high_threshold = img.max() * high_threshold_ratio
    #     low_threshold = high_threshold * low_threshold_ratio
    #
    #     m, n = img.shape
    #     response = np.zeros((m, n), dtype=np.int32)
    #
    #     weak = np.int32(CannyEdgeDetector.IN_BETWEEN_PIXEL_VALUE)
    #     strong = np.int32(CannyEdgeDetector.EDGE_PIXEL_VALUE)
    #
    #     #Response is strong, mark as edge
    #     strong_i, strong_j = np.where(img >= high_threshold)
    #
    #     #Response is in the middle, mark as inbetween
    #     weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))
    #
    #     response[strong_i, strong_j] = strong
    #     response[weak_i, weak_j] = weak
    #
    #     #Finally loop, overall the inbetween pixels, and mark any connected, directly or indirectly as edges
    #     for i in range(1, m - 1):
    #         for j in range(1, n - 1):
    #             if response[i, j] == CannyEdgeDetector.IN_BETWEEN_PIXEL_VALUE:
    #                 try:
    #                     if ((response[i + 1, j - 1] == CannyEdgeDetector.EDGE_PIXEL_VALUE) or (response[i + 1, j] == CannyEdgeDetector.EDGE_PIXEL_VALUE) or (response[i + 1, j + 1] == CannyEdgeDetector.EDGE_PIXEL_VALUE)
    #                             or (response[i, j - 1] == CannyEdgeDetector.EDGE_PIXEL_VALUE) or (response[i, j + 1] == CannyEdgeDetector.EDGE_PIXEL_VALUE)
    #                             or (response[i - 1, j - 1] == CannyEdgeDetector.EDGE_PIXEL_VALUE) or (response[i - 1, j] == CannyEdgeDetector.EDGE_PIXEL_VALUE) or (
    #                                     response[i - 1, j + 1] == CannyEdgeDetector.EDGE_PIXEL_VALUE)):
    #                         response[i, j] = CannyEdgeDetector.EDGE_PIXEL_VALUE
    #                     else:
    #                         response[i, j] = 0
    #                 except IndexError:
    #                     pass
    #
    #     return response
