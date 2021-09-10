#from proj2.align_image_code import align_images
from matplotlib.pyplot import get_current_fig_manager
import cv2
import scipy as sp
import scipy.signal as sps
import scipy.ndimage.interpolation as spi
import skimage.io as skio
import numpy as np
import math
from align_image_code import align_images
from skimage.color import rgb2gray

def cameraman():
    return cv2.imread("img/cameraman.png", cv2.IMREAD_GRAYSCALE)

derek = skio.imread("img/derek.jpg", 0)
nutmeg = skio.imread("img/nutmeg.jpg", 0)
apple = skio.imread("img/apple.jpeg", 0)
orange = skio.imread("img/orange.jpeg", 0)
mask = skio.imread("img/mask.jpg", 0)
taj = skio.imread("img/taj.jpg")

Dx = [[1, -1], [0, 0]]
Dy = [[1, 0], [-1, 0]]

def show(img):
    skio.imshow(img)
    skio.show()

def save(img, name):
    skio.imsave(name, img)

def convolve_x(img):
    """ Convolve IMG in the x direction """
    return sps.convolve2d(img, Dx)

def convolve_y(img):
    """ Convolve IMG in the x direction """
    return sps.convolve2d(img, Dy)

def convolve(img):
    """ Convolve IMG in both the x & y direction """
    convolution_x = sps.convolve2d(img, Dx, mode="same")
    convolution_y = sps.convolve2d(img, Dy, mode="same")

    return convolution_x, convolution_y

def gaussian_convolution_params(im, alpha, sigma):
    """ Convolve IM with the Gaussian described by (ALPHA, SIGMA) and crop off the edges. """
    im = sps.convolve2d(rgb2gray(im), gaussian(alpha, sigma), boundary='symm')
    return im[(alpha - 1) // 2 : len(im) - (alpha // 2), ((alpha - 1) // 2) : im.shape[1] - (alpha // 2)]

def gaussian_convolution(im, g):
    """ Convolve IM with the Gaussian described by G and crop off the edges. """
    im = sps.convolve2d(rgb2gray(im), g, boundary='symm')
    return im[(len(g) - 1) // 2 : len(im) - (len(g) // 2), ((len(g) - 1) // 2) : im.shape[1] - ((len(g)) // 2)]


def threshold_mask(x, y, threshold_x, threshold_y):
    mask_x = (abs(x) > threshold_x).astype('float32')
    mask_y = (abs(y) > threshold_y).astype('float32')
    return mask_x, mask_y

def edge_detection(img, threshold_x, threshold_y, debug=False):

    convolution_x, convolution_y = convolve(img)
    mask_x, mask_y = threshold_mask(convolution_x, convolution_y, threshold_x, threshold_y)

    if debug:
        skio.imshow(convolution_x)
        skio.show()

        skio.imshow(convolution_x)
        skio.show()

    return mask_x + mask_y

def normalize(img):
    return (img - np.amin(img)) / (np.amax(img) - np.amin(img))

def gaussian(alpha, sigma):
    g = cv2.getGaussianKernel(alpha, sigma)
    return np.outer(g, g.T)

def gaussian_of_image_then_convolution(img, threshold_x, threshold_y, alpha, sigma):
    g = gaussian(alpha, sigma)
    convolution_g = sps.convolve2d(img, g).astype('float32')
    return normalize(edge_detection(convolution_g, threshold_x, threshold_y, False))

def convolve_gaussian_then_image(img, threshold_x, threshold_y, alpha, sigma):
    g = gaussian(alpha, sigma)
    g = convolve_x(g)
    g = convolve_y(g)
    show(g)
    img = sps.convolve2d(img, g).astype('float32')
    img = (abs(img) > 7).astype('float32')
    
    return normalize(img).astype('float32')

# PART 1.2
# The image's edges before & after taking the Gaussian.
save(edge_detection(cameraman(), 48, 40), "out/cameraman_edges_raw.jpg")

# Take the Gaussian of the image BEFORE convolving with DX and DY.
show(normalize(gaussian_of_image_then_convolution(cameraman(), 28, 28, 4, 1)))

#save(gaussian_of_image_then_convolution(cameraman(), 40, 25, 4, 1), "out/cameraman_edges_gaussian.jpg")

# Convolve the Gaussian with DX and DY, THEN convolve the result with the image.
show(convolve_gaussian_then_image(cameraman(), 40, 25, 4, 1))

def unsharp_mask(img, alpha = 12, sigma = 6):
    #taj = cv2.imread("img/taj.jpg", cv2.IMREAD_GRAYSCALE)
    g = gaussian(alpha, sigma)

    """impulse = [[0] * alpha] * alpha
    impulse[(alpha - 1) // 2][(alpha - 1) // 2] = (impulse[(alpha - 1) // 2][(alpha - 1) // 2] + 0.707) ** 0.5
    impulse[(alpha + 1) // 2][(alpha - 1) // 2] = (impulse[(alpha + 1) // 2][(alpha - 1) // 2] + 0.707) ** 0.5
    impulse[(alpha - 1) // 2][(alpha + 1) // 2] = (impulse[(alpha - 1) // 2][(alpha + 1) // 2] + 0.707) ** 0.5
    impulse[(alpha + 1) // 2][(alpha + 1) // 2] = (impulse[(alpha + 1) // 2][(alpha + 1) // 2] + 0.707) ** 0.5

    print(impulse)"""

    #blur = sps.convolve2d(img, g)
    blur = gaussian_convolution(img, g)
    #show(blur)
    print(blur.shape, img.shape)
    return rgb2gray(img) - blur

def sharpen(img):
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]

    # Let's try subtracting a gaussian with the same parameters from the blurred version to get the sharp edges.
    unsharp_r = unsharp_mask(r, 12, 3)
    unsharp_g = unsharp_mask(g, 12, 3)
    unsharp_b = unsharp_mask(b, 12, 3)

    sharpened_version = np.dstack([r, g, b]) + np.dstack([unsharp_r, unsharp_g, unsharp_b])
    return sharpened_version

def resharpen_blurred_image(img):

    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    
    blur_r = gaussian_convolution_params(r, 14, 5)
    blur_g = gaussian_convolution_params(g, 14, 5)
    blur_b = gaussian_convolution_params(b, 14, 5)

    # Let's try subtracting a gaussian with the same parameters from the blurred version to get the sharp edges.
    unsharp_r = unsharp_mask(blur_r, 14, 5)
    unsharp_g = unsharp_mask(blur_g, 14, 5)
    unsharp_b = unsharp_mask(blur_b, 14, 5)

    resharpened_version = np.dstack([blur_r, blur_g, blur_b]) + np.dstack([unsharp_r, unsharp_g, unsharp_b])
    return resharpened_version

save(sharpen(taj), "out/taj_sharpened.jpg")
#save(resharpen_blurred_image(taj), "out/taj_blurred_sharpened.jpg")

def hybrid(lpi, hpi, cutoff_frequency):
    lpi, hpi = align_images(derek, nutmeg)
    print(hpi.shape)
    alpha = 17
    sigma = 13
    hpi = normalize(unsharp_mask(hpi, alpha, sigma))

    lpi = gaussian_convolution_params(lpi, alpha, sigma)

    #lpi = lpi[(alpha - 1) // 2 : len(lpi) - alpha // 2, (alpha - 1) : len(lpi) - (alpha // 2)]
    hybrid = ((hpi + lpi) / 2) 
    
    return hybrid

def gaussian_stack(image, conv_matrix, levels = 5):
    stack = [image]
    for i in range(0, levels):
        stack.append(gaussian_convolution(stack[i - 1], conv_matrix))
        #show(stack[i])
    return stack

def stacks(image, conv_matrix, levels = 5):
    gaussian_stack = [image]
    print(image.shape)
    laplacian_stack = [image]
    for i in range(1, levels):
        #print(gaussian_stack[i - 1][:,:,0].shape)
        conv = np.dstack([gaussian_convolution(gaussian_stack[i - 1][:,:,color], conv_matrix) for color in range(3)])
        #conv = gaussian_convolution(gaussian_stack[i - 1], conv_matrix)
        gaussian_stack.append(conv)
        laplacian_stack.append((gaussian_stack[i - 1][:,:,:] - conv))
        #print(min(laplacian_stack[i - 1][:,:]))
    return gaussian_stack, laplacian_stack
        
def blend(A, B, R, size, sigma, gray=False):
    # Easy way to simulate grayscale without changing the other methods
    if gray:
        A = rgb2gray(A)
        A = np.dstack([A, A, A])
        B = rgb2gray(B)
        B = np.dstack([B, B, B])
        if np.max(A) <= 1:
            A *= 255
        if np.max(B) <= 1:
            B *= 255
    # Apply a mask to make sure the image is 1 or 0
    #R = R[:,:] > 0.2
    
    # Easy way to simulate grayscale without changing the other methods

    #A = normalize(A)
    #B = normalize(B)
    R = normalize(R)
    #print(A[len(R) - 38, R.shape[1] - 16])
    #print(B[len(R) - 38, R.shape[1] - 16])
    #print(R[len(R) - 38, R.shape[1] - 16])

    g = gaussian(size, sigma)
    _, a_lap = stacks(A, g)
    _, b_lap = stacks(B, g)
    mask_gaussian, _ = stacks(R, g)
    #print(mask_gaussian[2][len(R) - 38, R.shape[1] - 16])

    ones = np.ones((mask_gaussian[0].shape[0], mask_gaussian[0].shape[1]))
    ones = np.dstack([ones, ones, ones])

    LS = []
    for i in range(1, len(a_lap)):
        LS.append(((mask_gaussian[i] * a_lap[i]) + ((ones - mask_gaussian[i]) * b_lap[i])))
        #LS.append(((ones - mask_gaussian[i]) * a_lap[i]) + (mask_gaussian[i] * b_lap[i]))

    LS = sum(LS)

    return normalize(LS)


#camera = skio.imread("img/cameraman.png", 0)
#g_camera = gaussian_convolution(camera, g)
#print(camera.shape, g_camera.shape)
#assert camera.shape[0:2] == g_camera.shape[0:2]

def stacks_demo():
    g = gaussian(12, 26)

    ga, l = stacks(derek, g)
    ga_img = np.concatenate(ga, axis=1)
    save(ga_img, "out/gaussian_stack.jpg")
    l_img = np.concatenate(l, axis=1)
    save(l_img, "out/laplacian_stack.jpg")

#stacks_demo()

#save(hybrid(derek, nutmeg, 1500), "out/derek_nutmeg_hybrid.jpg")
#save(blend(apple, orange, mask, 32, 6), "out/oraple.jpg")
def sf_himilaya():
    sf = skio.imread("img/sf.jpg", 0)
    himilaya = skio.imread("img/himilaya.jpg", 0)
    sf_mask = skio.imread("img/sf_mask.jpg", 0)
    save(blend(himilaya, sf, sf_mask, 32, 6), "out/himilaya.jpg")

#sf_himilaya()