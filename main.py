#from proj2.align_image_code import align_images
from matplotlib.pyplot import get_current_fig_manager
import cv2
import scipy as sp
import scipy.signal as sps
import scipy.ndimage.interpolation as spi
import skimage.io as skio
import numpy as np
import matplotlib.pyplot as plt
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

# Part 1.1
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
    """ Binarize two convolved images. """
    mask_x = (abs(x) > threshold_x).astype('float32')
    mask_y = (abs(y) > threshold_y).astype('float32')
    return mask_x, mask_y

def edge_detection(img, threshold_x, threshold_y, debug=False):
    """ Convolve IMG with D_x & D_y, separately binarize the images, and sum the two. """
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
    """ Create a 2D gaussian of width & height ALPHA. """
    g = cv2.getGaussianKernel(alpha, sigma)
    return np.outer(g, g.T)

# PART 1.2
def gaussian_of_image_then_convolution(img, threshold_x, threshold_y, alpha, sigma):
    """ Convolve the image with the Gaussian, then the D_x and D_y filters, binarizing the result. """
    g = gaussian(alpha, sigma)
    convolution_g = sps.convolve2d(img, g).astype('float32')
    return normalize(edge_detection(convolution_g, threshold_x, threshold_y, False))

def convolve_gaussian_then_image(img, threshold_x, threshold_y, alpha, sigma):
    """ 
    Convolve the Gaussian with D_x to produce a G_x,
    convolve the Gaussian with D_y to produce a G_y,
    and then convolve the image with these two filters, binarizing the result.
    """
    g = gaussian(alpha, sigma)
    save(g, "out/gaussian_reg.jpg")
    g_x = convolve_x(g)
    save(g_x, "out/gaussian_x.jpg")
    g_y = convolve_y(g)
    save(g_y, "out/gaussian_y.jpg")

    img_x = sps.convolve2d(img, g_x, mode="same").astype('float32')
    img_x = (abs(img_x) > threshold_x).astype('float32')
    img_y = sps.convolve2d(img, g_y, mode="same").astype('float32')
    img_y = (abs(img_y) > threshold_y).astype('float32')
    
    return ((img_x + img_y)).astype('float32')

def unsharp_mask(img, alpha = 12, sigma = 6):
    g = gaussian(alpha, sigma)
    blur = gaussian_convolution(img, g)
    return np.clip(rgb2gray(img) - blur, 0, 1)

def sharpen(img, alpha = 12, sigma = 3):
    """ Add the unsharp masked version of an image back to its original. """
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]

    # Let's try subtracting a gaussian with the same parameters from the blurred version to get the sharp edges.
    unsharp_r = unsharp_mask(r, alpha, sigma)
    unsharp_g = unsharp_mask(g, alpha, sigma)
    unsharp_b = unsharp_mask(b, alpha, sigma)

    sharpened_version = np.dstack([r, g, b]) + np.dstack([unsharp_r, unsharp_g, unsharp_b])
    return sharpened_version

def resharpen_blurred_image(img, alpha = 13, sigma = 5):

    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    
    # Blur the image.
    blur_r = gaussian_convolution_params(r, alpha, sigma)
    blur_g = gaussian_convolution_params(g, alpha, sigma)
    blur_b = gaussian_convolution_params(b, alpha, sigma)

    # Let's try subtracting a gaussian with the same parameters from the blurred version to get the sharp edges.
    unsharp_r = unsharp_mask(blur_r, alpha, sigma)
    unsharp_g = unsharp_mask(blur_g, alpha, sigma)
    unsharp_b = unsharp_mask(blur_b, alpha, sigma)

    # This is where I saved the intermediate steps of the Taj Mahal.
    #save(normalize(np.dstack([unsharp_r, unsharp_g, unsharp_b])), "out/taj_unsharp.jpg")
    #save(np.dstack([blur_r, blur_g, blur_b]), "out/taj_blurred.jpg")

    resharpened_version = np.dstack([blur_r, blur_g, blur_b]) + np.dstack([unsharp_r, unsharp_g, unsharp_b])
    return resharpened_version

def hybrid(lpi, hpi, sigma):
    """ 
    Align & blend the frequencies left by the Gaussian of size SIGMA in image LPI,
    and add them to the high frequencies produced by the unsharp mask for image HPI.
    Return the blended image.
    """
    lpi, hpi = align_images(lpi, hpi)

    alpha = sigma * 2
    hpi = normalize(unsharp_mask(hpi, alpha, sigma))

    hpi_freqs = np.log(np.abs(np.fft.fftshift(np.fft.fft2(hpi))))
    #plt.imshow(hpi_freqs)
    #plt.show()
    save(hpi_freqs, "out/high_freqs.jpg")


    lpi = gaussian_convolution_params(lpi, alpha, sigma)

    lpi_freqs = np.log(np.abs(np.fft.fftshift(np.fft.fft2(lpi))))
    #plt.imshow(lpi_freqs)
    #plt.show()
    save(lpi_freqs, "out/low_freqs.jpg")

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
    laplacian_stack = [image]
    for i in range(1, levels):
        conv = np.dstack([gaussian_convolution(gaussian_stack[i - 1][:,:,color], conv_matrix) for color in range(3)])
        gaussian_stack.append(conv)
        laplacian_stack.append(gaussian_stack[i - 1][:,:,:] - conv)
    return gaussian_stack, laplacian_stack
        
def blend(A, B, mask, size, sigma, name="oraple"):
    """ Given image A & B, blend between the two along MASK using a gaussian kernel of size SIZE & sigma SIGMA. """
    mask = normalize(mask)

    # Generate Gaussian & Laplacian stacks for both images.
    g = gaussian(size, sigma)
    _, a_lap = stacks(A, g)
    _, b_lap = stacks(B, g)
    mask_gaussian, _ = stacks(mask, g)

    ones = np.ones((mask_gaussian[0].shape[0], mask_gaussian[0].shape[1]))
    ones = np.dstack([ones, ones, ones])

    # Blend levels of the stacks together.
    LS = []
    for i in range(1, len(a_lap)):
        a_part = (mask_gaussian[i] * a_lap[i])
        b_part = ((ones - mask_gaussian[i]) * b_lap[i])
        total = a_part + b_part

        # Save the components
        save(a_part, "out/" + name + "_a" + str(i) + ".jpg")
        save(b_part, "out/" + name + "_b" + str(i) + ".jpg")
        save(total, "out/" + name + "_total" + str(i) + ".jpg")

        LS.append(total)

    LS = sum(LS)

    return LS

# PART 1.1
# Basic X & Y convolutions without any 
#show(convolve_x(cameraman()))
#show(convolve_y(cameraman()))

# PART 1.2
# The image's edges before & after taking the Gaussian.
#save(edge_detection(cameraman(), 48, 40), "out/cameraman_edges_raw.jpg")

# Take the Gaussian of the image BEFORE convolving with DX and DY.
#show(normalize(gaussian_of_image_then_convolution(cameraman(), 40, 25, 4, 1)))

#save(gaussian_of_image_then_convolution(cameraman(), 40, 25, 4, 1), "out/cameraman_edges_gaussian.jpg")

# Convolve the Gaussian with DX and DY, THEN convolve the result with the image.
#show(convolve_gaussian_then_image(cameraman(), 40, 25, 4, 1))
#save(convolve_gaussian_then_image(cameraman(), 40, 25, 4, 1), "out/cameraman_edges_gxgy.jpg")

# PART 2.1
#save(sharpen(taj), "out/taj_sharpened.jpg")
save(resharpen_blurred_image(taj), "out/taj_blurred_sharpened.jpg")

def resharpen_plains():
    plains = skio.imread("img/plains.jpg")
    save(resharpen_blurred_image(plains, alpha=8, sigma=3), "out/plains_resharpened.jpg")

#resharpen_plains()

# PART 2.3

def stacks_demo():
    g = gaussian(12, 26)

    ga, l = stacks(apple, g)
    ga_img = np.concatenate(ga, axis=1)
    save(ga_img, "out/oraple_gaussian_stack.jpg")
    l_img = np.concatenate(l, axis=1)
    save(l_img, "out/oraple_laplacian_stack.jpg")

#stacks_demo()

#hybrid(derek, nutmeg, 13)
#save(hybrid(derek, nutmeg, 13), "out/derek_nutmeg_hybrid.jpg")

def obama_putin():
    obama = skio.imread("img/obama.jpg", 0)
    putin = skio.imread("img/putin.jpg", 0)
    save(hybrid(obama, putin, 3), "out/obama_putin_hybrid4.jpg")

#obama_putin()

def obama_gambino():
    obama = skio.imread("img/obama2.jpg", 0)
    putin = skio.imread("img/bino2.jpg", 0)
    save(hybrid(putin, obama, 2), "out/obama_bino_hybrid2.jpg")

#obama_gambino()

def mona_putin():
    """ Blend the Mona Lisa & Putin without a mask. """
    mona = skio.imread("img/mona_l.jpg", 0)
    putin = skio.imread("img/mona_p.jpg", 0)
    save(hybrid(putin, mona, 4), "out/mona_putin_hybrid.jpg")

#mona_putin()


# PART 2.4

# The size of the kernel here is enormous; try lowering it for demo purposes.
#blend(apple, orange, mask, 90, 34)
#save(blend(apple, orange, mask, 90, 34), "out/oraple.jpg")

def sf_himilaya():
    """ This takes a really long time with the high gaussian kernel size. """
    sf = skio.imread("img/sf.jpg", 0)
    himilaya = skio.imread("img/himilaya.jpg", 0)
    sf_mask = skio.imread("img/sf_mask.jpg", 0)
    save(blend(himilaya, sf, sf_mask, 120, 53, name="sf"), "out/himilaya.jpg")

#sf_himilaya()

def sd_craterlake():
    sd = skio.imread("img/sd_a.jpg", 0)
    crater = skio.imread("img/sd_b.jpg", 0)
    sd_mask = skio.imread("img/sd_mask.jpg", 0)
    save(blend(crater, sd, sd_mask, 30, 12), "out/crater2.jpg")

#sd_craterlake()

def mona_putin_masked():
    sd = skio.imread("img/mona_p.jpg", 0)
    crater = skio.imread("img/mona_l.jpg", 0)
    sd_mask = skio.imread("img/mona_mask.jpg", 0)
    save(blend(crater, sd, sd_mask, 36, 24), "out/mona_putin.jpg")

#mona_putin_masked()