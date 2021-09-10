
from scipy.signal import convolve2d
from scipy.signal import unit_impulse
from scipy.ndimage.interpolation import rotate
from skimage import img_as_uint
from skimage import img_as_float
import skimage.io as io
import skimage.data as data
import skimage.transform as sktr
import numpy as np
import cv2
from matplotlib.pyplot import hist
import matplotlib.pyplot as plt
from align_image_code import align_images
from skimage.color import rgb2gray
import argparse

# Start by saving the cameraman image
#io.imsave("./cameraman.jpg", data.camera())

## 1.1

# Differential matrices
D_x = [[1, -1], [0, 0]]
D_y = [[1, 0], [-1, 0]]

# Helper functions to convolve the derivative on each axis
def convolve_x(src):
    return convolve2d(src, D_x)

def convolve_y(src):
    return convolve2d(src, D_y)

# Create a mask for the derivative images
def mask_image_derivative(dx, dy, threshold):
    mask_x = abs(dx[:,:]) > threshold
    mask_y = abs(dy[:,:]) > threshold
    return mask_x + mask_y

# Saves the borders produces by the derivative map
def borders_no_blur(image, output_name):
    thresh = 20
    dx = convolve_x(image)
    dy = convolve_y(image)
    mask = mask_image_derivative(dx, dy, thresh)
    io.imsave(output_name, img_as_uint(mask))


## 1.2

# Creates gaussian kernel
def gaussian_kernel(size, sigma):
    kernel = cv2.getGaussianKernel(size, sigma)
    return np.outer(kernel, np.transpose(kernel))

# Saves the blurred borders for the derivative map
def borders_blur(image, output_name):
    thresh = 20
    gauss = gaussian_kernel(20, 1.5)
    blurry = convolve2d(image, gauss)
    dx = convolve_x(blurry)
    dy = convolve_y(blurry)
    blurry_mask = mask_image_derivative(dx, dy, thresh)
    io.imsave(output_name, img_as_uint(blurry_mask))

# Saves the blurred borders for the derivative map using a preprocessed filter (DOF)
def borders_blur_dog(image, output_name):
    thresh = 20
    gauss = gaussian_kernel(20, 1.5)
    kernel_dx = convolve_x(gauss)
    kernel_dy = convolve_y(gauss)
    blurry_dx =  convolve2d(image, kernel_dx)
    blurry_dy =  convolve2d(image, kernel_dy)
    dof_blurry_mask = mask_image_derivative(blurry_dx, blurry_dy, thresh)
    io.imsave(output_name, img_as_uint(dof_blurry_mask))


## 1.3

# Crops an image given a split coefficient
def crop(image, divisor):
    x = image.shape[0]
    y = image.shape[1]
    x1 = int(x/divisor)
    x2 = x - int(x/divisor)
    y1 = int(y/divisor)
    y2 = y - int(y/divisor)
    return image[x1:x2, y1:y2]

# Calculates the number of perpendicular gradients
def how_many_perpendicular(image):
    # Counts perpendicular as in a range of 80 - 100 degrees
    return sum(sum(abs(abs(image[:,:]) - 90) < 10))

# Fixes the rotation for a channel
def fix_rotate(image):
    rotations = []
    # From -40 to 40 degrees of rotation skipping by 2 degrees
    for angle in range(-40, 40, 2):
        rotate_image = rotate(image, angle)
        # Crop the image in 5
        rotate_image = crop(rotate_image, 5)

        # Get the derivative maps for each axis and count the perpendicular angles
        dx = convolve_x(rotate_image)
        dy = convolve_y(rotate_image)
        this_num_horiz = how_many_perpendicular(dy)
        this_num_vert = how_many_perpendicular(dx)

        num_edges = this_num_horiz + this_num_vert
        rotations.append((num_edges, angle, (dx, dy)))

    # return the best count for the angles
    return max(rotations, key=lambda x: x[0])

# Fix the rotation for a channel and apply it for the others
def fix_rotate_color(image_name):
    img = io.imread(image_name)
    # Fix the 1st channel
    rotation = fix_rotate(img[:,:,0])
    angle = rotation[1]
    image_name = image_name.replace(".", "").replace("/", "").replace("jpg", "")
    image_histogram_name = "histogram_" + image_name
    dif = np.concatenate([rotation[2][0], rotation[2][1]])
    fig = hist(dif)
    # Save the histogram of angles
    plt.savefig(image_histogram_name)

    # If the image doesn't have more than one channel, return the channel
    if len(img.shape) < 3 or img.shape[2] == 1:
        return rotate(img, angle)

    # Else, apply the rotation to all of the channels
    R = rotate(img[:,:,0], angle)
    G = rotate(img[:,:,1], angle)
    B = rotate(img[:,:,2], angle)
    return np.dstack([R, G, B])

# Saves the fixed image
def fix_image(image_name, output_name):
    io.imsave(output_name, fix_rotate_color(image_name))


## 2.1

# Sharps an image
def sharpen(image, sz, sigma):
    gauss = gaussian_kernel(sz, sigma)
    lapl = np.zeros(gauss.shape)
    lapl[int(sz/2), int(sz/2)] = 1
    lapl = 2*lapl - gauss
    image = img_as_float(image)
    shape = image.shape
    RGB_shape = (shape[0], shape[1], 1)
    blurry_0 = convolve2d(image[:,:,0], lapl, mode="same").reshape(RGB_shape)
    blurry_1 = convolve2d(image[:,:,1], lapl, mode="same").reshape(RGB_shape)
    blurry_2 = convolve2d(image[:,:,2], lapl, mode="same").reshape(RGB_shape)

    final = np.dstack([blurry_0, blurry_1, blurry_2])
    return final

# Saves the sharpen image
def save_sharp(image, sz, sigma, output_image):
    sharp = sharpen(image, sz, sigma)
    io.imsave(output_image, sharp)


## 2.2

# Creates a hybrid image
def hybrid_image(im1, im2, s1, s2, output_image, gray1=False, gray2=False):
    im2, im1 = align_images(im2, im1)
    im1 = im1 / np.max(im1)
    im2 = im2 / np.max(im2)
    kernel_1 = gaussian_kernel(25, s1)
    # Blur every channel
    im1_blurry_0 = convolve2d(im1[:,:,0], kernel_1, mode="same")
    im1_blurry_1 = convolve2d(im1[:,:,1], kernel_1, mode="same")
    im1_blurry_2 = convolve2d(im1[:,:,2], kernel_1, mode="same")
    blurry_im1 = np.dstack([im1_blurry_0, im1_blurry_1, im1_blurry_2])

    kernel_2 = gaussian_kernel(25, s2)
    # Blur every channel
    im2_blurry_0 = convolve2d(im2[:,:,0], kernel_2, mode="same")
    im2_blurry_1 = convolve2d(im2[:,:,1], kernel_2, mode="same")
    im2_blurry_2 = convolve2d(im2[:,:,2], kernel_2, mode="same")
    blurry_im2 = np.dstack([im2_blurry_0, im2_blurry_1, im2_blurry_2])

    # Get high frequencies for image 2
    high_freq = im2 - blurry_im2

    if gray1:
        blurry_im1 = rgb2gray(blurry_im1)
        blurry_im1 = np.dstack([blurry_im1, blurry_im1, blurry_im1])
    if gray2:
        high_freq = rgb2gray(high_freq)
        high_freq = np.dstack([high_freq, high_freq, high_freq])

    final_hybrid = blurry_im1/2 + high_freq/2
    io.imsave(output_image, final_hybrid)


## 2.3

# Generates a gaussian stack
def gaussian_stack(image, size, sigma):
    gauss = gaussian_kernel(size, sigma)
    stack = [image]
    for i in range(0,6):
        blur_0 = convolve2d(stack[i][:, :, 0], gauss, mode="same")
        blur_1 = convolve2d(stack[i][:, :, 1], gauss, mode="same")
        blur_2 = convolve2d(stack[i][:, :, 2], gauss, mode="same")
        final_blur = np.dstack([blur_0, blur_1, blur_2])
        final_blur = final_blur
        stack.append(final_blur)
    return stack[1:]

# Generates a laplacian stack
def laplacian_stack(image, size, sigma):
    stack = [image]
    gauss_stack = gaussian_stack(image, size, sigma)
    buf = image
    for i in range(0,5):
        # Get the correspondant high frequencies for each blurred layer
        sharp = (buf - gauss_stack[i])
        buf = gauss_stack[i]
        stack.append(sharp/255)

    stack.append(gauss_stack[-1]/255)
    return stack[1:]

# Save the stacks for a given picture
def show_stack(image, size, sigma, output_image):
    gauss_stack = gaussian_stack(image, size, sigma)
    gauss_stack = [gauss/255 for gauss in gauss_stack]
    lapl_stack = laplacian_stack(image, size, sigma)
    lapl_stack = [lapl/255 for lapl in lapl_stack]
    fig_gauss = np.concatenate(gauss_stack[:], axis=1)
    io.imsave(output_image + "_gaussianStack.jpg", fig_gauss)
    fig_lapl = np.concatenate(lapl_stack[:], axis=1)
    io.imsave(output_image + "_laplacianStack.jpg", fig_lapl)


## 2.4

# Creates a multiresolution blend for two images and a given mask
def multiresolution_blending(A, B, R, size, sigma, gray):
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

    R = rgb2gray(R)
    # Apply a mask to make sure the image is 1 or 0
    R = R[:,:] > 0.2
    # Easy way to simulate grayscale without changing the other methods
    R = np.dstack([R, R, R])
    LA = laplacian_stack(A, size, sigma)
    LB = laplacian_stack(B, size, sigma)
    GR = gaussian_stack(R, size, sigma)

    LS = []
    for i in range(0, len(LA)):
        LS.append(GR[i]*LA[i] + (1 - GR[i])*LB[i])

    LS = sum(LS)

    return LS


descr_1_1 = "Question 1.1 -> python main.py 1.1 -f [Image] -o [OutputImage]\n"
descr_1_2 = """Question 1.2 -> python main.py 1.2 -f [Image] -o [OutputImage]
               (DOF result will be at 'DoG_'OutputImage)\n"""

descr_1_3 = "Question 1.3 -> python main.py 1.3 -f [Image] -o [OutputImage]\n"
descr_2_1 = """Question 2.1 -> python main.py 2.1 -f [Image]
             -o [OutputImage] -z [KernelSize] -m [KernelSigma]\n"""

descr_2_2 = """Question 2.2 -> python main.py 2.2 -f [LowPassImage]
               -s [HighPassImage] [--gray_one] [--gray_two]
               [-m LowPass sigma] [-g HighPass sigma] -o [OutputImage]\n"""

descr_2_3 = """Question 2.3 -> python main.py 2.3 -f [Image]
             -o [OutputImage] -z [KernelSize] -m [KernelSigma]\n"""

descr_2_4 = """Question 2.4 -> python main.py 2.4 -f [RegionImage]
             -s [NotRegionImage] -t [MaskImage] [-m KernelSigma] -o [OutputImage]
             [--gray-one (Use grayscale)]\n"""

descr_ls = [descr_1_1, descr_1_2, descr_1_3, descr_2_1,
            descr_2_2, descr_2_3, descr_2_4]

intro = "Project 2 for CS 194-26: Fun with filters and frequencies!\n"

parser = argparse.ArgumentParser( intro + "".join(descr_ls))

parser.add_argument('Question',
                    metavar='question',
                    type=str,
                    help="The number of the question to eval (E.g. 1.1).")

parser.add_argument('-f',
                    '--first',
                    dest="first",
                    type=str,
                    help="The name of the first input image.")

parser.add_argument('-s',
                    '--second',
                    dest="second",
                    type=str,
                    help="The name of the second input image.")

parser.add_argument('-t',
                    '--third',
                    dest="third",
                    type=str,
                    help="the name of the third input image.")

parser.add_argument('-o',
                    '--output',
                    dest="output",
                    type=str,
                    help="the name of the output image.")

parser.add_argument('-z',
                    '--size',
                    dest="size",
                    type=int,
                    default=20,
                    help='Sets the size for the gaussian kernel (Q1, Q2.1,  Q2.3-4)')

parser.add_argument('-m',
                    '--sigma_one',
                    dest="sigma1",
                    type=int,
                    default=20,
                    help='Sets the sigma for the gaussian kernel (Q1, Q2.1-4)')

parser.add_argument('-g',
                    '--sigma_two',
                    dest="sigma2",
                    type=int,
                    default=20,
                    help='Sets the sigma for the gaussian kernel of the second image (Q2.2)')

parser.add_argument('--gray_one',
                    dest="gray1",
                    action='store_const',
                    const=True, default=False,
                    help='Toggles greyscale image1(Q2.2)')

parser.add_argument('--gray_two',
                    dest="gray2",
                    action='store_const',
                    const=True, default=False,
                    help='Toggles greyscale image2 (Q2.2)')

args = parser.parse_args()

if args.Question == "1.1":
    img = io.imread(args.first)
    borders_no_blur(img, args.output)
if args.Question == "1.2":
    img = io.imread(args.first)
    borders_blur(img, args.output)
    borders_blur_dog(img, "DoG_"+args.output)
if args.Question == "1.3":
    fix_image(args.first, args.output)
if args.Question == "2.1":
    img = io.imread(args.first)
    save_sharp(img, args.size, args.sigma1, args.output)
if args.Question == "2.2":
    im1 = io.imread(args.first)
    im2 = io.imread(args.second)
    hybrid_image(im1, im2, args.sigma1, args.sigma2, args.output, gray1=args.gray1, gray2=args.gray2)
if args.Question == "2.3":
    im1 = io.imread(args.first)
    show_stack(im1, args.size, args.sigma1, args.output)
if args.Question == "2.4":
    im1 = io.imread(args.first)
    im2 = io.imread(args.second)
    im3 = io.imread(args.third)
    result = multiresolution_blending(im1, im2, im3, args.size, args.sigma1, args.gray1)
    io.imsave(args.output, result)

