from skimage import filters
import torch
import cv2
import numpy as np
import math
import imquality.brisque as brisque
from skimage.metrics import mean_squared_error
from scipy.fftpack import fft2, fftshift


def psnr(target, ref):
    # assume RGB image
    target_data = target.astype(float)
    ref_data = ref.astype(float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)


# define function for mean squared error (MSE)
def mse(target, ref):
    # the MSE between the two images is the sum of the squared difference between the two images
    err = np.sum((target.astype('float') - ref.astype('float')) ** 2)
    err /= float(target.shape[0] * target.shape[1])

    return err


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


# define function that combines all three image quality metrics
def compare_images(target, ref):
    scores = []
    scores.append(psnr(target, ref))
    scores.append(mse(target, ref))
    scores.append(ssim(target, ref))

    return scores


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def ms_ssim(img1, img2, window_sizes=[11, 7, 5, 3], sigmas=[1.5, 1.5 * 2, 1.5 * 2 ** 2, 1.5 * 2 ** 3], L=1.0,
            alpha=0.84):
    """Calculates the MS-SSIM between two images"""
    mssim = torch.tensor(1.0)
    for i, (window_size, sigma) in enumerate(zip(window_sizes, sigmas)):
        ssim_i = ssim(img1, img2)
        mssim = ssim_i ** alpha
    return mssim


def mutual_information(image1, image2):
    """Calculates mutual information between two images."""

    # Calculate joint histogram
    hist_2d, _, _ = np.histogram2d(image1.ravel(), image2.ravel(), bins=256)

    # Convert bins counts to probability values
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # Marginal for x over y
    py = np.sum(pxy, axis=0)  # Marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals

    # Calculate MI
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def calculate_entropy(image):
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 255))

    prob = hist / np.sum(hist)

    entropy = -np.sum(prob * np.log2(prob + 1e-10))  # Add small value to avoid log(0)

    return entropy


def fqi(fused_image, image1, image2):
    """
    Calculates the Feature Quality Index (FQI) for image fusion.

    Args:
        fused_image: The fused image obtained from image fusion techniques.
        image1: The first source image.
        image2: The second source image.

    Returns:
        FQI value.
    """

    # Calculate edge information using Sobel filter
    edge_fused = filters.sobel(fused_image)
    edge_image1 = filters.sobel(image1)
    edge_image2 = filters.sobel(image2)

    # Calculate structural similarity (SSIM)
    ssim_value = ssim(fused_image, image1) + ssim(fused_image, image2)

    # Calculate edge preservation factor
    edge_preservation = np.sum(np.abs(edge_fused - edge_image1)) + np.sum(np.abs(edge_fused - edge_image2))

    # Calculate FQI
    fqi_value = ssim_value * (1 - edge_preservation)

    return fqi_value


def calculate_spatial_frequency(image):
    f = fft2(image)

    fshift = fftshift(f)

    magnitude_spectrum = np.abs(fshift)

    # Calculate average magnitude within a specific frequency range (e.g., high frequencies)

    center_x, center_y = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2

    high_freq_mask = np.zeros_like(magnitude_spectrum)

    high_freq_mask[center_x - 100:center_x + 100, center_y - 100:center_y + 100] = 1

    spatial_freq = np.mean(magnitude_spectrum * high_freq_mask)

    return spatial_freq


def evaluation(mri, ct, fused_img):
    imageWidth = mri.shape[0]
    imageHeight = mri.shape[1]
    psnr = PSNR(mri, ct)
    SSIM = ssim(mri, ct)
    FMI = mutual_information(mri, ct)
    SD = np.std(mri)
    ENt = calculate_entropy(mri)
    Bri = brisque.score(fused_img)
    mse = mean_squared_error(mri, ct)
    rmse = np.sqrt(mse)
    fused_variance = np.var(fused_img)
    FQI = fqi(fused_img, mri, ct)
    sf = calculate_spatial_frequency(fused_img)
    CC = np.corrcoef(mri.ravel(), ct.ravel())[0, 1]
    Eval = [SSIM, psnr, FMI, ENt, Bri, rmse, FQI, sf, CC, fused_variance]
    return Eval
