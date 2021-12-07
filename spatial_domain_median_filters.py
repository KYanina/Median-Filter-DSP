import dippykit as dip
import numpy as np
import math
import matplotlib.pyplot as plt

figure_counter = 0


def create_figure(images, titles, rows, cols, add_color_bar=False):
    global figure_counter
    dip.figure(figure_counter)
    figure_counter += 1
    for i in range(len(images)):
        dip.subplot(rows, cols, i + 1)
        dip.imshow(images[i], 'gray')
        dip.title(titles[i])
        if add_color_bar:
            dip.colorbar()


# General function that takes image(single channel with values range [0,1]),
# block function that will be applied to calculate new pixel values
# and the block size(should be of odd size)
def filter_image(image, func, block_size=(3, 3)):
    bh, bw = block_size
    width, height = image.shape
    result = np.copy(image)
    for i in range(bw // 2, width - bw // 2):
        for j in range(bh // 2, height - bh // 2):
            block = image[i - bw // 2:i + 1 + bw // 2, j - bh // 2:j + 1 + bh // 2]
            result[i, j] = func(block)
    return result


# returns the median value in given block x
def median_filter(x):
    return np.median(x)


# for given block x looks at the central pixel.
# If it is minimum of maximum among other values in the block then return median of the block,
# otherwise pixel without any change
def median_sorted_filter(x):
    width, height = x.shape
    length = width * height
    ar = x.flatten()
    pixel = ar[length // 2]
    ar = np.concatenate((ar[:length // 2], ar[1 + length // 2:]))
    ar = np.sort(ar)
    if pixel <= ar[0] or pixel >= ar[-1]:
        return np.median(x)
    else:
        return pixel


# The masked weighted median filter
# THe weights of all pixels are defind using the given mask. THen median is chosen mased on the mask
def weighted_mask_filter(mask):
    def f(x):
        width, height = x.shape
        length = width * height
        ar = x.flatten()
        mask_flat = mask.flatten()
        ar_extended = []
        for i in range(length):
            ar_extended.extend([ar[i]] * mask_flat[i])
        ar_extended.sort()
        return ar_extended[len(ar_extended) // 2]

    return f


# Weighted median filter. Each pixel has the weight according to it's value.
# The range of weights is defined by parameter buckets.
# For example, if buckets is 10, then values from range[0,0.1] will get the weight of 1,
# values from [0.4,0.5] will get the weight of 5, and from [0.9,1] the weight of 10.
# THen the median value is returned
def weighted_median_filter(buckets):
    def f(x):
        ar = x.flatten()
        ar.sort()
        s = 0
        for p in ar:
            s += 1 + math.ceil(p * buckets)
        median = (s // 2) + 1
        accum = 0
        for p in ar:
            accum += 1 + math.ceil(p * buckets)
            if accum >= median:
                return p

    return f


if __name__ == "__main__":
    # read the image
    im = dip.im_read('cameraman.tif')  # Images used for the analysisc01_2.tif c01_1.tif mri.jpg s29_MRI.tif s09_MRI.tif t071_MRIT1.png' t071_MRIT2 US.jpg USkidney.jpg C011.PNG
    # turn to gray scale

    if im.ndim > 2:
        im = np.mean(im, axis=2)

    # introduce noise
    im_noisy = dip.image_noise(im, mode='poisson')
    # im_noisy = im #to run original image without added noise

    med_filtered = filter_image(im_noisy, median_filter, (9, 9))
    med_sor_filtered = filter_image(im_noisy, median_sorted_filter, (9, 9))
    #mask = np.array([[1, 2, 1], [2, 3, 2], [1, 2, 1]]) #3x3
    #mask = np.array([[1, 2, 3, 2, 1], [2, 3, 4, 3, 2], [3, 4, 5, 4, 3], [2, 3, 4, 3, 2], [1, 2, 3, 2, 1]])  # 5x5
    mask = np.array([[1, 2, 3, 4, 5, 4, 3, 2, 1], [2, 3, 4, 5, 6, 5, 4, 3, 2], [3, 4, 5, 6, 7, 6, 5, 4, 3],[4, 5, 6, 7, 8, 7, 6, 5, 4],[5, 6, 7, 8, 9, 8, 7, 6, 5],[4, 5, 6, 7, 8, 7, 6, 5, 4],[3, 4, 5, 6, 7, 6, 5, 4, 3],[2, 3, 4, 5, 6, 5, 4, 3, 2],[1, 2, 3, 4, 5, 4, 3, 2, 1]]) #9x9
    weghted_mask_image = filter_image(im_noisy, weighted_mask_filter(mask))  # regular weighted median
    weighted_median_image = filter_image(im_noisy, weighted_median_filter(10), (9, 9))  # dynamic weighted median

    diff = im_noisy - weighted_median_image

    create_figure([im, med_filtered, med_sor_filtered, weghted_mask_image, weighted_median_image,diff ],# diff,
                 ['Original', 'Median 5x5', 'Adaptive Median 5x5', 'Weighted Median 5x5',
                  'Dynamic Weighted Median 5x5', ], 2, 3) #'original weghted_mask_image diff'
    create_figure([im, im_noisy, weighted_median_image, diff], ['orig', 'noise', 'weighted', 'diff'], 2, 2)
    dip.show()


    [SSIM_noise, a] = dip.SSIM(im_noisy, im)
    PSNR_noise = dip.PSNR(im, im_noisy)
    print("SSIM_noise = {:.2f} ".format(SSIM_noise))
    print("PSNR_noise = {:.2f} dB".format(PSNR_noise))
    [SSIM_median, b] = dip.SSIM(med_filtered, im)
    PSNR_median = dip.PSNR(im, med_filtered)
    print("SSIM_median = {:.2f}".format(SSIM_median))
    print("PSNR_med = {:.2f} dB".format(PSNR_median))
    [SSIM_adaptive, c] = dip.SSIM(med_sor_filtered, im)
    PSNR_adaptive = dip.PSNR(im, med_sor_filtered)
    print("SSIM_adaptive = {:.2f}".format(SSIM_adaptive))
    print("PSNR_adaptive = {:.2f} dB".format(PSNR_adaptive))
    [SSIM_weighted, d] = dip.SSIM(weghted_mask_image, im)
    PSNR_weighted = dip.PSNR(im, weghted_mask_image)
    print("SSIM_weighted = {:.2f}".format(SSIM_weighted))
    print("PSNR_weight = {:.2f} dB".format(PSNR_weighted))
    [SSIM_dyn, g] = dip.SSIM(weighted_median_image, im)
    PSNR_dyn = dip.PSNR(im, weighted_median_image)
    print("SSIM_dyn = {:.2f}".format(SSIM_dyn))
    print("PSNR_dyn = {:.2f} dB".format(PSNR_dyn))

    #Quality assesment part for the project
    #Input image without "ideal reference" is compared to the filter outputs
    from IQA import OMQDI
    BO, BQ1, BQ2 = OMQDI(im_noisy, med_filtered)
    AO, AQ1, AQ2 = OMQDI(im_noisy, med_sor_filtered)
    CO, CQ1, CQ2 = OMQDI(im_noisy, weghted_mask_image)
    DO, DQ1, DQ2 = OMQDI(im_noisy, weighted_median_image)
    EO, EQ1, EQ2 = OMQDI(im_noisy, im)
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(im_noisy)
    ax1.set_title("Original image")
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(med_filtered)
    ax2.set_title("Median 5x5")
    ax2.set_xlabel(f'OMQDI: {round(BO, 3)}, EPF: {round(BQ1, 3)}, NSF: {round(BQ2, 3)}')
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(med_sor_filtered)
    ax3.set_title("Adaptive Median 5x5")
    ax3.set_xlabel(f'OMQDI: {round(AO, 3)}, EPF: {round(AQ1, 3)}, NSF: {round(AQ2, 3)}')
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(weghted_mask_image)
    ax4.set_title("Weighted Median 5x5")
    ax4.set_xlabel(f'OMQDI: {round(CO, 3)}, EPF: {round(CQ1, 3)}, NSF: {round(CQ2, 3)}')
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(weighted_median_image)
    ax5.set_title("Dynamic Weighted Median 5x5")
    ax5.set_xlabel(f'OMQDI: {round(DO, 3)}, EPF: {round(DQ1, 3)}, NSF: {round(DQ2, 3)}')
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(im)
    ax6.set_title("Original Image")
    ax6.set_xlabel(f'OMQDI: {round(EO, 3)}, EPF: {round(EQ1, 3)}, NSF: {round(EQ2, 3)}')
    plt.show()
