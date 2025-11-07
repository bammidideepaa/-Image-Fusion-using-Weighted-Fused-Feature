import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def Image_Results():
    I = [[15, 20, 25, 28, 29]]
    Images = np.load('CT_Image.npy', allow_pickle=True)
    GT = np.load('PET_Image.npy', allow_pickle=True)
    unet = np.load('Fused_Image_ResNet.npy', allow_pickle=True)
    Resnet = np.load('Fused_Image_Unet.npy', allow_pickle=True)
    PROPOSED = np.load('Fused_Image.npy', allow_pickle=True)
    for i in range(len(I[0])):
        plt.subplot(2, 3, 1)
        plt.title('MRI')
        plt.imshow(Images[I[0][i]])
        plt.subplot(2, 3, 2)
        plt.title('PET')
        plt.imshow(GT[I[0][i]])

        plt.subplot(2, 3, 3)
        plt.title('Ref 1')
        plt.imshow(unet[I[0][i]])

        plt.subplot(2, 3, 4)
        plt.title('Ref 3')
        plt.imshow(Resnet[I[0][i]])

        plt.subplot(2, 3, 5)
        plt.title('Fused Image')
        plt.imshow(PROPOSED[I[0][i]])
        plt.tight_layout()
        plt.show()
        cv.imwrite('./Results/Image_Results/MRI-' + str(i + 1) + '.png', Images[I[0][i]])
        cv.imwrite('./Results/Image_Results/PET-' + str(i + 1) + '.png', GT[I[0][i]])
        cv.imwrite('./Results/Image_Results/Unet-' + str(i + 1) + '.png', unet[I[0][i]])
        cv.imwrite('./Results/Image_Results/resunet-' + str(i + 1) + '.png',
                   Resnet[I[0][i]])
        cv.imwrite('./Results/Image_Results/fused-' + str(i + 1) + '.png',
                   PROPOSED[I[0][i]])


def Sample_Images():
    Orig = np.load('PET_Image.npy', allow_pickle=True)
    ind = [1, 2, 3, 4, 5, 6]
    fig, ax = plt.subplots(2, 3)
    fig.canvas.manager.set_window_title('Sample Images')
    plt.suptitle("Sample Images")
    plt.subplot(2, 3, 1)
    plt.title('Image-1')
    plt.imshow(Orig[ind[0]])
    plt.subplot(2, 3, 2)
    plt.title('Image-2')
    plt.imshow(Orig[ind[1]])
    plt.subplot(2, 3, 3)
    plt.title('Image-3')
    plt.imshow(Orig[ind[2]])
    plt.subplot(2, 3, 4)
    plt.title('Image-4')
    plt.imshow(Orig[ind[3]])
    plt.subplot(2, 3, 5)
    plt.title('Image-5')
    plt.imshow(Orig[ind[4]])
    plt.subplot(2, 3, 6)
    plt.title('Image-6')
    plt.imshow(Orig[ind[5]])
    plt.show()


if __name__ == '__main__':
    Image_Results()
    # Sample_Images()
