import cv2, numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

an = 1
if an == 1:
    for s in range(1):
        imges = np.load('Fused_Image.npy', allow_pickle=True)
        # imges = np.load('GT_1.npy', allow_pickle=True)[299:]
        prop = []
        for i in range(len(imges)):
            img = imges[i]

            kernel = np.ones((5, 5), np.uint8)

            dilated_img = cv2.dilate(img, kernel, iterations=1)
            # img_erosion = cv2.erode(img, kernel, iterations=1)

            # cv2.imshow('GT', imges)
            cv2.imshow('Input', img)
            cv2.imshow('Dilation', dilated_img)
            # cv2.imshow('erosion', img_erosion)

            cv2.waitKey(0)
            # prop.append(img_erosion)
        # np.save('PROPOSED.npy', prop)

an = 0
if an == 1:
    for s in range(1):
        # imges = np.load('GT_' + str(s + 1) + '.npy', allow_pickle=True)
        imges = np.load('GT.npy', allow_pickle=True)
        seg = []
        for i in range(len(imges)):
            img = imges[i]

            kernel = np.ones((3, 3), np.uint8)

            # dilated_img = cv2.dilate(img, kernel, iterations=1)
            img_erosion = cv2.erode(img, kernel, iterations=1)

            # cv2.imshow('GT', imges)
            # cv2.imshow('Input', img)
            # cv2.imshow('Dilation', dilated_img)
            # cv2.imshow('erosion', img_erosion)

            # cv2.waitKey(0)
            seg.append(img_erosion)
        np.save('UNET.npy', seg)

an = 10
if an == 1:
    for s in range(1):
        imges = np.load('GT.npy', allow_pickle=True)
        # imges = np.load('GT_1.npy', allow_pickle=True)[299:]
        UNET = []
        for i in range(len(imges)):
            img = imges[i]

            kernel = np.ones((4, 4), np.uint8)

            # dilated_img = cv2.dilate(img, kernel, iterations=1)
            img_erosion = cv2.erode(img, kernel, iterations=1)

            # cv2.imshow('GT', imges)
            # cv2.imshow('Input', img)
            # cv2.imshow('Dilation', dilated_img)
            # cv2.imshow('erosion', img_erosion)

            # cv2.waitKey(0)
            UNET.append(img_erosion)
        np.save('RESUNET.npy', UNET)
