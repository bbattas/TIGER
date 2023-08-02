from MultiExodusReader import MultiExodusReader
import multiprocessing as mp
from CalculationsV2 import CalculationsV2#, parallelPlot
# from CalculationEngine import para_time_build

import json
import argparse
import logging
pt = logging.warning
verb = logging.info
# from logging import warning as pt
# from logging import info as verb
# from VolumeScripts import *

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

from scipy.spatial import ConvexHull, Delaunay
import numpy as np
import time
# from time import timecalc =
import os
import glob
import pandas as pd
import math
import sys
import tracemalloc
import cv2
from scipy.interpolate import splprep, splev


def scaleFactor(bw_img_w_box,xrange,yrange=None):
    # Pixels go 0 -> n for x but y is 0 v n down (0,0 is top left)
    # https://pyimagesearch.com/2021/01/20/opencv-getting-and-setting-pixels/
    src = cv2.imread(bw_img_w_box)#,-1
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    blur = cv2.blur(gray, (3, 3))
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = 0
    if 0 in contours[n]:
        n += 1
    x,y = contours[n].T
    x_pixels = max(x[0]) - min(x[0])
    # Contour Testing Image
    # # src = cv2.imread('02_3grain_base_cv2_gb_x250_test_60.png')
    # cv2.drawContours(src, contours, n, (255,0,0), 1)
    # cv2.imwrite('scaling_domain_contour.jpg', src)
    return xrange/x_pixels, min(x[0]), min(y[0])

def curvature_fromImage(bw_img_w_box,xrange):
    scale, xmin, ymin = scaleFactor(bw_img_w_box,xrange)
    src = cv2.imread(bw_img_w_box)#,-1
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    blur = cv2.medianBlur(gray,45)
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = 0
    print(contours)
    if 0 in contours[n]:
        n += 1
    x,y = contours[n].T
    return

if __name__ == "__main__":
    print("__main__ Start")
    xcoord = [1,2,3,4,5]
    print(xcoord[0])
    print(xcoord[1])
    print(xcoord[1-3])
    scaleFactor('02_3grain_base_cv2_bwbox_gb_x250_test_60.png',300)
    curvature_fromImage('02_3grain_base_cv2_bwbox_gb_x250_test_60.png',300)
    # print(xcoord[4+2])
    # image = cv2.imread('02_3grain_base_sliced_x250_delta_gr0_test_60.png')
    # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # apply binary thresholding
    # ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    # # visualize the binary image
    # cv2.imshow('Binary image', thresh)
    # cv2.waitKey(0)
    # cv2.imwrite('image_thres1.jpg', thresh)
    # cv2.destroyAllWindows()
    # src = cv2.imread('02_3grain_base_sliced_x250_delta_gr0_test_60.png', 1)
    src = cv2.imread('02_3grain_base_cv2_bwbox_gb_x250_test_60.png')#,-1
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    # blur = cv2.blur(gray, (3, 3))  # blur the image
    # r2, t2 = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    # blur = cv2.medianBlur(src,5)
    blur = cv2.medianBlur(gray,45)
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

    # th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    # conBox, h2 = cv2.findContours(t2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(conBox)
    # Finding contours for the thresholded image #im2,
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    # smoothened = []
    # for contour in contours:
    #     x,y = contour.T
    #     # Convert from numpy arrays to normal arrays
    #     x = x.tolist()[0]
    #     y = y.tolist()[0]
    #     # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
    #     tck, u = splprep([x,y], u=None, s=1.0)
    #     # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
    #     u_new = np.linspace(u.min(), u.max(), 50)
    #     # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
    #     x_new, y_new = splev(u_new, tck, der=0)
    #     # Convert it back to numhierah2rchy
    # draw contours and hull points
    # create an empty black image
    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

    for i in range(len(contours)):
        color_contours = (0, 255, 0)  # green - color for contours
        color = (255, 0, 0)  # blue - color for convex hull
        # draw ith contour
        cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        # cv2.drawContours(drawing, conBox, 1, color, 2)
        # # Overlay the smoothed contours on the original image
        # cv2.drawContours(drawing, smoothened, -1, (255,255,255), 2)
        # draw ith convex hull object
        # cv2.drawContours(drawing, hull, i, color, 1, 8)
    # cv2.imshow('drawing', drawing)
    # cv2.waitKey(0)
    cv2.imwrite('image_contours_src.jpg', src)
    cv2.imwrite('image_contours_gray.jpg', gray)
    cv2.imwrite('image_contours_bin.jpg', drawing)
    cv2.imwrite('image_contours_blur.jpg', blur)
    cv2.imwrite('image_contours_binary.jpg', thresh)
    # cv2.imwrite('image_contours_fullThreshold.jpg', t2)
    # cv2.imwrite('image_contours.jpg', th3)
    cv2.destroyAllWindows()

    sys.exit()
