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



if __name__ == "__main__":
    print("__main__ Start")
    # image = cv2.imread('02_3grain_base_sliced_x250_delta_gr0_test_60.png')
    # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # apply binary thresholding
    # ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    # # visualize the binary image
    # cv2.imshow('Binary image', thresh)
    # cv2.waitKey(0)
    # cv2.imwrite('image_thres1.jpg', thresh)
    # cv2.destroyAllWindows()
    src = cv2.imread('02_3grain_base_sliced_x250_delta_gr0_test_60.png', 1)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    blur = cv2.blur(gray, (3, 3))  # blur the image
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    # Finding contours for the thresholded image #im2,
    contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # create an empty black image
    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

    # draw contours and hull points
    for i in range(len(contours)):
        color_contours = (0, 255, 0)  # green - color for contours
        color = (255, 0, 0)  # blue - color for convex hull
        # draw ith contour
        cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        # draw ith convex hull object
        # cv2.drawContours(drawing, hull, i, color, 1, 8)
    cv2.imshow('drawing', drawing)
    cv2.waitKey(0)
    cv2.imwrite('image_contours.jpg', th3)
    cv2.destroyAllWindows()

    sys.exit()
