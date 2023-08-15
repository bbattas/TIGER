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
from matplotlib.patches import Polygon, Circle
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


# def find_circle(self,A,B,C):
#     a = A[1]-B[1]
#     b = A[0]-B[0]
#     c = A[1]-C[1]
#     d = A[0]-C[0]
#     e = ((A[1]**2-B[1]**2)-(B[0]**2-A[0]**2))/2.0
#     f = ((A[1]**2-C[1]**2)-(C[0]**2-A[0]**2))/2.0
#     x0 = -(d*e-b*f)/(b*c-a*d)
#     y0 = -(a*f-c*e)/(b*c-a*d)
#     r = math.sqrt((A[1]-x0)**2+(A[0]-y0)**2)
#     return x0,y0,r

# def check_linear(pt0,pt1,pt2):
#     return
# fig, ax = plt.subplots()
#     plt.scatter(*a,c='red')
#     plt.scatter(*b,c='black')
#     plt.scatter(*c,c='green')
#     cir = plt.Circle((x_ctr,y_ctr),r)
#     ax.add_patch(cir)
#     plt.show()

def find_circle(a,b,c,fullcontour,returnXY=False):
    # Using the process from http://paulbourke.net/geometry/circlesphere/
    # slope of ab and bc
    tol = 1e-8
    if b[0] == a[0] and b[0] == c[0]:
        # Vertical line
        print('vertical')
        if returnXY:
            return -1, 0, 0, 0
        else:
            return -1, 0
    # Calculate Slopes between the points
    m1 = (b[1] - a[1])/(b[0] - a[0])
    m2 = (c[1] - b[1])/(c[0] - b[0])
    if m2 == m1:
        # Horizontal line
        if returnXY:
            return -1, 0, 0, 0
        else:
            return -1, 0
    # Correction for when 2 points are stacked vertically to prevent the slope = nan
    if b[0] == a[0]:
        m1 = (b[1] - a[1])/(tol)
    if b[0] == c[0]:
        m2 = (c[1] - b[1])/(tol)
    # Calculate the center points based on where the lines perpendicular cross
    x_ctr = (m1*m2*(a[1] - c[1]) + m2*(a[0] + b[0]) - m1*(b[0] + c[0]))/(2*(m2 - m1))
    # y_ctr = -(1/m1)*(x_ctr - ((a[0] + b[0])/2) ) + ((a[1] + b[1])/2)
    y_ctr = (m1*(a[1] + b[1]) - m2*(b[1] + c[1]) + (a[0] - c[0]))/(2*(m1-m2))
    r = math.sqrt((a[0] - x_ctr)**2 + (a[1] - y_ctr)**2)
    # this returns -1 when the point is outside the contour, 1 inside and 0 on it
    signage = cv2.pointPolygonTest(fullcontour, (x_ctr,y_ctr), False)
    if returnXY:
        return r, signage, x_ctr, y_ctr
    else:
        return r, signage

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

def curvature_fromImage(bw_img_w_box,xrange,nn):
    scale, xmin, ymin = scaleFactor(bw_img_w_box,xrange)
    src = cv2.imread(bw_img_w_box)#,-1
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    blur = cv2.medianBlur(gray,45)
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)#CHAIN_APPROX_SIMPLE)
    n = 0
    # print(contours)
    if 0 in contours[n]:
        n += 1
    x,y = contours[n].T
    if xmin in x[0] or ymin in y[0]:
        print("Plot Boundary in second contour, using third")
        n +=1
        x,y = contours[n].T
    xy = np.asarray([np.asarray([a,b]) for (a,b) in zip(x[0],y[0])])
    rad_loc = []
    for i in range(len(xy)):
        if (i + nn) > (len(xy) - 1):
            rad_loc.append(find_circle(xy[i-nn],xy[i],xy[i+nn-len(xy)],contours[n]))
        else:
            rad_loc.append(find_circle(xy[i-nn],xy[i],xy[i+nn],contours[n]))
    print(rad_loc)
    radii = [row[0] for row in rad_loc]
    sign = [row[1] for row in rad_loc]
    curvature = 1/(np.asarray(radii)*scale)
    curvature = np.where(np.asarray(radii)==-1,0,-1*np.asarray(sign)*curvature)
    print(np.average(curvature))
    # # Testing Plot
    cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter((x-xmin)*scale, (y-ymin)*scale, c=curvature, vmin=min(curvature), vmax=max(curvature), s=35, cmap=cm)
    plt.colorbar(sc)
    plt.show()
    return curvature

if __name__ == "__main__":
    print("__main__ Start")
    xcoord = [1,2,3,4,5]
    print(xcoord[0])
    print(xcoord[1])
    print(xcoord[1-3])
    scaleFactor('02_3grain_base_cv2_bwbox_gb_x250_test_60.png',300)
    read_ti = time.perf_counter()
    # curvature_fromImage('02_3grain_base_cv2_bwbox_gb_x250_test_60.png',300,3)
    curvature_fromImage('02_2grain_full_test_cv2_bwbox_gb_x250_test_60.png',300,3)
    read_tf = time.perf_counter()
    print("  Finished testing:",round(read_tf-read_ti,2),"s")
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
    full_contours, full_hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    x1,y1 = contours[1].T
    xy1 = np.asarray([np.asarray([a,b]) for (a,b) in zip(x1[0],y1[0])])
    xfull,yfull = full_contours[1].T
    xyfull = np.asarray([np.asarray([a,b]) for (a,b) in zip(xfull[0],yfull[0])])
    print(xyfull)
    print(xyfull[2])
    print(xy1)
    res = (xyfull[:, None] == xy1).all(-1).any(-1)
    print(res)
    print(~res)
    print(xyfull[res])
    straight = ~res
    n = len(straight) +1
    doublestraight = np.concatenate((straight, straight), axis=None)
    print(len(straight))
    print(len(doublestraight))
    print(straight[:10])
    print(straight[-5:5])
    print(doublestraight[n-5:n])
    mins = []
    maxes = []
    ls = len(straight) + 1
    for i,val in enumerate(straight):
        if val and straight[i-1]==False and all(doublestraight[i:i+20]):
            mins.append(i)
        if val and doublestraight[i+1]==False and all(doublestraight[ls+i-20:ls+i]):
            maxes.append(i)
    print(mins)
    print(maxes)
    print(min(maxes, key=lambda x: x-mins[1]))
    # pairs = [[lf,maxes[maxes > lf].min() ] for lf in mins]
    linpars = []
    if mins[0] > maxes[0]:
        for i in range(len(mins)):
            if i == (len(mins)-1):
                print(i,'last')
                linpars.append([mins[i],maxes[0]])
            else:
                linpars.append([mins[i],maxes[i+1]])
    else:
        for i in range(len(mins)):
            linpars.append([mins[i],maxes[i]])
    print(linpars)
    straightlen = 0
    for n in linpars:
        straightlen += math.sqrt((xyfull[n[0]][0] - xyfull[n[1]][0])**2 +
                                 (xyfull[n[0]][1] - xyfull[n[1]][1])**2)
    print(straightlen)
    print(cv2.arcLength(full_contours[1],True))
    # print(pairs)
    plt.scatter(xyfull[:,0],xyfull[:,1],s=10)
    plt.scatter(xyfull[mins][:,0],xyfull[mins][:,1],s=15)
    plt.scatter(xyfull[maxes][:,0],xyfull[maxes][:,1],s=15)
    plt.show()
    # out = np.where(res,np.zeros_like(xyfull),xyfull)
    # print(out)
    print(len(xy1))
    print(len(xyfull))
    # print(len(out))
    # out = []
    # for i,row in enumerate(xyfull):
    #     if row in xy1:
    #         out.append('True')
    #     else:
    #         out.append('False')
    # print(out)
    # print(np.where(xyfull == xy1,axis=1))
    # idx, = np.where((xyfull == xy1[:,None]).all(axis=-1).any(0))
    print((xyfull[:] == xy1[:]))
    tempres = ((xyfull == xy1[:,None]).all(-1).any(0))
    if (tempres == res).all():
        print("same")
    # print(list(check))
    # print(idx)
    # print(full_contours)
    # print(len(contours[1]))
    # print(np.unique(full_contours[1],axis=0))
    # print(len(np.unique(full_contours[1],axis=0)))
    # print(~np.isin(full_contours[1],contours[1]))
    # print(np.where(~np.isin(full_contours[1],contours[1]),full_contours,[-1,-1]))
    # print(len(full_contours[1]))
    # print(full_contours[1][~np.isin(full_contours[1],contours[1])])
    # print(full_contours[1] == contours[1][:, None])
    # check = (full_contours[1][:, None] == contours[1]).all(-1).any(-1)
    # print(check)
    # create hull array for convex hull points
    hull = []

    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))
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
        cv2.drawContours(drawing, hull, i, color, 1, 8)

    cv2.imwrite('image_contours_bin.jpg', drawing)

    # # Apply HoughLinesP method to
    # # to directly obtain line end points
    # lines_list =[]
    # lines = cv2.HoughLinesP(
    #             thresh, # Input edge image
    #             1, # Distance resolution in pixels
    #             np.pi/180, # Angle resolution in radians
    #             threshold=100, # Min number of votes for valid line
    #             minLineLength=5, # Min allowed length of line
    #             maxLineGap=10 # Max allowed gap between line for joining them
    #             )

    # # Iterate over points
    # for points in lines:
    #     # Extracted points nested in the list
    #     x1,y1,x2,y2=points[0]
    #     # Draw the lines joing the points
    #     # On the original image
    #     cv2.line(drawing,(x1,y1),(x2,y2),(0,255,0),2)
    #     # Maintain a simples lookup list for points
    #     lines_list.append([(x1,y1),(x2,y2)])

    # # Save the result image
    # cv2.imwrite('detectedLines.png',drawing)
    # cv2.imshow('drawing', drawing)
    # cv2.waitKey(0)
    cv2.imwrite('image_contours_src.jpg', src)
    cv2.imwrite('image_contours_gray.jpg', gray)
    # cv2.imwrite('image_contours_bin.jpg', drawing)
    cv2.imwrite('image_contours_blur.jpg', blur)
    cv2.imwrite('image_contours_binary.jpg', thresh)
    # cv2.imwrite('image_contours_fullThreshold.jpg', t2)
    # cv2.imwrite('image_contours.jpg', th3)
    cv2.destroyAllWindows()

    sys.exit()
