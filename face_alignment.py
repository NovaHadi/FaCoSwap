# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 01:56:20 2022

@author: N.H.Lestriandoko
"""


#from imutils import face_utils

import numpy as np
#import dlib
#import matplotlib.pyplot as plt
#import imageio
#import os
import cv2
#import math
from face_morphing import calculateDelaunayTriangles


def applyAffineTransform(src, srcTri, dstTri, size) :
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    return dst

def manual_aligning_68_v3(img, shape, mean_points):
    size = img.shape
    rect = (0, 0, size[1], size[0])
    # extract landmark points

    points1 = shape[0:60]
    points2 = mean_points[0:60]
    

    points1 = np.append(points1, [[points1[0][0],2],[points1[16][0],2]], axis=0)

    points2 = np.append(points2, [[points2[0][0],2], [points2[16][0],2]], axis=0)
                        
    points1 = np.append(points1, [[0, 0], [0, size[1]-1], [size[0]-1, 0], [size[1]-1, size[0]-1], 
                                  [size[1]/2-1, size[0]-1], [size[1]-1, size[0]/2-1], [size[1]*0.25-1, 0], 
                                  [size[1]/2-1, 0], [size[1]*0.75-1, 0], [0, size[0]/2-1]], axis=0)
    points2 = np.append(points2, [[0, 0], [0, size[1]-1], [size[0]-1, 0], [size[1]-1, size[0]-1], 
                                  [size[1]/2-1, size[0]-1], [size[1]-1, size[0]/2-1], [size[1]*0.25-1, 0], 
                                  [size[1]/2-1, 0], [size[1]*0.75-1, 0], [0, size[0]/2-1]], axis=0)

    # Allocate space for final output
    output =  np.zeros(img.shape, dtype = img.dtype)
    
    # Compute weighted average point coordinates
    p1 = [];
    p2 = [];
    points = [];
    for i in range(0, len(points1)):
        x = points2[i][0]
        y = points2[i][1]
        points.append((x,y))
        
        p1.append((points1[i][0],points1[i][1]))
        p2.append((points2[i][0],points2[i][1]))

    # Calculate delaunay triangulation
    dt, delaunay_img = calculateDelaunayTriangles(rect, points,img)
    #dt1, delaunay_img1 = calculateDelaunayTriangles(rect, p1,img)
    
    for dtr in dt :
        x = int(dtr[0])
        y = int(dtr[1])
        z = int(dtr[2])
        
        t1 = [p1[x], p1[y], p1[z]]
        t2 = [p2[x], p2[y], p2[z]]
        t = [ points[x], points[y], points[z] ]
    
        # Find bounding rectangle for each triangle
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))
        r = cv2.boundingRect(np.float32([t]))
        
        # Offset points by left top corner of the respective rectangles
        t1Rect = []
        t2Rect = []
        tRect = []
        
        for i in range(0, 3):
            tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
            t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
            t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

        # Get mask by filling triangle
        mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
        cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

        # Apply warpImage to small rectangular patches
        img1Rect = img[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

        size = (r[2], r[3])
        #print(size)
        warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)

        # Alpha blend rectangular patches
        imgRect = warpImage1 
    
        # Copy triangular region of the rectangular patch to the output image
        output[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = output[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask

    
    # return the aligned face
    return output
