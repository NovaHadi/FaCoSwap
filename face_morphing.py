# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 04:50:08 2022

@author: ACER
"""
import streamlit as st
import cv2
#import math
import numpy as np

@st.cache_data
def draw_delaunay(img_ori, _subdiv, delaunay_color ) :
    img = img_ori.copy()
    triangleList = _subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])
    
    for t in triangleList :
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            cv2.line(img, pt1, pt2, delaunay_color, 1)
            cv2.line(img, pt2, pt3, delaunay_color, 1)
            cv2.line(img, pt3, pt1, delaunay_color, 1)
    return img

@st.cache_data
def applyAffineTransform(src, srcTri, dstTri, size) :
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    return dst

@st.cache_data
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

@st.cache_data
def calculateDelaunayTriangles(rect, points, img):
    delaunay_img = img.copy()

    #create subdiv
    subdiv = cv2.Subdiv2D(rect);
    
    # Insert points into subdiv
    for p in points:
        ptup = tuple((p[0].item(), p[1].item()))
        subdiv.insert(ptup) 
    
    delaunay_img = draw_delaunay( delaunay_img, subdiv, (255, 255, 255))
    
    triangleList = subdiv.getTriangleList();
    
    delaunayTri = []
    
    pt = []    
        
    for t in triangleList:        
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        
        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            ind = []
            #Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)    
            # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph 
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        
        pt = []        
    return delaunayTri, delaunay_img

@st.cache_data
def morphing(img1, img2, mask_im, delaunay_img, delaunay_img2, points1, points2, alpha):
    output =  img1.copy()
    #delaunay_img = img1.copy()
    #delaunay_img2 = img2.copy()
    #mask_img = np.zeros(img1.shape, dtype = np.float32)
    mask_img = mask_im.copy()
    size = img1.shape
    rect = (0, 0, size[1], size[0])
    # Compute weighted average point coordinates
    # points--> destination; p1-->source1; p2-->source2;
    # morphing p1+p2-->points
    p1 = [];
    p2 = [];
    points = [];
    for i in range(0, len(points1)):
        #x = ( 1 - alpha ) * points1[i][0] + alpha * points2[i][0]
        #y = ( 1 - alpha ) * points1[i][1] + alpha * points2[i][1]
        x = ( alpha) * points1[i][0] + (1-alpha) * points2[i][0]
        y = ( alpha) * points1[i][1] + (1-alpha) * points2[i][1]
        points.append((x,y))
        
        p1.append((points1[i][0],points1[i][1]))
        p2.append((points2[i][0],points2[i][1]))

    # Calculate delaunay triangulation
    dt, delaunay_img = calculateDelaunayTriangles(rect, points, delaunay_img)
    dt2, delaunay_img2 = calculateDelaunayTriangles(rect, p2, delaunay_img2)
    
    
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
        img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

        size = (r[2], r[3])
        #print(size)
        warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
        warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

        # Alpha blend rectangular patches
        #imgRect = warpImage1 
        imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2
    
        # Copy triangular region of the rectangular patch to the output image
        output[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = output[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask
        mask_img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = mask_img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask
        #output[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = output[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask
    
    # return the aligned face
    return output, mask_img, delaunay_img, delaunay_img2

@st.cache_data
def morphing_original(img1, img2, d_img1, d_img2, mask_im, points1, points2, alpha):
    output =  img1.copy()
    delaunay_img = d_img1
    delaunay_img2 = d_img2
    #delaunay_img = img1.copy()
    #delaunay_img2 = img2.copy()
    #mask_img = np.zeros(img1.shape, dtype = np.float32)
    mask_img = mask_im.copy()
    size = img1.shape
    rect = (0, 0, size[1], size[0])
    # Compute weighted average point coordinates
    # points--> destination; p1-->source1; p2-->source2;
    # morphing p1+p2-->points
    p1 = [];
    p2 = [];
    points = [];
    for i in range(0, len(points1)):
        x = ( 1 - alpha ) * points1[i][0] + alpha * points2[i][0]
        y = ( 1 - alpha ) * points1[i][1] + alpha * points2[i][1]
        #x = ( alpha) * points1[i][0] + (1-alpha) * points2[i][0]
        #y = ( alpha) * points1[i][1] + (1-alpha) * points2[i][1]
        points.append((x,y))
        
        p1.append((points1[i][0],points1[i][1]))
        p2.append((points2[i][0],points2[i][1]))

    # Calculate delaunay triangulation
    dt, delaunay_img = calculateDelaunayTriangles(rect, points, delaunay_img)
    dt2, delaunay_img2 = calculateDelaunayTriangles(rect, p2, delaunay_img2)
    
    
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
        img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

        size = (r[2], r[3])
        #print(size)
        warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
        warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

        # Alpha blend rectangular patches
        #imgRect = warpImage1 
        imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2
    
        # Copy triangular region of the rectangular patch to the output image
        output[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = output[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask
        mask_img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = mask_img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask
        #output[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = output[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask
    
    # return the aligned face
    return output, mask_img, delaunay_img, delaunay_img2
