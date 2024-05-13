# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 10:41:39 2022

@author: N.H.Lestriandoko

Non Discriminative Texture (NDT and iNDT)

"""

from imutils import face_utils

import streamlit as st
import numpy as np
#import dlib
#import matplotlib.pyplot as plt
#import imageio
#import os
import cv2
#import math
import face_morphing as mp

@st.cache_data
def getEyebrowsMorphing(shape, shape2, img1, img2, mask_img, delaunay_img, delaunay_img2, alpha):
    w = 5
    w2 = 4
    points1 = shape[17:22] 
    points1 = np.append(points1,[[points1[0][0],points1[0][1]-w],
                                 [points1[1][0],points1[1][1]-w],
                                 [points1[2][0],points1[2][1]-w],
                                 [points1[3][0],points1[3][1]-w],
                                 [points1[4][0]+w,points1[4][1]-w]], axis=0)
    points1 = np.append(points1,[[points1[0][0],points1[0][1]+w2],
                                 [points1[4][0]+w,points1[4][1]+w2]], axis=0)

    points2 = shape2[17:22]
    points2 = np.append(points2,[[points2[0][0],points2[0][1]-w],
                                 [points2[1][0],points2[1][1]-w],
                                 [points2[2][0],points2[2][1]-w],
                                 [points2[3][0],points2[3][1]-w],
                                 [points2[4][0]+w,points2[4][1]-w]], axis=0)
    points2 = np.append(points2,[[points2[0][0],points2[0][1]+w2],
                                 [points2[4][0]+w,points2[4][1]+w2]], axis=0)
    output, mask, delaunay_img, delaunay_img2 =mp.morphing(img1, img2, mask_img, 
                                                         delaunay_img, delaunay_img2, points1, points2, alpha)    
    
    points1a = shape[22:27]
    points1a = np.append(points1a,[[points1a[0][0]-w,points1a[0][1]-w],
                                 [points1a[1][0],points1a[1][1]-w],
                                 [points1a[2][0],points1a[2][1]-w],
                                 [points1a[3][0],points1a[3][1]-w],
                                 [points1a[4][0],points1a[4][1]-w]], axis=0)
    points1a = np.append(points1a,[[points1a[0][0]-w,points1a[0][1]+w2],
                                 [points1a[4][0],points1a[4][1]+w2]], axis=0)
    points2a = shape2[22:27]
    points2a = np.append(points2a,[[points2a[0][0]-w,points2a[0][1]-w],
                                 [points2a[1][0],points2a[1][1]-w],
                                 [points2a[2][0],points2a[2][1]-w],
                                 [points2a[3][0],points2a[3][1]-w],
                                 [points2a[4][0],points2a[4][1]-w]], axis=0)
    points2a = np.append(points2a,[[points2a[0][0]-w,points2a[0][1]+w2],
                                 [points2a[4][0],points2a[4][1]+w2]], axis=0)
    output, mask, delaunay_img, delaunay_img2 =mp.morphing(output, img2, mask, 
                                                         delaunay_img, delaunay_img2, points1a, points2a, alpha) 
    all_points = np.concatenate((points1, points1a), axis=0)
    ori_points = shape[17:27]
    
    return output, mask, delaunay_img, delaunay_img2, all_points, ori_points

@st.cache_data
def getEyebrowsMorphing_v2(shape, shape2, img1, img2, mask_img, delaunay_img, delaunay_img2, alpha):
    w = 6
    w3 = 6
    s = 2
    height, width, channel = img1.shape
    size = width
    if (size>=300) and (size<450):
        w = w * 2
        w3 = w3 * 2
    elif (size>=450) :
        w = w * 3
        w3 = w3 * 3
    else:
        w = 6
        w3 = 6

    # divide an eyebrow into three morphing areas
    # =============== RIGHT EYEBROW =================    
    points1 = shape[17:22] 
    points2 = shape2[17:22]
    r_eye_pts = shape[36:42]       
    r_eye_pts2 = shape2[36:42]       

    # right eyebrow - area 1 :
    r_pts_1 = shape[17:20]
    r_pts_1 = np.append(r_pts_1,[[points1[0][0],points1[0][1]-w],
                                 [points1[1][0],points1[1][1]-w],
                                 [points1[2][0],points1[2][1]-w],
                                 [r_eye_pts[0][0]-s*w,r_eye_pts[0][1]],
                                 [r_eye_pts[0][0],r_eye_pts[0][1]-w3],
                                 [r_eye_pts[1][0],r_eye_pts[1][1]-w3]], axis=0)

    avg_r_pts_1 = shape2[17:20]
    avg_r_pts_1 = np.append(avg_r_pts_1,[[points2[0][0],points2[0][1]-w],
                                 [points2[1][0],points2[1][1]-w],
                                 [points2[2][0],points2[2][1]-w],
                                 [r_eye_pts2[0][0]-s*w,r_eye_pts2[0][1]],
                                 [r_eye_pts2[0][0],r_eye_pts2[0][1]-w3],
                                 [r_eye_pts2[1][0],r_eye_pts2[1][1]-w3]], axis=0)
    output, mask, delaunay_img, delaunay_img2 =mp.morphing(img1, img2, mask_img,  
                                                         delaunay_img, delaunay_img2, r_pts_1, avg_r_pts_1, alpha)    

    # right eyebrow - area 2 :
    r_pts_2 = shape[19:21]
    r_pts_2 = np.append(r_pts_2,[[points1[2][0],points1[2][1]-w],
                                 [points1[3][0],points1[3][1]-w],
                                 [r_eye_pts[1][0],r_eye_pts[1][1]-w3],
                                 [r_eye_pts[2][0],r_eye_pts[2][1]-w3]], axis=0)

    avg_r_pts_2 = shape2[19:21]
    avg_r_pts_2 = np.append(avg_r_pts_2,[[points2[2][0],points2[2][1]-w],
                                 [points2[3][0],points2[3][1]-w],
                                 [r_eye_pts2[1][0],r_eye_pts2[1][1]-w3],
                                 [r_eye_pts2[2][0],r_eye_pts2[2][1]-w3]], axis=0)
    output, mask, delaunay_img, delaunay_img2 =mp.morphing(output, img2, mask, 
                                                         delaunay_img, delaunay_img2, r_pts_2, avg_r_pts_2, alpha)    
    
    # right eyebrow - area 3 :
    r_pts_3 = shape[20:22]
    """
    r_pts_3 = np.append(r_pts_3,[[points1[3][0]-int(w/2),points1[3][1]-w],
                                 [points1[4][0]+w,points1[4][1]-w],
                                 [r_eye_pts[2][0],r_eye_pts[2][1]-w3],
                                 [r_eye_pts[3][0],r_eye_pts[3][1]-w3],
                                 [r_eye_pts[3][0]+s*w,r_eye_pts[3][1]]], axis=0)
                                 

    avg_r_pts_3 = shape2[20:22]
    avg_r_pts_3 = np.append(avg_r_pts_3,[[points2[3][0]-int(w/2),points2[3][1]-w],
                                 [points2[4][0]+w,points2[4][1]-w],
                                 [r_eye_pts2[2][0],r_eye_pts2[2][1]-w3],
                                 [r_eye_pts2[3][0],r_eye_pts2[3][1]-w3],
                                 [r_eye_pts2[3][0]+s*w,r_eye_pts2[3][1]]], axis=0)
    """
    r_pts_3 = np.append(r_pts_3,[[points1[3][0],points1[3][1]-w],
                                 [points1[4][0]+w,points1[4][1]-w],
                                 [r_eye_pts[2][0],r_eye_pts[2][1]-w3],
                                 [r_eye_pts[3][0],r_eye_pts[3][1]-w3],
                                 [r_eye_pts[3][0]+s*w,r_eye_pts[3][1]]], axis=0)
                                 

    avg_r_pts_3 = shape2[20:22]
    avg_r_pts_3 = np.append(avg_r_pts_3,[[points2[3][0],points2[3][1]-w],
                                 [points2[4][0]+w,points2[4][1]-w],
                                 [r_eye_pts2[2][0],r_eye_pts2[2][1]-w3],
                                 [r_eye_pts2[3][0],r_eye_pts2[3][1]-w3],
                                 [r_eye_pts2[3][0]+s*w,r_eye_pts2[3][1]]], axis=0)
    output, mask, delaunay_img, delaunay_img2 =mp.morphing(output, img2, mask,
                                                         delaunay_img, delaunay_img2, r_pts_3, avg_r_pts_3, alpha)    

    # =============== LEFT EYEBROW =================    
    points1a = shape[22:27] 
    points2a = shape2[22:27]
    l_eye_pts = shape[42:48]       
    l_eye_pts2 = shape2[42:48]       

    # left eyebrow - area 1 :
    l_pts_1 = shape[22:24]
    l_pts_1 = np.append(l_pts_1,[[points1a[0][0]-w,points1a[0][1]-w],
                                 [points1a[1][0],points1a[1][1]-w],
                                 [l_eye_pts[0][0]-s*w,l_eye_pts[0][1]],
                                 [l_eye_pts[0][0],l_eye_pts[0][1]-w3],
                                 [l_eye_pts[1][0],l_eye_pts[1][1]-w3]], axis=0)
        
    avg_l_pts_1 = shape2[22:24]
    avg_l_pts_1 = np.append(avg_l_pts_1,[[points2a[0][0]-w,points2a[0][1]-w],
                                 [points2a[1][0],points2a[1][1]-w],
                                 [l_eye_pts2[0][0]-s*w,l_eye_pts2[0][1]],
                                 [l_eye_pts2[0][0],l_eye_pts2[0][1]-w3],
                                 [l_eye_pts2[1][0],l_eye_pts2[1][1]-w3]], axis=0)
    output, mask, delaunay_img, delaunay_img2 =mp.morphing(output, img2, mask, 
                                                         delaunay_img, delaunay_img2, l_pts_1, avg_l_pts_1, alpha)    

    # left eyebrow - area 2 :
    l_pts_2 = shape[23:25]
    l_pts_2 = np.append(l_pts_2,[[points1a[1][0],points1a[1][1]-w],
                                   [points1a[2][0],points1a[2][1]-w],
                                   [l_eye_pts[1][0],l_eye_pts[1][1]-w3],
                                   [l_eye_pts[2][0],l_eye_pts[2][1]-w3]], axis=0)

    avg_l_pts_2 = shape2[23:25]
    avg_l_pts_2 = np.append(avg_l_pts_2,[[points2a[1][0],points2a[1][1]-w],
                                   [points2a[2][0],points2a[2][1]-w],
                                   [l_eye_pts2[1][0],l_eye_pts2[1][1]-w3],
                                   [l_eye_pts2[2][0],l_eye_pts2[2][1]-w3]], axis=0)
    output, mask, delaunay_img, delaunay_img2 =mp.morphing(output, img2, mask, 
                                                         delaunay_img, delaunay_img2, l_pts_2, avg_l_pts_2, alpha)    
    
    # left eyebrow - area 3 :
    l_pts_3 = shape[24:27]
    l_pts_3 = np.append(l_pts_3,[[points1a[2][0],points1a[2][1]-w],
                                   [points1a[3][0],points1a[3][1]-w],
                                   [points1a[4][0],points1a[4][1]-w],
                                   [l_eye_pts[2][0],l_eye_pts[2][1]-w3],
                                   [l_eye_pts[3][0],l_eye_pts[3][1]-w3],
                                   [l_eye_pts[3][0]+s*w,l_eye_pts[3][1]]], axis=0)

    avg_l_pts_3 = shape2[24:27]
    avg_l_pts_3 = np.append(avg_l_pts_3,[[points2a[2][0],points2a[2][1]-w],
                                   [points2a[3][0],points2a[3][1]-w],
                                   [points2a[4][0],points2a[4][1]-w],
                                   [l_eye_pts2[2][0],l_eye_pts2[2][1]-w3],
                                   [l_eye_pts2[3][0],l_eye_pts2[3][1]-w3],
                                   [l_eye_pts2[3][0]+s*w,l_eye_pts2[3][1]]], axis=0)
    output, mask, delaunay_img, delaunay_img2 =mp.morphing(output, img2, mask, 
                                                         delaunay_img, delaunay_img2, l_pts_3, avg_l_pts_3, alpha)    


    all_points = np.concatenate((r_pts_1,r_pts_2,r_pts_3,l_pts_1,l_pts_2,l_pts_3), axis=0)
    #all_points = np.append((all_points, [l_pts_3,r_pts_1,r_pts_2,r_pts_3]), axis=0)
    ori_points = shape[17:27]
    
    return output, mask, delaunay_img, delaunay_img2, all_points, ori_points

@st.cache_data
def getEyesMorphing(shape, shape2, img1, img2, mask_img, delaunay_img, delaunay_img2, alpha):
    w = 6
    s = 2
    height, width, channel = img1.shape
    size = width
    if (size>=300) and (size<450):
        w = w * 2
    elif (size>=450) :
        w = w * 3
    else:
        w = 6

    points1 = shape[36:42]
    points1 = np.append(points1,[[points1[0][0]-s*w,points1[0][1]],
                                 [points1[1][0],points1[1][1]-w],
                                 [points1[2][0],points1[2][1]-w],
                                 [points1[3][0]+s*w,points1[3][1]],
                                 [points1[4][0],points1[4][1]+w],
                                 [points1[5][0],points1[5][1]+w]], axis=0)
    points2 = shape2[36:42]
    points2 = np.append(points2,[[points2[0][0]-s*w,points2[0][1]],
                                 [points2[1][0],points2[1][1]-w],
                                 [points2[2][0],points2[2][1]-w],
                                 [points2[3][0]+s*w,points2[3][1]],
                                 [points2[4][0],points2[4][1]+w],
                                 [points2[5][0],points2[5][1]+w]], axis=0)
    output, mask, delaunay_img, delaunay_img2 =mp.morphing(img1, img2, mask_img, 
                                                         delaunay_img, delaunay_img2, points1, points2, alpha)    
    points1a = shape[42:48]
    points1a = np.append(points1a,[[points1a[0][0]-s*w,points1a[0][1]],
                                 [points1a[1][0],points1a[1][1]-w],
                                 [points1a[2][0],points1a[2][1]-w],
                                 [points1a[3][0]+s*w,points1a[3][1]],
                                 [points1a[4][0],points1a[4][1]+w],
                                 [points1a[5][0],points1a[5][1]+w]], axis=0)
    points2a = shape2[42:48]
    points2a = np.append(points2a,[[points2a[0][0]-s*w,points2a[0][1]],
                                 [points2a[1][0],points2a[1][1]-w],
                                 [points2a[2][0],points2a[2][1]-w],
                                 [points2a[3][0]+s*w,points2a[3][1]],
                                 [points2a[4][0],points2a[4][1]+w],
                                 [points2a[5][0],points2a[5][1]+w]], axis=0)
    output, mask, delaunay_img, delaunay_img2 =mp.morphing(output, img2, mask, 
                                                         delaunay_img, delaunay_img2, points1a, points2a, alpha)    
    all_points = np.concatenate((points1, points1a), axis=0)
    ori_points = shape[36:48]

    return output, mask, delaunay_img, delaunay_img2, all_points, ori_points

@st.cache_data
def getNoseMorphing(shape, shape2, img1, img2, mask_img, delaunay_img, delaunay_img2, alpha):
    w = 8
    #points1 = shape[[27,31,32,33,34,35]]
    height, width, channel = img1.shape
    size = width
    if (size>=300) and (size<450):
        w = w * 2
    elif (size>=450) :
        w = w * 3
    else:
        w = 8

    points1 = shape[27:36]
    points1 = np.append(points1,[[points1[4][0]-w,points1[4][1]],
                                 [points1[6][0],points1[6][1]+w/2],
                                 [points1[8][0]+w,points1[8][1]]], axis=0)

    #points2 = shape2[[27,31,32,33,34,35]]
    points2 = shape2[27:36]
    points2 = np.append(points2,[[points2[4][0]-w,points2[4][1]],
                                 [points2[6][0],points2[6][1]+w/2],
                                 [points2[8][0]+w,points2[8][1]]], axis=0)
    output, mask, delaunay_img, delaunay_img2 =mp.morphing(img1, img2, mask_img, 
                                                         delaunay_img, delaunay_img2, points1, points2, alpha)    
    all_points = points1.astype(int)
    #ori_points = shape[[27,31,32,33,34,35]]
    ori_points = shape[27:36]

    return output, mask, delaunay_img, delaunay_img2, all_points, ori_points

@st.cache_data
def getMouthMorphing(shape, shape2, img1, img2, mask_img, delaunay_img, delaunay_img2, alpha):
    w = 8
    height, width, channel = img1.shape
    size = width
    if (size>=300) and (size<450):
        w = w * 2
    elif (size>=450) :
        w = w * 3
    else:
        w = 8

    points1 = shape[48:60]
    points1 = np.append(points1,[[points1[0][0]-w,points1[0][1]], 
                                 [points1[1][0],points1[1][1]-w],
                                 [points1[2][0],points1[2][1]-w],
                                 [points1[3][0],points1[3][1]-w],
                                 [points1[4][0],points1[4][1]-w],
                                 [points1[5][0],points1[5][1]-w],
                                 [points1[6][0]+w,points1[6][1]], 
                                 [points1[7][0],points1[7][1]+w],
                                 [points1[8][0],points1[8][1]+w],
                                 [points1[9][0],points1[9][1]+w],
                                 [points1[10][0],points1[10][1]+w],
                                 [points1[11][0],points1[11][1]+w]], axis=0)
    points2 = shape2[48:60]
    points2 = np.append(points2,[[points2[0][0]-w,points2[0][1]], 
                                 [points2[1][0],points2[1][1]-w],
                                 [points2[2][0],points2[2][1]-w],
                                 [points2[3][0],points2[3][1]-w],
                                 [points2[4][0],points2[4][1]-w],
                                 [points2[5][0],points2[5][1]-w],
                                 [points2[6][0]+w,points2[6][1]], 
                                 [points2[7][0],points2[7][1]+w],
                                 [points2[8][0],points2[8][1]+w],
                                 [points2[9][0],points2[9][1]+w],
                                 [points2[10][0],points2[10][1]+w],
                                 [points2[11][0],points2[11][1]+w]], axis=0)
    output, mask, delaunay_img, delaunay_img2 =mp.morphing(img1, img2, mask_img, 
                                                         delaunay_img, delaunay_img2, points1, points2, alpha)    
    all_points = points1
    ori_points = shape[48:60]

    return output, mask, delaunay_img, delaunay_img2, all_points, ori_points

@st.cache_data
def getMaskCenter(img1, mask):
    src_mask = np.zeros(img1.shape, img1.dtype)
    src_mask[mask>0] = 255
    poly = np.argwhere(mask[:,:,2]>0)
    r = cv2.boundingRect(np.float32([poly]))    
    center = (r[1]+int(r[3]/2)), (r[0]+int(r[2]/2))
    return center, src_mask

@st.cache_data
def check_multiple_faces(_face_rects):
    nFaces = len(_face_rects)            
    if nFaces > 1:
        x1, y1, w1, h1 = rect_to_bb(_face_rects[0])
        x2, y2, w2, h2 = rect_to_bb(_face_rects[1])
        if w1>w2:
            face_rect = _face_rects[0]
        else:
            face_rect = _face_rects[1]
    else:
        face_rect = _face_rects[0]    
    return face_rect

@st.cache_data
def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

@st.cache_data
def face_part_replacement(img1, _detector, _predictor, mean_face, alpha, facepart_type):
    img1 = np.uint8(img1)
    img2 = np.uint8(mean_face)
    face_rects1 = _detector(img1, 2)
    if len(face_rects1)>0:
        face_rect1 = check_multiple_faces(face_rects1)                    
    
    face_rects2 = _detector(img2, 2)
    if len(face_rects2)>0:
        face_rect2 = check_multiple_faces(face_rects2)                    
    # extract the landmarks
    landmarks = _predictor(img1, face_rect1)
    landmarks2 = _predictor(img2, face_rect2)
    # to np.array
    shape = face_utils.shape_to_np(landmarks)
    shape2 = face_utils.shape_to_np(landmarks2) 
    
    mask_img = np.zeros(img1.shape, dtype = np.float32)
    delaunay_img = img1.copy()
    delaunay_img2 = img2.copy()
    output_0 = img1.copy()
    if facepart_type=='eyebrows':
        output_0, mask, delaunay_img, delaunay_img2, all_points, ori_points = getEyebrowsMorphing_v2(shape, shape2, img1, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask)
        output = cv2.seamlessClone(output_0, img1, src_mask, center, cv2.NORMAL_CLONE)
        
    if facepart_type=='eyes':
        output_0, mask, delaunay_img, delaunay_img2, all_points, ori_points = getEyesMorphing(shape, shape2, img1, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask)
        output = cv2.seamlessClone(output_0, img1, src_mask, center, cv2.NORMAL_CLONE)
        
    if facepart_type=='nose':
        output_0, mask, delaunay_img, delaunay_img2, all_points, ori_points = getNoseMorphing(shape, shape2, img1, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask)
        output = cv2.seamlessClone(output_0, img1, src_mask, center, cv2.NORMAL_CLONE)

    if facepart_type=='mouth' :
        output_0, mask, delaunay_img, delaunay_img2, all_points, ori_points = getMouthMorphing(shape, shape2, img1, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask)
        output = cv2.seamlessClone(output_0, img1, src_mask, center, cv2.NORMAL_CLONE)

    if facepart_type=='eyebrows-eyes' :
        output, mask1, delaunay_img, delaunay_img2, all_points, ori_points = getEyebrowsMorphing_v2(shape, shape2, img1, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask1)
        seamless_img = cv2.seamlessClone(output, img1, src_mask, center, cv2.NORMAL_CLONE)
        
        output, mask2, delaunay_img, delaunay_img2, all_points, ori_points = getEyesMorphing(shape, shape2, seamless_img, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask2)
        output = cv2.seamlessClone(output, seamless_img, src_mask, center, cv2.NORMAL_CLONE)
        mask = mask1 + mask2
        
    if facepart_type=='eyebrows-nose' :
        output, mask1, delaunay_img, delaunay_img2, all_points, ori_points = getEyebrowsMorphing_v2(shape, shape2, img1, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask1)
        seamless_img = cv2.seamlessClone(output, img1, src_mask, center, cv2.NORMAL_CLONE)
        
        output, mask2, delaunay_img, delaunay_img2, all_points, ori_points = getNoseMorphing(shape, shape2, seamless_img, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask2)
        output = cv2.seamlessClone(output, seamless_img, src_mask, center, cv2.NORMAL_CLONE)
        mask = mask1 + mask2

    if facepart_type=='eyebrows-mouth' :
        output, mask1, delaunay_img, delaunay_img2, all_points, ori_points = getEyebrowsMorphing_v2(shape, shape2, img1, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask1)
        seamless_img = cv2.seamlessClone(output, img1, src_mask, center, cv2.NORMAL_CLONE)
        
        output, mask2, delaunay_img, delaunay_img2, all_points, ori_points = getMouthMorphing(shape, shape2, seamless_img, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask2)
        output = cv2.seamlessClone(output, seamless_img, src_mask, center, cv2.NORMAL_CLONE)
        mask = mask1 + mask2

    if facepart_type=='eyes-nose' :
        output, mask1, delaunay_img, delaunay_img2, all_points, ori_points = getEyesMorphing(shape, shape2, img1, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask1)
        seamless_img = cv2.seamlessClone(output, img1, src_mask, center, cv2.NORMAL_CLONE)
        
        output, mask2, delaunay_img, delaunay_img2, all_points, ori_points = getNoseMorphing(shape, shape2, seamless_img, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask2)
        output = cv2.seamlessClone(output, seamless_img, src_mask, center, cv2.NORMAL_CLONE)
        mask = mask1 + mask2

    if facepart_type=='eyes-mouth' :
        output, mask1, delaunay_img, delaunay_img2, all_points, ori_points = getEyesMorphing(shape, shape2, img1, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask1)
        seamless_img = cv2.seamlessClone(output, img1, src_mask, center, cv2.NORMAL_CLONE)
        
        output, mask2, delaunay_img, delaunay_img2, all_points, ori_points = getMouthMorphing(shape, shape2, seamless_img, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask2)
        output = cv2.seamlessClone(output, seamless_img, src_mask, center, cv2.NORMAL_CLONE)
        mask = mask1 + mask2

    if facepart_type=='nose-mouth' :
        output, mask1, delaunay_img, delaunay_img2, all_points, ori_points = getNoseMorphing(shape, shape2, img1, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask1)
        seamless_img = cv2.seamlessClone(output, img1, src_mask, center, cv2.NORMAL_CLONE)
        
        output, mask2, delaunay_img, delaunay_img2, all_points, ori_points = getMouthMorphing(shape, shape2, seamless_img, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask2)
        output = cv2.seamlessClone(output, seamless_img, src_mask, center, cv2.NORMAL_CLONE)
        mask = mask1 + mask2

    if facepart_type=='eyebrows-eyes-nose' :
        output, mask1, delaunay_img, delaunay_img2, all_points, ori_points = getEyebrowsMorphing_v2(shape, shape2, img1, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask1)
        seamless_img = cv2.seamlessClone(output, img1, src_mask, center, cv2.NORMAL_CLONE)
        
        output, mask2, delaunay_img, delaunay_img2, all_points, ori_points = getEyesMorphing(shape, shape2, seamless_img, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask2)
        seamless_img2 = cv2.seamlessClone(output, seamless_img, src_mask, center, cv2.NORMAL_CLONE)

        output, mask3, delaunay_img, delaunay_img2, all_points, ori_points = getNoseMorphing(shape, shape2, seamless_img2, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask3)
        output = cv2.seamlessClone(output, seamless_img2, src_mask, center, cv2.NORMAL_CLONE)
        mask = mask1 + mask2 + mask3

    if facepart_type=='eyebrows-eyes-mouth' :
        output, mask1, delaunay_img, delaunay_img2, all_points, ori_points = getEyebrowsMorphing_v2(shape, shape2, img1, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask1)
        seamless_img = cv2.seamlessClone(output, img1, src_mask, center, cv2.NORMAL_CLONE)
        
        output, mask2, delaunay_img, delaunay_img2, all_points, ori_points = getEyesMorphing(shape, shape2, seamless_img, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask2)
        seamless_img2 = cv2.seamlessClone(output, seamless_img, src_mask, center, cv2.NORMAL_CLONE)

        output, mask3, delaunay_img, delaunay_img2, all_points, ori_points = getMouthMorphing(shape, shape2, seamless_img2, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask3)
        output = cv2.seamlessClone(output, seamless_img2, src_mask, center, cv2.NORMAL_CLONE)
        mask = mask1 + mask2 + mask3

    if facepart_type=='eyebrows-nose-mouth' :
        output, mask1, delaunay_img, delaunay_img2, all_points, ori_points = getEyebrowsMorphing_v2(shape, shape2, img1, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask1)
        seamless_img = cv2.seamlessClone(output, img1, src_mask, center, cv2.NORMAL_CLONE)
        
        output, mask2, delaunay_img, delaunay_img2, all_points, ori_points = getNoseMorphing(shape, shape2, seamless_img, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask2)
        seamless_img2 = cv2.seamlessClone(output, seamless_img, src_mask, center, cv2.NORMAL_CLONE)

        output, mask3, delaunay_img, delaunay_img2, all_points, ori_points = getMouthMorphing(shape, shape2, seamless_img2, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask3)
        output = cv2.seamlessClone(output, seamless_img2, src_mask, center, cv2.NORMAL_CLONE)
        mask = mask1 + mask2 + mask3

    if facepart_type=='eyes-nose-mouth' :
        output, mask1, delaunay_img, delaunay_img2, all_points, ori_points = getEyesMorphing(shape, shape2, img1, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask1)
        seamless_img = cv2.seamlessClone(output, img1, src_mask, center, cv2.NORMAL_CLONE)
        
        output, mask2, delaunay_img, delaunay_img2, all_points, ori_points = getNoseMorphing(shape, shape2, seamless_img, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask2)
        seamless_img2 = cv2.seamlessClone(output, seamless_img, src_mask, center, cv2.NORMAL_CLONE)

        output, mask3, delaunay_img, delaunay_img2, all_points, ori_points = getMouthMorphing(shape, shape2, seamless_img2, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask3)
        output = cv2.seamlessClone(output, seamless_img2, src_mask, center, cv2.NORMAL_CLONE)
        mask = mask1 + mask2 + mask3

    if facepart_type=='eyebrows-eyes-nose-mouth' :
        output_1, mask1, delaunay_img, delaunay_img2, all_points, ori_points = getEyebrowsMorphing_v2(shape, shape2, img1, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask1)
        seamless_img = cv2.seamlessClone(output_1, img1, src_mask, center, cv2.NORMAL_CLONE)
        
        output, mask2, delaunay_img, delaunay_img2, all_points, ori_points = getEyesMorphing(shape, shape2, seamless_img, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        output_2, t_mask2, t_delaunay_img, t_delaunay_img2, t_all_points, t_ori_points = getEyesMorphing(shape, shape2, output_1, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask2)
        seamless_img2 = cv2.seamlessClone(output, seamless_img, src_mask, center, cv2.NORMAL_CLONE)

        output, mask3, delaunay_img, delaunay_img2, all_points, ori_points = getNoseMorphing(shape, shape2, seamless_img2, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        output_3, t_mask3, t_delaunay_img, t_delaunay_img2, t_all_points, t_ori_points = getNoseMorphing(shape, shape2, output_2, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask3)
        seamless_img3 = cv2.seamlessClone(output, seamless_img2, src_mask, center, cv2.NORMAL_CLONE)

        output, mask4, delaunay_img, delaunay_img2, all_points, ori_points = getMouthMorphing(shape, shape2, seamless_img3, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        output_0, t_mask4, t_delaunay_img, t_delaunay_img2, t_all_points, t_ori_points = getMouthMorphing(shape, shape2, output_3, img2, mask_img, delaunay_img, delaunay_img2, alpha)
        center, src_mask = getMaskCenter(img1, mask4)
        output = cv2.seamlessClone(output, seamless_img3, src_mask, center, cv2.NORMAL_CLONE)        
        mask = mask1 + mask2 + mask3 + mask4

    # return the aligned face
    return output, output_0, mask, delaunay_img, delaunay_img2, all_points, ori_points