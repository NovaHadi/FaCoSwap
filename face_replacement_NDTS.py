# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 05:00:12 2022

@author: N.H.Lestriandoko

Non Discriminative Texture and Shape (NDTS and iNDTS)
"""
#from imutils import face_utils

import streamlit as st
import numpy as np
#import dlib
#import matplotlib.pyplot as plt
#import imageio
#import os
import cv2
#import math
import face_morphing as mp
from imutils import face_utils
from face_alignment import manual_aligning_68_v3

@st.cache_data
def getMaskCenter(img1, mask):
    src_mask = np.zeros(img1.shape, img1.dtype)
    src_mask[mask>0] = 255
    poly = np.argwhere(mask[:,:,2]>0)
    r = cv2.boundingRect(np.float32([poly]))    
    center = (r[1]+int(r[3]/2)), (r[0]+int(r[2]/2))
    return center, src_mask

@st.cache_data
def getEyebrowsMorphing_v1(shape, shape2, img1, img2, d_img1, d_img2, alpha):
    w = 6
    w3 = 6
    s = 2
    mask_img = np.zeros(img1.shape, dtype = np.float32)
    
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
                                 [r_eye_pts[0][0]-s*w,r_eye_pts[0][1]-w3],
                                 [r_eye_pts[1][0],r_eye_pts[1][1]-w3]], axis=0)

    avg_r_pts_1 = shape2[17:20]
    avg_r_pts_1 = np.append(avg_r_pts_1,[[points2[0][0],points2[0][1]-w],
                                 [points2[1][0],points2[1][1]-w],
                                 [points2[2][0],points2[2][1]-w],
                                 [r_eye_pts2[0][0]-s*w,r_eye_pts2[0][1]-w3],
                                 [r_eye_pts2[1][0],r_eye_pts2[1][1]-w3]], axis=0)
    output, mask, d_img1, d_img2 = mp.morphing_original(img1, img2, d_img1, d_img2, mask_img, r_pts_1, avg_r_pts_1, alpha)    

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
    output, mask,  d_img1, d_img2 = mp.morphing_original(output, img2, d_img1, d_img2, mask, r_pts_2, avg_r_pts_2, alpha)    
    
    # right eyebrow - area 3 :
    r_pts_3 = shape[20:22]
    r_pts_3 = np.append(r_pts_3,[[points1[3][0],points1[3][1]-w],
                                 [points1[4][0]+w,points1[4][1]-w],
                                 [r_eye_pts[2][0],r_eye_pts[2][1]-w3],
                                 [r_eye_pts[3][0]+s*w,r_eye_pts[3][1]-w3]], axis=0)
                                 

    avg_r_pts_3 = shape2[20:22]
    avg_r_pts_3 = np.append(avg_r_pts_3,[[points2[3][0],points2[3][1]-w],
                                 [points2[4][0]+w,points2[4][1]-w],
                                 [r_eye_pts2[2][0],r_eye_pts2[2][1]-w3],
                                 [r_eye_pts2[3][0]+s*w,r_eye_pts2[3][1]-w3]], axis=0)
    output, mask, d_img1, d_img2 = mp.morphing_original(output, img2, d_img1, d_img2, mask, r_pts_3, avg_r_pts_3, alpha)    

    # =============== LEFT EYEBROW =================    
    points1a = shape[22:27] 
    points2a = shape2[22:27]
    l_eye_pts = shape[42:48]       
    l_eye_pts2 = shape2[42:48]       

    # left eyebrow - area 1 :
    l_pts_1 = shape[22:24]
    l_pts_1 = np.append(l_pts_1,[[points1a[0][0]-w,points1a[0][1]-w],
                                 [points1a[1][0],points1a[1][1]-w],
                                 [l_eye_pts[0][0]-s*w,l_eye_pts[0][1]-w3],
                                 [l_eye_pts[1][0],l_eye_pts[1][1]-w3]], axis=0)
        
    avg_l_pts_1 = shape2[22:24]
    avg_l_pts_1 = np.append(avg_l_pts_1,[[points2a[0][0]-w,points2a[0][1]-w],
                                 [points2a[1][0],points2a[1][1]-w],
                                 [l_eye_pts2[0][0]-s*w,l_eye_pts2[0][1]-w3],
                                 [l_eye_pts2[1][0],l_eye_pts2[1][1]-w3]], axis=0)
    output, mask, d_img1, d_img2 = mp.morphing_original(output, img2, d_img1, d_img2, mask, l_pts_1, avg_l_pts_1, alpha)    

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
    output, mask, d_img1, d_img2 = mp.morphing_original(output, img2, d_img1, d_img2, mask, l_pts_2, avg_l_pts_2, alpha)    
    
    # left eyebrow - area 3 :
    l_pts_3 = shape[24:27]
    l_pts_3 = np.append(l_pts_3,[[points1a[2][0],points1a[2][1]-w],
                                   [points1a[3][0],points1a[3][1]-w],
                                   [points1a[4][0],points1a[4][1]-w],
                                   [l_eye_pts[2][0],l_eye_pts[2][1]-w3],
                                   [l_eye_pts[3][0]+s*w,l_eye_pts[3][1]-w3]], axis=0)

    avg_l_pts_3 = shape2[24:27]
    avg_l_pts_3 = np.append(avg_l_pts_3,[[points2a[2][0],points2a[2][1]-w],
                                   [points2a[3][0],points2a[3][1]-w],
                                   [points2a[4][0],points2a[4][1]-w],
                                   [l_eye_pts2[2][0],l_eye_pts2[2][1]-w3],
                                   [l_eye_pts2[3][0]+s*w,l_eye_pts2[3][1]-w3]], axis=0)
    output, mask, d_img1, d_img2 = mp.morphing_original(output, img2, d_img1, d_img2, mask, l_pts_3, avg_l_pts_3, alpha)    


    all_points = np.concatenate((r_pts_1,r_pts_2,r_pts_3,l_pts_1,l_pts_2,l_pts_3), axis=0)
    ori_points = shape[17:27]
    
    return output, mask, d_img1, d_img2, all_points, ori_points

@st.cache_data
def get_right_EyebrowsMorphing(shape, shape2, img1, img2, d_img1, d_img2, alpha):
    w = 6
    w3 = 6
    s = 2
    mask_img = np.zeros(img1.shape, dtype = np.float32)

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
                                 [r_eye_pts[0][0]-s*w,r_eye_pts[0][1]-w3],
                                 [r_eye_pts[0][0],r_eye_pts[0][1]],
                                 [r_eye_pts[1][0],r_eye_pts[1][1]]], axis=0)

    avg_r_pts_1 = shape2[17:20]
    avg_r_pts_1 = np.append(avg_r_pts_1,[[points2[0][0],points2[0][1]-w],
                                 [points2[1][0],points2[1][1]-w],
                                 [points2[2][0],points2[2][1]-w],
                                 [r_eye_pts2[0][0]-s*w,r_eye_pts2[0][1]-w3],
                                 [r_eye_pts2[0][0],r_eye_pts2[0][1]],
                                 [r_eye_pts2[1][0],r_eye_pts2[1][1]]], axis=0)
    output, mask, d_img1, d_img2 = mp.morphing_original(img1, img2, d_img1, d_img2, mask_img, r_pts_1, avg_r_pts_1, alpha)    

    # right eyebrow - area 2 :
    r_pts_2 = shape[19:21]
    r_pts_2 = np.append(r_pts_2,[[points1[2][0],points1[2][1]-w],
                                 [points1[3][0],points1[3][1]-w],
                                 [r_eye_pts[1][0],r_eye_pts[1][1]],
                                 [r_eye_pts[2][0],r_eye_pts[2][1]]], axis=0)

    avg_r_pts_2 = shape2[19:21]
    avg_r_pts_2 = np.append(avg_r_pts_2,[[points2[2][0],points2[2][1]-w],
                                 [points2[3][0],points2[3][1]-w],
                                 [r_eye_pts2[1][0],r_eye_pts2[1][1]],
                                 [r_eye_pts2[2][0],r_eye_pts2[2][1]]], axis=0)
    output, mask, d_img1, d_img2 = mp.morphing_original(output, img2, d_img1, d_img2, mask, r_pts_2, avg_r_pts_2, alpha)    
    
    # right eyebrow - area 3 :
    r_pts_3 = shape[20:22]
    r_pts_3 = np.append(r_pts_3,[[points1[3][0],points1[3][1]-w],
                                 [points1[4][0]+w,points1[4][1]-w],
                                 [r_eye_pts[2][0],r_eye_pts[2][1]],
                                 [r_eye_pts[3][0],r_eye_pts[3][1]],
                                 [r_eye_pts[3][0]+s*w,r_eye_pts[3][1]-w3]], axis=0)
                                 

    avg_r_pts_3 = shape2[20:22]
    avg_r_pts_3 = np.append(avg_r_pts_3,[[points2[3][0],points2[3][1]-w],
                                 [points2[4][0]+w,points2[4][1]-w],
                                 [r_eye_pts2[2][0],r_eye_pts2[2][1]],
                                 [r_eye_pts2[3][0],r_eye_pts2[3][1]],
                                 [r_eye_pts2[3][0]+s*w,r_eye_pts2[3][1]-w3]], axis=0)
    output, mask, d_img1, d_img2 = mp.morphing_original(output, img2, d_img1, d_img2, mask, r_pts_3, avg_r_pts_3, alpha)    

    all_points = np.concatenate((r_pts_1,r_pts_2,r_pts_3), axis=0)
    ori_points = shape[17:22]
    
    return output, mask, d_img1, d_img2, all_points, ori_points

@st.cache_data
def get_left_EyebrowsMorphing(shape, shape2, img1, img2, d_img1, d_img2, alpha):
    w = 6
    w3 = 6
    s = 2
    mask_img = np.zeros(img1.shape, dtype = np.float32)
    # divide an eyebrow into three morphing areas
    # =============== LEFT EYEBROW =================    
    points1a = shape[22:27] 
    points2a = shape2[22:27]
    l_eye_pts = shape[42:48]       
    l_eye_pts2 = shape2[42:48]       

    # left eyebrow - area 1 :
    l_pts_1 = shape[22:24]
    l_pts_1 = np.append(l_pts_1,[[points1a[0][0]-w,points1a[0][1]-w],
                                 [points1a[1][0],points1a[1][1]-w],
                                 [l_eye_pts[0][0]-s*w,l_eye_pts[0][1]-w3],
                                 [l_eye_pts[0][0],l_eye_pts[0][1]],
                                 [l_eye_pts[1][0],l_eye_pts[1][1]]], axis=0)
        
    avg_l_pts_1 = shape2[22:24]
    avg_l_pts_1 = np.append(avg_l_pts_1,[[points2a[0][0]-w,points2a[0][1]-w],
                                 [points2a[1][0],points2a[1][1]-w],
                                 [l_eye_pts2[0][0]-s*w,l_eye_pts2[0][1]-w3],
                                 [l_eye_pts2[0][0],l_eye_pts2[0][1]],
                                 [l_eye_pts2[1][0],l_eye_pts2[1][1]]], axis=0)
    output, mask, d_img1, d_img2 = mp.morphing_original(img1, img2, d_img1, d_img2, mask_img, l_pts_1, avg_l_pts_1, alpha)    

    # left eyebrow - area 2 :
    l_pts_2 = shape[23:25]
    l_pts_2 = np.append(l_pts_2,[[points1a[1][0],points1a[1][1]-w],
                                   [points1a[2][0],points1a[2][1]-w],
                                   [l_eye_pts[1][0],l_eye_pts[1][1]],
                                   [l_eye_pts[2][0],l_eye_pts[2][1]]], axis=0)

    avg_l_pts_2 = shape2[23:25]
    avg_l_pts_2 = np.append(avg_l_pts_2,[[points2a[1][0],points2a[1][1]-w],
                                   [points2a[2][0],points2a[2][1]-w],
                                   [l_eye_pts2[1][0],l_eye_pts2[1][1]],
                                   [l_eye_pts2[2][0],l_eye_pts2[2][1]]], axis=0)
    output, mask, d_img1, d_img2 = mp.morphing_original(output, img2, d_img1, d_img2, mask, l_pts_2, avg_l_pts_2, alpha)    
    
    # left eyebrow - area 3 :
    l_pts_3 = shape[24:27]
    l_pts_3 = np.append(l_pts_3,[[points1a[2][0],points1a[2][1]-w],
                                   [points1a[3][0],points1a[3][1]-w],
                                   [points1a[4][0],points1a[4][1]-w],
                                   [l_eye_pts[2][0],l_eye_pts[2][1]],
                                   [l_eye_pts[3][0],l_eye_pts[3][1]],
                                   [l_eye_pts[3][0]+s*w,l_eye_pts[3][1]-w3]], axis=0)

    avg_l_pts_3 = shape2[24:27]
    avg_l_pts_3 = np.append(avg_l_pts_3,[[points2a[2][0],points2a[2][1]-w],
                                   [points2a[3][0],points2a[3][1]-w],
                                   [points2a[4][0],points2a[4][1]-w],
                                   [l_eye_pts2[2][0],l_eye_pts2[2][1]],
                                   [l_eye_pts2[3][0],l_eye_pts2[3][1]],
                                   [l_eye_pts2[3][0]+s*w,l_eye_pts2[3][1]-w3]], axis=0)
    output, mask, d_img1, d_img2 = mp.morphing_original(output, img2, d_img1, d_img2, mask, l_pts_3, avg_l_pts_3, alpha)    


    all_points = np.concatenate((l_pts_1,l_pts_2,l_pts_3), axis=0)
    ori_points = shape[22:27]
    
    return output, mask, d_img1, d_img2, all_points, ori_points

@st.cache_data
def get_right_EyesMorphing(shape, shape2, img1, img2, d_img1, d_img2, alpha):
    mask_img = np.zeros(img1.shape, dtype = np.float32)
    w = 6
    s = 2
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
    output, mask, d_img1, d_img2 = mp.morphing_original(img1, img2, d_img1, d_img2, mask_img, points1, points2, alpha)    

    all_points = points1
    ori_points = shape[36:42]
    
    return output, mask, d_img1, d_img2, all_points, ori_points

@st.cache_data
def get_left_EyesMorphing(shape, shape2, img1, img2, d_img1, d_img2, alpha):
    mask_img = np.zeros(img1.shape, dtype = np.float32)
    w = 6
    s = 2
    points1 = shape[42:48]
    points1 = np.append(points1,[[points1[0][0]-s*w,points1[0][1]],
                                 [points1[1][0],points1[1][1]-w],
                                 [points1[2][0],points1[2][1]-w],
                                 [points1[3][0]+s*w,points1[3][1]],
                                 [points1[4][0],points1[4][1]+w],
                                 [points1[5][0],points1[5][1]+w]], axis=0)
    points2 = shape2[42:48]
    points2 = np.append(points2,[[points2[0][0]-s*w,points2[0][1]],
                                 [points2[1][0],points2[1][1]-w],
                                 [points2[2][0],points2[2][1]-w],
                                 [points2[3][0]+s*w,points2[3][1]],
                                 [points2[4][0],points2[4][1]+w],
                                 [points2[5][0],points2[5][1]+w]], axis=0)
    output, mask, d_img1, d_img2 = mp.morphing_original(img1, img2, d_img1, d_img2, mask_img, points1, points2, alpha)    

    all_points = points1
    ori_points = shape[42:48]
    
    return output, mask, d_img1, d_img2, all_points, ori_points

@st.cache_data
def get_NoseMorphing(shape, shape2, img1, img2, d_img1, d_img2, alpha):
    mask_img = np.zeros(img1.shape, dtype = np.float32)
    w = 8
    points1 = shape[27:36]
    points1 = np.append(points1,[[points1[4][0]-w,points1[4][1]],
                                 [points1[6][0],points1[6][1]+w/2],
                                 [points1[8][0]+w,points1[8][1]]], axis=0)

    points2 = shape2[27:36]
    points2 = np.append(points2,[[points2[4][0]-w,points2[4][1]],
                                 [points2[6][0],points2[6][1]+w/2],
                                 [points2[8][0]+w,points2[8][1]]], axis=0)
    output, mask, d_img1, d_img2 = mp.morphing_original(img1, img2, d_img1, d_img2, mask_img, points1, points2, alpha)    

    all_points = points1
    ori_points = shape[42:48]
    
    return output, mask, d_img1, d_img2, all_points, ori_points

@st.cache_data
def get_MouthMorphing(shape, shape2, img1, img2, d_img1, d_img2, alpha):
    mask_img = np.zeros(img1.shape, dtype = np.float32)
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
    output, mask, d_img1, d_img2 = mp.morphing_original(img1, img2, d_img1, d_img2, mask_img, points1, points2, alpha)    

    all_points = points1
    ori_points = shape[42:48]
    
    return output, mask, d_img1, d_img2, all_points, ori_points

@st.cache_data
def eyebrows(shape, shape2, img1, img2, d_img1, d_img2):
    # replace the source landmarks with the destination landmarks
    shape3 = shape.copy()
    shape3[17:27] = shape2[17:27] #eyebrows
    morphed_landmark= manual_aligning_68_v3(img1, shape, shape3) #eyebrows
    # face part morphing
    output_r, mask_r,  d_img1, d_img2, all_points, ori_points = get_right_EyebrowsMorphing(shape, shape2, img1, img2, d_img1, d_img2, alpha=1)
    center, src_mask_r = getMaskCenter(morphed_landmark, mask_r)
    output_seamless_r = cv2.seamlessClone(output_r, morphed_landmark, src_mask_r, center, cv2.NORMAL_CLONE)
    
    output_l, mask_l,  d_img1, d_img2, all_points, ori_points = get_left_EyebrowsMorphing(shape, shape2, output_seamless_r, img2, d_img1, d_img2, alpha=1)
    output_replacement, t_mask_l,  t_d_img1, t_d_img2, t_all_points, t_ori_points = get_left_EyebrowsMorphing(shape, shape2, output_r, img2, d_img1, d_img2, alpha=1)
    center, src_mask_l = getMaskCenter(output_seamless_r, mask_l)
    output = cv2.seamlessClone(output_l, output_seamless_r, src_mask_l, center, cv2.NORMAL_CLONE)
    mask = mask_r + mask_l 
    
    return output, d_img1, d_img2, mask, output_replacement

@st.cache_data
def eyes(shape, shape2, img1, img2, d_img1, d_img2):
    # replace the source landmarks with the destination landmarks
    shape3 = shape.copy()
    shape3[36:48] = shape2[36:48] #eyes
    morphed_landmark = manual_aligning_68_v3(img1, shape, shape3)
    # face part morphing
    output_r, mask_r,  d_img1, d_img2, all_points, ori_points = get_right_EyesMorphing(shape, shape2, img1, img2, d_img1, d_img2, alpha=1)
    center, src_mask_r = getMaskCenter(morphed_landmark, mask_r)
    output_seamless_r = cv2.seamlessClone(output_r, morphed_landmark, src_mask_r, center, cv2.NORMAL_CLONE)
    
    output_l, mask_l,  d_img1, d_img2, all_points, ori_points = get_left_EyesMorphing(shape, shape2, output_seamless_r, img2, d_img1, d_img2, alpha=1)
    output_replacement, mask_l,  d_img1, d_img2, all_points, ori_points = get_left_EyesMorphing(shape, shape2, output_r, img2, d_img1, d_img2, alpha=1)
    center, src_mask_l = getMaskCenter(output_seamless_r, mask_l)
    output = cv2.seamlessClone(output_l, output_seamless_r, src_mask_l, center, cv2.NORMAL_CLONE)
    mask = mask_r + mask_l
    
    return output, d_img1, d_img2, mask, output_replacement

@st.cache_data
def nose(shape, shape2, img1, img2, d_img1, d_img2):
    # replace the source landmarks with the destination landmarks
    shape3 = shape.copy()
    shape3[27:36] = shape2[27:36] #nose
    morphed_landmark = manual_aligning_68_v3(img1, shape, shape3) #eyebrows
    # face part morphing
    output_replacement, mask,  d_img1, d_img2, all_points, ori_points = get_NoseMorphing(shape, shape2, img1, img2, d_img1, d_img2, alpha=1)
    center, src_mask = getMaskCenter(morphed_landmark, mask)
    output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)
    
    return output, d_img1, d_img2, mask, output_replacement

@st.cache_data
def mouth(shape, shape2, img1, img2, d_img1, d_img2):
    # replace the source landmarks with the destination landmarks
    shape3 = shape.copy()
    shape3[48:60] = shape2[48:60] #mouth
    morphed_landmark = manual_aligning_68_v3(img1, shape, shape3) #eyebrows
    # face part morphing
    output_replacement, mask, d_img1, d_img2, all_points, ori_points = get_MouthMorphing(shape, shape2, img1, img2, d_img1, d_img2, alpha=1)
    center, src_mask = getMaskCenter(morphed_landmark, mask)
    output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)
    
    return output, d_img1, d_img2, mask, output_replacement

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
def LmPt_Morph(img1, img2, parts, _detector, _predictor):
    img1 = np.uint8(img1)
    img2 = np.uint8(img2)
    d_img1 = img1.copy()
    d_img2 = img2.copy()
    face_rects1 = _detector(img1, 2)
    if len(face_rects1)>0:
        face_rect1 = check_multiple_faces(face_rects1)                    
    
    face_rects2 = _detector(img2, 2)
    if len(face_rects2)>0:
        face_rect2 = check_multiple_faces(face_rects2)                    
    
    landmarks = _predictor(img1, face_rect1)
    landmarks2 = _predictor(img2, face_rect2)
    # to np.array
    shape = face_utils.shape_to_np(landmarks)
    shape2 = face_utils.shape_to_np(landmarks2) 
    output = img1.copy()

    if parts == 'eyebrows':
        output, d_img1, d_img2, mask, output_replacement = eyebrows(shape, shape2, img1, img2, d_img1, d_img2)
        
        shape3 = shape.copy()
        shape3[17:27] = shape2[17:27] #eyebrows
        morphed_img = manual_aligning_68_v3(img1, shape, shape3) 

    elif parts == 'eyes':
        output, d_img1, d_img2, mask, output_replacement = eyes(shape, shape2, img1, img2, d_img1, d_img2)

        shape3 = shape.copy()
        shape3[36:48] = shape2[36:48] #eyes
        morphed_img = manual_aligning_68_v3(img1, shape, shape3) 

    elif parts == 'nose':
        output, d_img1, d_img2, mask, output_replacement = nose(shape, shape2, img1, img2, d_img1, d_img2)

        shape3 = shape.copy()
        shape3[27:36] = shape2[27:36] #nose
        morphed_img = manual_aligning_68_v3(img1, shape, shape3) 

    elif parts == 'mouth':
        output, d_img1, d_img2, mask, output_replacement = mouth(shape, shape2, img1, img2, d_img1, d_img2)

        shape3 = shape.copy()
        shape3[48:60] = shape2[48:60] #mouth
        morphed_img = manual_aligning_68_v3(img1, shape, shape3) 
        
    elif parts == 'eyebrows-eyes':
        output_0, d_img1, d_img2, mask1, output_replacement_0 = eyebrows(shape, shape2, img1, img2, d_img1, d_img2)

        output, d_img1, d_img2, mask2, output_replacement = eyes(shape, shape2, output_0, img2, d_img1, d_img2)
        t_output, t_d_img1, t_d_img2, t_mask2, output_replacement = eyes(shape, shape2, output_replacement_0, img2, d_img1, d_img2)
        mask = mask1 + mask2

        shape3 = shape.copy()
        shape3[17:27] = shape2[17:27] #eyebrows
        shape3[36:48] = shape2[36:48] #eyes
        #shape3[27:36] = shape2[27:36] #nose
        #shape3[48:60] = shape2[48:60] #mouth
        morphed_img = manual_aligning_68_v3(img1, shape, shape3) 
    
    elif parts == 'eyebrows-nose':
        output_0, d_img1, d_img2, mask1, output_replacement_0 = eyebrows(shape, shape2, img1, img2, d_img1, d_img2)

        output, d_img1, d_img2, mask2, output_replacement = nose(shape, shape2, output_0, img2, d_img1, d_img2)
        t_output, t_d_img1, t_d_img2, t_mask2, output_replacement = nose(shape, shape2, output_replacement_0, img2, d_img1, d_img2)
        mask = mask1 + mask2

        shape3 = shape.copy()
        shape3[17:27] = shape2[17:27] #eyebrows
        #shape3[36:48] = shape2[36:48] #eyes
        shape3[27:36] = shape2[27:36] #nose
        #shape3[48:60] = shape2[48:60] #mouth
        morphed_img = manual_aligning_68_v3(img1, shape, shape3) 

    elif parts == 'eyebrows-mouth':
        output_0, d_img1, d_img2, mask1, output_replacement_0 = eyebrows(shape, shape2, img1, img2, d_img1, d_img2)

        output, d_img1, d_img2, mask2, output_replacement = mouth(shape, shape2, output_0, img2, d_img1, d_img2)
        t_output, t_d_img1, t_d_img2, t_mask2, output_replacement = mouth(shape, shape2, output_replacement_0, img2, d_img1, d_img2)
        mask = mask1 + mask2

        shape3 = shape.copy()
        shape3[17:27] = shape2[17:27] #eyebrows
        #shape3[36:48] = shape2[36:48] #eyes
        #shape3[27:36] = shape2[27:36] #nose
        shape3[48:60] = shape2[48:60] #mouth
        morphed_img = manual_aligning_68_v3(img1, shape, shape3) 

    elif parts == 'eyes-nose':
        output_0, d_img1, d_img2, mask1, output_replacement_0 = eyes(shape, shape2, img1, img2, d_img1, d_img2)

        output, d_img1, d_img2, mask2, output_replacement = nose(shape, shape2, output_0, img2, d_img1, d_img2)
        t_output, t_d_img1, t_d_img2, t_mask2, output_replacement = nose(shape, shape2, output_replacement_0, img2, d_img1, d_img2)
        mask = mask1 + mask2

        shape3 = shape.copy()
        #shape3[17:27] = shape2[17:27] #eyebrows
        shape3[36:48] = shape2[36:48] #eyes
        shape3[27:36] = shape2[27:36] #nose
        #shape3[48:60] = shape2[48:60] #mouth
        morphed_img = manual_aligning_68_v3(img1, shape, shape3) 


    elif parts == 'eyes-mouth':
        output_0, d_img1, d_img2, mask1, output_replacement_0 = eyes(shape, shape2, img1, img2, d_img1, d_img2)

        output, d_img1, d_img2, mask2, output_replacement = mouth(shape, shape2, output_0, img2, d_img1, d_img2)
        t_output, t_d_img1, t_d_img2, t_mask2, output_replacement = mouth(shape, shape2, output_replacement_0, img2, d_img1, d_img2)
        mask = mask1 + mask2

        shape3 = shape.copy()
        #shape3[17:27] = shape2[17:27] #eyebrows
        shape3[36:48] = shape2[36:48] #eyes
        #shape3[27:36] = shape2[27:36] #nose
        shape3[48:60] = shape2[48:60] #mouth
        morphed_img = manual_aligning_68_v3(img1, shape, shape3) 

    elif parts == 'nose-mouth':
        output_0, d_img1, d_img2, mask1, output_replacement_0 = nose(shape, shape2, img1, img2, d_img1, d_img2)

        output, d_img1, d_img2, mask2, output_replacement = mouth(shape, shape2, output_0, img2, d_img1, d_img2)
        t_output, t_d_img1, t_d_img2, t_mask2, output_replacement = mouth(shape, shape2, output_replacement_0, img2, d_img1, d_img2)
        mask = mask1 + mask2

        shape3 = shape.copy()
        #shape3[17:27] = shape2[17:27] #eyebrows
        #shape3[36:48] = shape2[36:48] #eyes
        shape3[27:36] = shape2[27:36] #nose
        shape3[48:60] = shape2[48:60] #mouth
        morphed_img = manual_aligning_68_v3(img1, shape, shape3) 

    elif parts == 'eyebrows-eyes-nose':
        output_0, d_img1, d_img2, mask1, output_replacement_0 = eyebrows(shape, shape2, img1, img2, d_img1, d_img2)

        output_1, d_img1, d_img2, mask2, output_replacement = eyes(shape, shape2, output_0, img2, d_img1, d_img2)
        t_output_1, t_d_img1, t_d_img2, t_mask2, output_replacement_1 = eyes(shape, shape2, output_replacement_0, img2, d_img1, d_img2)

        output, d_img1, d_img2, mask3, output_replacement = nose(shape, shape2, output_1, img2, d_img1, d_img2)
        t_output, t_d_img1, t_d_img2, t_mask3, output_replacement = nose(shape, shape2, output_replacement_1, img2, d_img1, d_img2)
        mask = mask1 + mask2 + mask3
 
        shape3 = shape.copy()
        shape3[17:27] = shape2[17:27] #eyebrows
        shape3[36:48] = shape2[36:48] #eyes
        shape3[27:36] = shape2[27:36] #nose
        #shape3[48:60] = shape2[48:60] #mouth
        morphed_img = manual_aligning_68_v3(img1, shape, shape3) 

    elif parts == 'eyebrows-eyes-mouth':
        output_0, d_img1, d_img2, mask1, output_replacement_0 = eyebrows(shape, shape2, img1, img2, d_img1, d_img2)

        output_1, d_img1, d_img2, mask2, output_replacement = eyes(shape, shape2, output_0, img2, d_img1, d_img2)
        t_output_1, t_d_img1, t_d_img2, t_mask2, output_replacement_1 = eyes(shape, shape2, output_replacement_0, img2, d_img1, d_img2)

        output, d_img1, d_img2, mask3, output_replacement = mouth(shape, shape2, output_1, img2, d_img1, d_img2)
        t_output, t_d_img1, t_d_img2, t_mask3, output_replacement = mouth(shape, shape2, output_replacement_1, img2, d_img1, d_img2)
        mask = mask1 + mask2 + mask3

        shape3 = shape.copy()
        shape3[17:27] = shape2[17:27] #eyebrows
        shape3[36:48] = shape2[36:48] #eyes
        #shape3[27:36] = shape2[27:36] #nose
        shape3[48:60] = shape2[48:60] #mouth
        morphed_img = manual_aligning_68_v3(img1, shape, shape3) 
    
    elif parts == 'eyebrows-nose-mouth':
        output_0, d_img1, d_img2, mask1, output_replacement_0 = eyebrows(shape, shape2, img1, img2, d_img1, d_img2)

        output_1, d_img1, d_img2, mask2, output_replacement = nose(shape, shape2, output_0, img2, d_img1, d_img2)
        t_output_1, t_d_img1, t_d_img2, t_mask2, output_replacement_1 = nose(shape, shape2, output_replacement_0, img2, d_img1, d_img2)

        output, d_img1, d_img2, mask3, output_replacement = mouth(shape, shape2, output_1, img2, d_img1, d_img2)
        t_output, t_d_img1, t_d_img2, t_mask3, output_replacement = mouth(shape, shape2, output_replacement_1, img2, d_img1, d_img2)
        mask = mask1 + mask2 + mask3

        shape3 = shape.copy()
        shape3[17:27] = shape2[17:27] #eyebrows
        #shape3[36:48] = shape2[36:48] #eyes
        shape3[27:36] = shape2[27:36] #nose
        shape3[48:60] = shape2[48:60] #mouth
        morphed_img = manual_aligning_68_v3(img1, shape, shape3) 
    
    elif parts == 'eyes-nose-mouth':
        output_0, d_img1, d_img2, mask1, output_replacement_0 = eyes(shape, shape2, img1, img2, d_img1, d_img2)

        output_1, d_img1, d_img2, mask2, output_replacement = nose(shape, shape2, output_0, img2, d_img1, d_img2)
        t_output_1, t_d_img1, t_d_img2, t_mask2, output_replacement_1 = nose(shape, shape2, output_replacement_0, img2, d_img1, d_img2)

        output, d_img1, d_img2, mask3, output_replacement = mouth(shape, shape2, output_1, img2, d_img1, d_img2)
        t_output, t_d_img1, t_d_img2, t_mask3, output_replacement = mouth(shape, shape2, output_replacement_1, img2, d_img1, d_img2)
        mask = mask1 + mask2 + mask3

        shape3 = shape.copy()
        #shape3[17:27] = shape2[17:27] #eyebrows
        shape3[36:48] = shape2[36:48] #eyes
        shape3[27:36] = shape2[27:36] #nose
        shape3[48:60] = shape2[48:60] #mouth
        morphed_img = manual_aligning_68_v3(img1, shape, shape3) 
    
    else:
        output_0, d_img1, d_img2, mask1, output_replacement_0 = eyebrows(shape, shape2, img1, img2, d_img1, d_img2)

        output_1, d_img1, d_img2, mask2, output_replacement = eyes(shape, shape2, output_0, img2, d_img1, d_img2)
        t_output_1, t_d_img1, t_d_img2, t_mask2, output_replacement_1 = eyes(shape, shape2, output_replacement_0, img2, d_img1, d_img2)

        output_2, d_img1, d_img2, mask3, output_replacement = nose(shape, shape2, output_1, img2, d_img1, d_img2)
        t_output_2, t_d_img1, t_d_img2, t_mask3, output_replacement_2 = nose(shape, shape2, output_replacement_1, img2, d_img1, d_img2)

        output, d_img1, d_img2, mask4, output_replacement = mouth(shape, shape2, output_2, img2, d_img1, d_img2)
        t_output, t_d_img1, t_d_img2, t_mask4, output_replacement = mouth(shape, shape2, output_replacement_2, img2, d_img1, d_img2)
        mask = mask1 + mask2 + mask3 + mask4

        shape3 = shape.copy()
        shape3[17:27] = shape2[17:27] #eyebrows
        shape3[36:48] = shape2[36:48] #eyes
        shape3[27:36] = shape2[27:36] #nose
        shape3[48:60] = shape2[48:60] #mouth
        morphed_img = manual_aligning_68_v3(img1, shape, shape3) 
    
    return output, d_img1, d_img2, mask, output_replacement, morphed_img

