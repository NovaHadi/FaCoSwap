# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 03:28:14 2023

@author: N.H.Lestriandoko

"""
import numpy as np
import cv2
import streamlit as st
import face_morphing as mp

from imutils import face_utils
from face_alignment import manual_aligning_68_v3

@st.cache_data()
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

@st.cache_data()
def check_multiple_faces(face_rects):
    nFaces = len(face_rects)            
    if nFaces > 1:
        x1, y1, w1, h1 = rect_to_bb(face_rects[0])
        x2, y2, w2, h2 = rect_to_bb(face_rects[1])
        if w1>w2:
            face_rect = face_rects[0]
        else:
            face_rect = face_rects[1]
    else:
        face_rect = face_rects[0]    
    return face_rect

@st.cache_data()
def getMaskCenter(img1, mask):
    src_mask = np.zeros(img1.shape, img1.dtype)
    src_mask[mask>0] = 255
    poly = np.argwhere(mask[:,:,2]>0)
    r = cv2.boundingRect(np.float32([poly]))    
    center = (r[1]+int(r[3]/2)), (r[0]+int(r[2]/2))
    return center, src_mask

@st.cache_data()
def wholeface_swap_1(img1, img2, _detector, _predictor):
    
    face1 = _detector(img1, 2)
    landmarks1 = _predictor(img1, face1[0])
    shape1 = face_utils.shape_to_np(landmarks1)

    face2 = _detector(img2, 2)
    landmarks2 = _predictor(img2, face2[0])
    shape2 = face_utils.shape_to_np(landmarks2)
    
    w = 8
    #points1 = shape[0:17] # face border
    points1 = shape1[0:68] # whole face
    points1 = np.append(points1, [[shape1[17][0],shape1[17][1]-w],
                     [shape1[18][0],shape1[18][1]-w],
                     [shape1[19][0],shape1[19][1]-w],
                     [shape1[24][0],shape1[24][1]-w],
                     [shape1[25][0],shape1[25][1]-w],
                     [shape1[26][0],shape1[26][1]-w]], axis=0)
                     #[0,0],[0,149],[149,0],[149,149]], axis=0)

    points2 = shape2[0:68] # whole face
    points2 = np.append(points2, [[shape2[17][0],shape2[17][1]-w],
                     [shape2[18][0],shape2[18][1]-w],
                     [shape2[19][0],shape2[19][1]-w],
                     [shape2[24][0],shape2[24][1]-w],
                     [shape2[25][0],shape2[25][1]-w],
                     [shape2[26][0],shape2[26][1]-w]], axis=0)
                     #[0,0],[0,149],[149,0],[149,149]], axis=0)

    d_img1 = img1.copy()
    d_img2 = img2.copy()
    mask_img = np.zeros(img1.shape, dtype = np.float32)
    
    morphed_landmark = manual_aligning_68_v3(img1, points1, points2)    
    output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img2, d_img1, d_img2, mask_img, points1, points2, alpha=1)
    
    center, src_mask = getMaskCenter(morphed_landmark, mask)
    output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)

    return output

@st.cache_data()
def wholeface_swap_2(img1, img2, _detector, _predictor):
    
    face1 = _detector(img1, 2)
    landmarks1 = _predictor(img1, face1[0])
    shape1 = face_utils.shape_to_np(landmarks1)

    face2 = _detector(img2, 2)
    landmarks2 = _predictor(img2, face2[0])
    shape2 = face_utils.shape_to_np(landmarks2)
    
    w = 8      # for 150 x 150 pixels 
    #w = 8 * 3  # for 500 x 500 pixels
    left_edge1 = 0
    right_edge1 = 0
    left_edge2 = 0
    right_edge2 = 0
    if shape1[16][0]-shape1[45][0]>30:
        left_edge1 = int((shape1[16][0]-shape1[45][0])/2)
        left_edge2 = int((shape2[16][0]-shape2[45][0])/2)
    if shape1[36][0]-shape1[0][0]>30:
        right_edge1 = int((shape1[36][0]-shape1[0][0])/2)
        right_edge2 = int((shape2[36][0]-shape2[0][0])/2)
    #points1 = shape[0:17] # face border
    points1_ff = shape1[0:68]
    points1_ff = np.append(points1_ff, [[shape1[17][0],shape1[17][1]-w],
                     [shape1[18][0],shape1[18][1]-w],
                     [shape1[19][0],shape1[19][1]-w],
                     [shape1[24][0],shape1[24][1]-w],
                     [shape1[25][0],shape1[25][1]-w],
                     [shape1[26][0],shape1[26][1]-w]], axis=0)
    
    points1 = shape1[17:60] # half face with mouth
    points1 = np.append(points1, [[shape1[17][0],shape1[17][1]-w],
                     [shape1[36][0],shape1[2][1]],
                     [shape1[45][0],shape1[14][1]],
                     [shape1[41][0],shape1[48][1]],
                     [shape1[46][0],shape1[54][1]],
                     [shape1[48][0]-w,shape1[48][1]],  # beside mouth corner - point 49
                     [shape1[57][0],shape1[57][1]+ w],  # below mouth - point 58
                     [shape1[54][0]+w,shape1[54][1]],  # beside mouth corner - point 55
                     [shape1[18][0],shape1[18][1]-w],
                     [shape1[19][0],shape1[19][1]-w],
                     [shape1[24][0],shape1[24][1]-w],
                     [shape1[25][0],shape1[25][1]-w],
                     [shape1[26][0],shape1[26][1]-w],
                     [shape1[0][0]+right_edge1,shape1[0][1]],
                     [shape1[1][0]+right_edge1,shape1[1][1]],
                     [shape1[15][0]-left_edge1,shape1[15][1]],
                     [shape1[16][0]-left_edge1,shape1[16][1]]], axis=0)

    points2_ff = shape2[0:68]
    points2_ff = np.append(points2_ff, [[shape2[17][0],shape2[17][1]-w],
                     [shape2[18][0],shape2[18][1]-w],
                     [shape2[19][0],shape2[19][1]-w],
                     [shape2[24][0],shape2[24][1]-w],
                     [shape2[25][0],shape2[25][1]-w],
                     [shape2[26][0],shape2[26][1]-w]], axis=0)

    points2 = shape2[17:60] # half face with mouth
    points2 = np.append(points2, [[shape2[17][0],shape2[17][1]-w],
                     [shape2[36][0],shape2[2][1]],
                     [shape2[45][0],shape2[14][1]],
                     [shape2[41][0],shape2[48][1]],
                     [shape2[46][0],shape2[54][1]],
                     [shape2[48][0]-w,shape2[48][1]],  # beside mouth corner - point 49
                     [shape2[57][0],shape2[57][1]+ w],  # below mouth - point 58
                     [shape2[54][0]+w,shape2[54][1]],  # beside mouth corner - point 55
                     [shape2[18][0],shape2[18][1]-w],
                     [shape2[19][0],shape2[19][1]-w],
                     [shape2[24][0],shape2[24][1]-w],
                     [shape2[25][0],shape2[25][1]-w],
                     [shape2[26][0],shape2[26][1]-w],
                     [shape2[0][0]+right_edge2,shape2[0][1]],
                     [shape2[1][0]+right_edge2,shape2[1][1]],
                     [shape2[15][0]-left_edge2,shape2[15][1]],
                     [shape2[16][0]-left_edge2,shape2[16][1]]], axis=0)

    d_img1 = img1.copy()
    d_img2 = img2.copy()
    mask_img = np.zeros(img1.shape, dtype = np.float32)
    
    morphed_landmark = manual_aligning_68_v3(img1, points1, points2)    
    output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img2, d_img1, d_img2, mask_img, points1, points2, alpha=1)
    
    center, src_mask = getMaskCenter(morphed_landmark, mask)
    output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)

    return output

@st.cache_data()
def wholeface_swap_3(img1, img2, _detector, _predictor):
    
    face1 = _detector(img1, 2)
    landmarks1 = _predictor(img1, face1[0])
    shape1 = face_utils.shape_to_np(landmarks1)

    face2 = _detector(img2, 2)
    landmarks2 = _predictor(img2, face2[0])
    shape2 = face_utils.shape_to_np(landmarks2)
    
    w = 8
    #points1 = shape[0:17] # face border
    points1_ff = shape1[0:68]
    points1_ff = np.append(points1_ff, [[shape1[17][0],shape1[17][1]-w],
                     [shape1[18][0],shape1[18][1]-w],
                     [shape1[19][0],shape1[19][1]-w],
                     [shape1[24][0],shape1[24][1]-w],
                     [shape1[25][0],shape1[25][1]-w],
                     [shape1[26][0],shape1[26][1]-w]], axis=0)
    
    points1 = shape1[15:48] # half face
    add_points1 = shape1[0:2]
    points1 = np.append(points1,add_points1,axis=0)
    points1 = np.append(points1, [[shape1[17][0],shape1[17][1]-w], # above right eyebrows - point 18
                     [shape1[36][0],shape1[2][1]],     # below right eye corner - point 37x,3y
                     [shape1[45][0],shape1[14][1]],    # below left eye corner - point 46x,15y
                     [shape1[18][0],shape1[18][1]-w],  # above right eyebrows - point 19
                     [shape1[19][0],shape1[19][1]-w],  # above right eyebrows - point 20
                     [shape1[24][0],shape1[24][1]-w],  # above left eyebrows - point 25
                     [shape1[25][0],shape1[25][1]-w],  # above left eyebrows - point 26
                     [shape1[26][0],shape1[26][1]-w]], axis=0) # above left eyebrows - point 27
                     #[0,0],[0,149],[149,0],[149,149]], axis=0)

    points2_ff = shape2[0:68]
    points2_ff = np.append(points2_ff, [[shape2[17][0],shape2[17][1]-w],
                     [shape2[18][0],shape2[18][1]-w],
                     [shape2[19][0],shape2[19][1]-w],
                     [shape2[24][0],shape2[24][1]-w],
                     [shape2[25][0],shape2[25][1]-w],
                     [shape2[26][0],shape2[26][1]-w]], axis=0)

    points2 = shape2[15:48] # half face
    add_points2 = shape2[0:2]
    points2 = np.append(points2,add_points2,axis=0)
    points2 = np.append(points2, [[shape2[17][0],shape2[17][1]-w],
                     [shape2[36][0],shape2[2][1]],     # below right eye corner
                     [shape2[45][0],shape2[14][1]],    # below left eye corner
                     [shape2[18][0],shape2[18][1]-w],
                     [shape2[19][0],shape2[19][1]-w],
                     [shape2[24][0],shape2[24][1]-w],
                     [shape2[25][0],shape2[25][1]-w],
                     [shape2[26][0],shape2[26][1]-w]], axis=0)
                     #[0,0],[0,149],[149,0],[149,149]], axis=0)

    d_img1 = img1.copy()
    d_img2 = img2.copy()
    mask_img = np.zeros(img1.shape, dtype = np.float32)
    
    morphed_landmark = manual_aligning_68_v3(img1, points1, points2)    
    output_replacement, mask,  d_img1, d_img2 = mp.morphing_original(morphed_landmark, img2, d_img1, d_img2, mask_img, points1, points2, alpha=1)
    
    center, src_mask = getMaskCenter(morphed_landmark, mask)
    output = cv2.seamlessClone(output_replacement, morphed_landmark, src_mask, center, cv2.NORMAL_CLONE)

    return output
