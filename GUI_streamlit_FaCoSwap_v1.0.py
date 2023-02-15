# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 02:22:52 2023

@author: N.H.Lestriandoko
National Research and Innovative Agency (BRIN) - Indonesia
University of Twente - The Netherlands

===============================================
=== FaCoSwap implementation using Streamlit ===
===============================================

"""
import streamlit as st
import numpy as np
import cv2
import dlib
import face_replacement  as fpr
import face_replacement_NDTS  as ndts
import face_morphing as mp

#import general as gn

from PIL import Image

@st.cache()
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

@st.cache()
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

@st.cache()
def crop_to_bbox(img, bbox):
    shape_img = img.shape[:2]
    bbox = np.clip(bbox, 0., 1.)
    bbox = np.reshape(bbox, [-1, 2]) * shape_img
    bbox = bbox.flatten().astype(np.int32)
    img_cropped = img[bbox[0]:bbox[2], bbox[1]:bbox[3], :] # crop
    return img_cropped

@st.cache()
def detect_face_0(img, detector, predictor, padding, size):
        
    #img = dlib.load_rgb_image(input_dir)
    img_crop = img

    if img.shape[0]<img.shape[1]:
        img = rotate_bound(img, angle=90)
        face_rects = detector(img, 2)
        if len(face_rects)<1:
            img = rotate_bound(img, angle=270)
            face_rects = detector(img, 2)
    else :
        face_rects = detector(img, 2)
        if len(face_rects)<1:
            img = rotate_bound(img, angle=180)
            face_rects = detector(img, 2)

    if len(face_rects)<1:   
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        face_rects = detector(gray, 0)
        if len(face_rects)<1:   
            print('[DETECTION FAILED] ')
            return img

    if len(face_rects)>0:
                            
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
            
        # extract the landmarks
        landmarks = predictor(img, face_rect)
        # align and crop the face         
        img_crop = dlib.get_face_chip(img, landmarks, size=size, padding=padding)
        return img_crop

    
dlib_path = '/app/facoswap/dlib-models-master/'
predictor68_path = dlib_path +'shape_predictor_68_face_landmarks.dat'
face_rec_model_path = dlib_path +'dlib_face_recognition_resnet_model_v1.dat'
predictor = dlib.shape_predictor(predictor68_path)
detector = dlib.get_frontal_face_detector()

if __name__ == '__main__':

    add_selectbox1 = st.sidebar.selectbox(
        'Which part(s) would you like to swap?',
        ('eyes', 'eyebrows', 'nose', 'mouth', 'eyebrows-eyes-nose-mouth', 
         'eyebrows-eyes-nose', 'eyebrows-eyes-mouth', 'eyebrows-nose-mouth', 'eyes-nose-mouth', 
         'eyebrows-eyes', 'eyebrows-nose', 'eyebrows-mouth', 'eyes-nose', 'eyes-mouth', 'nose-mouth' )
    )
        
    add_selectbox2 = st.sidebar.selectbox(
        'Which method would you like to use? NDTS-swap texture and shape, NDT-swap texture only, NDS-swap shape only.',
        ('NDTS', 'NDT', 'NDS')
    )
    
    add_selectbox3 = st.sidebar.selectbox(
        'Image resolution: ("150" means the image is resized into 150x150 pixels)',
        ('150', '320', '500') 
    )
        
    st.title(':male-scientist: Welcome To FaCoSwap v1.0!')
    instructions = """ FaCoSwap is a face component swap tool for face analysis. 
        Upload your own images and click the swap button to change the face part! """
    st.write(instructions)
    
    c1, c2 = st.columns(2)
    file1 = c1.file_uploader('Upload Face 1')
    if file1:        
        img1 = Image.open(file1)
        crop_img1 = detect_face_0(np.asarray(img1), detector, predictor, padding=0.6, size=int(add_selectbox3))
        c1.image(crop_img1)
    
    file2 = c2.file_uploader('Upload Face 2')
    if file2:
        img2 = Image.open(file2)
        crop_img2 = detect_face_0(np.asarray(img2), detector, predictor, padding=0.6, size=int(add_selectbox3))
        c2.image(crop_img2)
    
    st.title(":curly_haired_person: Face component swapping ")
    st.write('---:eye:-:nose:-:lips:-:ear:-:eye:---')

    col1, col2 = st.columns(2)
    if file1 and file2 and col1.button("Swap face 1"):
        st.image(crop_img2)
        
    if file1 and file2 and col2.button("Swap face 2"):
        st.image(crop_img1)
        

