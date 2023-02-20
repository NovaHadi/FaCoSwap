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
#import face_morphing as mp

from PIL import Image
from face_alignment import manual_aligning_68_v3
from imutils import face_utils
#from io import BytesIO

@st.cache_data()
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
def crop_to_bbox(img, bbox):
    shape_img = img.shape[:2]
    bbox = np.clip(bbox, 0., 1.)
    bbox = np.reshape(bbox, [-1, 2]) * shape_img
    bbox = bbox.flatten().astype(np.int32)
    img_cropped = img[bbox[0]:bbox[2], bbox[1]:bbox[3], :] # crop
    return img_cropped

@st.cache_data()
def detect_face_0(img, _detector, _predictor, padding, size):
    status = True    
    #img = dlib.load_rgb_image(input_dir)
    img_crop = img

    face_rects = detector(img, 2)
    if len(face_rects)<1:
        img = rotate_bound(img, angle=90)
        face_rects = detector(img, 2)
        if len(face_rects)<1:
            img = rotate_bound(img, angle=90)
            face_rects = detector(img, 2)
            if len(face_rects)<1:
                img = rotate_bound(img, angle=90)
                face_rects = detector(img, 2)


    if len(face_rects)<1:   
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        face_rects = detector(gray, 0)
        if len(face_rects)<1:   
            status = False
            #print('[DETECTION FAILED] ')
            return img, status 

    if len(face_rects)>0:
        status =True
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
        return img_crop, status

@st.cache_data
def NDS_morphing(img1, img2, _predictor):
    face1 = detector(img1, 2)
    landmarks1 = predictor(img1, face1[0])
    shape1 = face_utils.shape_to_np(landmarks1)
    
    face2 = detector(img2, 2)
    landmarks2 = predictor(img2, face2[0])
    shape2 = face_utils.shape_to_np(landmarks2)
    
    output = manual_aligning_68_v3(img1, shape1, shape2)    
    return output

@st.cache_data
def check_landmarks(img, _predictor):
    face = detector(img, 2)
    landmarks1 = predictor(img, face[0])
    shape = face_utils.shape_to_np(landmarks1)
    for (i, (x, y)) in enumerate(shape):
        cv2.circle(img, (x, y), 2, (0, 255, 100), -1) 
    
    return shape.size, img
    
@st.cache_resource
def load_model(dlib_path):
    predictor68_path = dlib_path +'shape_predictor_68_face_landmarks.dat'
    #face_rec_model_path = dlib_path +'dlib_face_recognition_resnet_model_v1.dat'
    predictor = dlib.shape_predictor(predictor68_path)    
    return predictor    


if __name__ == '__main__':

    dlib_path = '/app/facoswap/dlib-models-master/'
    predictor = load_model(dlib_path)
    detector = dlib.get_frontal_face_detector()
        
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
        
    st.title(':male-scientist: Welcome To FaCoSwap v1.2!')
    instructions = """ FaCoSwap is a face component swap tool for face analysis. 
        Upload your own images and click the swap button to change the face part! """
    st.write(instructions)

    cam = st.checkbox(":camera: camera")
    if cam:    
        picture = st.camera_input("Take a picture")
    
        if picture:
            #st.image(picture)
            byte_im = picture.getvalue()
            
            btn = st.download_button(
                  label="Download Image",
                  data=byte_im,
                  file_name="download-image.png",
                  mime="image/jpeg",
                  )
    
    c1, c2 = st.columns(2)
    file1 = c1.file_uploader('Upload Face 1')
    if file1:        
        img1 = Image.open(file1)
        crop_img1, status1 = detect_face_0(np.asarray(img1), detector, predictor, padding=0.6, size=int(add_selectbox3))
        c1.image(crop_img1)
        if not status1:
            c1.write("No face detected!")
    else:
        crop_img1 = None
    
    file2 = c2.file_uploader('Upload Face 2')
    if file2:
        img2 = Image.open(file2)
        crop_img2, status2 = detect_face_0(np.asarray(img2), detector, predictor, padding=0.6, size=int(add_selectbox3))
        c2.image(crop_img2)            
        if not status2:
            c2.write("No face detected!")
    else:
        crop_img2 = None
    
    st.title(":curly_haired_person: Face component swapping ")
    st.write('---:eye:-:nose:-:lips:-:ear:-:eye:---')

    col1, col2 = st.columns(2)
    if file1 and file2 and col1.button("Swap face 1"):
        if add_selectbox2=='NDTS' and status1 and status2 :
            output, d_img1, d_img2, mask, output_replacement, morphed_img  = ndts.LmPt_Morph(crop_img1, crop_img2, add_selectbox1, detector, predictor)
            st.image(output)
        elif add_selectbox2=='NDT' and status1 and status2 :
            output, morphed_img, mask_img, delaunay_img1, delaunay_img2, all_points, ori_points = fpr.face_part_replacement(crop_img1, detector, predictor, crop_img2, 1, add_selectbox1)
            st.image(output)
        elif add_selectbox2=='NDS' and status1 and status2 :
            output = NDS_morphing(crop_img1, crop_img2, predictor)
            st.image(output)
        else:
            #st.image(Image.open(file2))
            if status1 and not status2:
                st.write("Faces required! Please upload face 2.")
            elif not status1 and status2:
                st.write("Faces required! Please upload face 1.")
            else:
                st.write("Faces required! Please upload face 1 and face 2.")

    if  file1 and file2 and col2.button("Swap face 2"):
        if add_selectbox2=='NDTS' and status1 and status2 :
            output, d_img2, d_img1, mask, output_replacement, morphed_img  = ndts.LmPt_Morph(crop_img2, crop_img1, add_selectbox1, detector, predictor)
            st.image(output)
        elif add_selectbox2=='NDT' and status1 and status2 :
            output, morphed_img, mask_img, delaunay_img2, delaunay_img1, all_points, ori_points = fpr.face_part_replacement(crop_img2, detector, predictor, crop_img1, 1, add_selectbox1)
            st.image(output)
        elif add_selectbox2=='NDS' and status1 and status2 :
            output = NDS_morphing(crop_img2, crop_img1, predictor)
            st.image(output)
        else:
            #st.image(Image.open(file1))
            if status1 and not status2:
                st.write("Faces required! Please upload face 2.")
            elif not status1 and status2:
                st.write("Faces required! Please upload face 1.")
            else:
                st.write("Faces required! Please upload face 1 and face 2.")
        
    
    
