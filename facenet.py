### Face Detection   : OpenCV CascadeClassifier 
### Face Recognition : Facenet
###(this will train all faces (96x96), then open Webcam to recognize face)
# Usage: python3 facenet.py 
import time
from multiprocessing.dummy import Pool
import cv2
import os
import glob
import pandas as pd
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
import tensorflow.keras as keras
from facenet_utils import *
from inception_blocks_v2 import *

## for GPU
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)
tf.keras.backend.set_image_data_format('channels_first')

PADDING = 50
ready_to_detect_identity = True

FRmodel = faceRecoModel(input_shape=(3, 96, 96))

def triplet_loss(y_true, y_pred, alpha = 0.3):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_pred -- python list containing three objects:
    anchor -- the encodings for the anchor images, of shape (None, 128)
    positive -- the encodings for the positive images, of shape (None, 128)
    negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Step 1: 計算anchor和positive的編碼(距離)
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: 計算anchor和negative的編碼(距離)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: 將先前計算出的距離相減並加上邊距alpha
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: 將上述計算出的損失與零取最大值，再將所有樣本加總起來
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

def prepare_database():
    database = {}
    # load all the images of individuals to recognize into the database
    for file in glob.glob("faces/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        print(identity)
        database[identity] = img_path_to_encoding(file, FRmodel)

    return database

def webcam_face_recognizer(database):
    """
    Runs a loop that extracts images from the computer's webcam and determines whether or not
    it contains the face of a person in our database.

    If it contains a face, an audio message will be played welcoming the user.
    If not, the program will process the next frame from the webcam
    """
    global ready_to_detect_identity

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # OpenCV CascadeClassifier
    
    while vc.isOpened():
        _, frame = vc.read()
        img = frame

        # We do not want to detect a new identity while the program is in the process of identifying another person
        if ready_to_detect_identity:
            img = process_frame(img, frame, face_cascade)   
        
        cv2.imshow("preview", img)
        key = cv2.waitKey(1)
		
        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")

def process_frame(img, frame, face_cascade):
    """
    Determine whether the current frame contains the faces of people from our database
    """
    global ready_to_detect_identity
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through all the faces detected and determine whether or not they are in the database
    identities = []
    for (x, y, w, h) in faces:
        x1 = x-PADDING
        y1 = y-PADDING
        x2 = x+w+PADDING
        y2 = y+h+PADDING
		
        img = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,0,0),2) # frame turn to blue		
        identity = find_identity(frame, x1, y1, x2, y2)
 
        if identity is not None:
            identities.append(identity)
            cv2.putText(img, identity, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        else:
            cv2.putText(img, 'Unknown', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
			
    if identities != []:                                         	       		
        ready_to_detect_identity = False              
        pool = Pool(processes=1) # run this as a separate process so that the camera feedback does not freeze
        pool.apply_async(welcome_users, [identities]) # welcome users	
		
    cv2.imwrite('sign-in.png',img)
    return img

def find_identity(frame, x1, y1, x2, y2):
    """
    Determine whether the face contained within the bounding box exists in our database

    x1,y1_____________
    |                 |
    |                 |
    |_________________x2,y2

    """
    height, width, channels = frame.shape
    # The padding is necessary since the OpenCV face detector creates the bounding box around the face and not the head
    part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    
    return who_is_it(part_image, database, FRmodel)

def who_is_it(image, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    encoding = img_to_encoding(image, model)
    
    min_dist = 100
    identity = None
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding)
		
        #print('distance for %s is %s' %(name, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 0.70:
        return None
    else:
        return str(identity)

def welcome_users(identities):
    """ Outputs a welcome audio message to the users """
    global ready_to_detect_identity
    welcome_message = 'Welcome '
    df = pd.read_csv('faces_namelist.csv', encoding='utf-8')
    namelist = df.set_index('id').T.to_dict()

    if len(identities) == 1:
        welcome_message += '%s, have a nice day.' % namelist[identities[0]]['name']
        #print('Welcome %s, have a nice day!' %(identities[0]))
    else:
        for identity_id in range(len(identities)-1):
            welcome_message += '%s, ' % namelist[identities[identity_id]]['name']
        welcome_message += 'and %s, ' % namelist[identities[-1]]['name']
        welcome_message += 'have a nice day!'
    print(welcome_message)

    # Allow the program to start detecting identities again
    ready_to_detect_identity = True

if __name__ == "__main__":
    database = prepare_database()
    webcam_face_recognizer(database)

# ### References:
# 
# - Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
# - Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf) 
# - The pretrained model we use is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
# - Our implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet 
