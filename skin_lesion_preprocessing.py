### Skin Lesion Dataset Preprocessing
# cd ~/tf
# download & unzip skin-cancer-mnist-ham10000.zip
# cd skin-cancer-mnist-ham10000
# unzip HAM10000_images_part_1.zip
# unzip HAM10000_images_part_2.zip
# python ../skin_lesion_preprocessing.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import shutil

# Create a new directory for the images
base_dir = '../data/ham10000'
os.mkdir(base_dir)

# Training file directory
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)

# Validation file directory
val_dir = os.path.join(base_dir, 'val')
os.mkdir(val_dir)

# Create new folders in the training directory for each of the classes
nv = os.path.join(train_dir, 'nv')
os.mkdir(nv)
mel = os.path.join(train_dir, 'mel')
os.mkdir(mel)
bkl = os.path.join(train_dir, 'bkl')
os.mkdir(bkl)
bcc = os.path.join(train_dir, 'bcc')
os.mkdir(bcc)
akiec = os.path.join(train_dir, 'akiec')
os.mkdir(akiec)
vasc = os.path.join(train_dir, 'vasc')
os.mkdir(vasc)
df = os.path.join(train_dir, 'df')
os.mkdir(df)

# Create new folders in the validation directory for each of the classes
nv = os.path.join(val_dir, 'nv')
os.mkdir(nv)
mel = os.path.join(val_dir, 'mel')
os.mkdir(mel)
bkl = os.path.join(val_dir, 'bkl')
os.mkdir(bkl)
bcc = os.path.join(val_dir, 'bcc')
os.mkdir(bcc)
akiec = os.path.join(val_dir, 'akiec')
os.mkdir(akiec)
vasc = os.path.join(val_dir, 'vasc')
os.mkdir(vasc)
df = os.path.join(val_dir, 'df')
os.mkdir(df)

# Read the metadata
df = pd.read_csv('HAM10000_metadata.csv')

# Display some information in the dataset
df.head()

# Set y as the labels
y = df['dx']

# Split the metadata into training and validation
df_train, df_val = train_test_split(df, test_size=0.1, random_state=101, stratify=y)

# Print the shape of the training and validation split
print(df_train.shape)
print(df_val.shape)

# Find the number of values in the training and validation set
df_train['dx'].value_counts()
df_val['dx'].value_counts()

# Transfer the images into folders
# Set the image id as the index
df.set_index('image_id', inplace=True)

# Get a list of images in each of the two folders
folder_1 = os.listdir('ham10000_images_part_1')
folder_2 = os.listdir('ham10000_images_part_2')

# Get a list of train and val images
train_list = list(df_train['image_id'])
val_list = list(df_val['image_id'])

# Transfer the training images
for image in train_list:

    fname = image + '.jpg'
    label = df.loc[image, 'dx']

    if fname in folder_1:
        # source path to image
        src = os.path.join('ham10000_images_part_1', fname)
        # destination path to image
        dst = os.path.join(train_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    if fname in folder_2:
        # source path to image
        src = os.path.join('ham10000_images_part_2', fname)
        # destination path to image
        dst = os.path.join(train_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

# Transfer the validation images
for image in val_list:

    fname = image + '.jpg'
    label = df.loc[image, 'dx']

    if fname in folder_1:
        # source path to image
        src = os.path.join('ham10000_images_part_1', fname)
        # destination path to image
        dst = os.path.join(val_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    if fname in folder_2:
        # source path to image
        src = os.path.join('ham10000_images_part_2', fname)
        # destination path to image
        dst = os.path.join(val_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

# Check how many training images are in each folder
print(len(os.listdir(train_dir+'/nv')))
print(len(os.listdir(train_dir+'/mel')))
print(len(os.listdir(train_dir+'/bkl')))
print(len(os.listdir(train_dir+'/bcc')))
print(len(os.listdir(train_dir+'/akiec')))
print(len(os.listdir(train_dir+'/vasc')))
print(len(os.listdir(train_dir+'/df')))

# Check how many validation images are in each folder
print(len(os.listdir(val_dir+'/nv')))
print(len(os.listdir(val_dir+'/mel')))
print(len(os.listdir(val_dir+'/bkl')))
print(len(os.listdir(val_dir+'/bcc')))
print(len(os.listdir(val_dir+'/akiec')))
print(len(os.listdir(val_dir+'/vasc')))
print(len(os.listdir(val_dir+'/df')))