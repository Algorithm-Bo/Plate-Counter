import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import os
import random
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2 as cv

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

# folders with original pictures and masks
image_folder = 'data/images_train_40'
mask_folder = 'data/binary_masks_train_40'

# output folder for augmented images and masks
output_foldername = 'data/augmented_images_train_easy'

filename_list = os.listdir(image_folder)
print('number of files: ', len(filename_list))

# filenames for augmented data starts with save_as_counter.PNG
save_as_counter = len(filename_list)

image_list = np.zeros((len(filename_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
mask_list = np.zeros((len(filename_list), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

# train images
print('Load images and masks')

for n in range(0, len(filename_list)):
    image_path = image_folder + '/' + str(n) + '.png'
    img = imread(image_path)[:,:,:IMG_CHANNELS]  
    image_list[n] = img
    mask_path = mask_folder + '/' + str(n) + '.png'
    print(mask_path)
    mask = imread(mask_path)[:,:,:1]
    mask_list[n] = mask
    
images = image_list
segmaps = np.array(mask_list * 255, dtype=np.uint16)

ia.seed(42)

aug1 = iaa.SomeOf(3, [ # choose 3 randomly
    # 5 geometric
    iaa.Fliplr(1.0), # flip horizontally
    iaa.Flipud(1.0), # flip vertically
    iaa.Affine(rotate=(-180, +180)), # rotate
    iaa.ScaleX((0.8, 1.0)), # scale X
    iaa.ScaleY((0.8, 1.0)), # scale Y
    # 2 color
    iaa.AddToHueAndSaturation((-50, 50), per_channel=True), # add to hue/sat
    iaa.AddToBrightness((-30, 30)), # add brightness
    ],
random_order=True)

# Augment images and segmaps.
images_aug = []
segmaps_aug = []
num_augmentations = 20
for _ in range(num_augmentations):
    for n in range(0, len(images)):
        image = images[n]
        segmap = segmaps[n]
        segmap = np.expand_dims(segmap, axis=0)
        images_aug_i, segmaps_aug_i = aug1(image=image, segmentation_maps=segmap)
        images_aug.append(images_aug_i)
        segmaps_aug.append(segmaps_aug_i)

# save original images
counter = 0
for elem in image_list:
    img_original = np.squeeze(image_list[counter])
    mask_original = np.squeeze(mask_list[counter])
   
    # create output folders
    if not os.path.isdir(output_foldername):
        os.mkdir(output_foldername)
    if not os.path.isdir(output_foldername + '/img'):
        os.mkdir(output_foldername + '/img')
    if not os.path.isdir(output_foldername + '/masks'):
        os.mkdir(output_foldername + '/masks')
    
    img_path = output_foldername + '/img/' + str(counter) + '.PNG'
    mask_path = output_foldername + '/masks/' + str(counter) + '.PNG'
    
    imsave(img_path, img_original)
    imsave(mask_path, mask_original * 255)

    counter += 1
    
# save augmented images and masks to folders
counter2 = 0
for elem2 in images_aug:
    img_augmented = np.squeeze(images_aug[counter2])
    
    mask_augmented = np.squeeze(segmaps_aug[counter2])
    # print(img_augmented.shape)
 
    img_aug_path = output_foldername + '/img/' + str(save_as_counter) + '.PNG'
    mask_aug_path = output_foldername + '/masks/' + str(save_as_counter) + '.PNG'
    
    imsave(img_aug_path, img_augmented)
    imsave(mask_aug_path, mask_augmented * 255)

    counter2 += 1
    save_as_counter += 1
   