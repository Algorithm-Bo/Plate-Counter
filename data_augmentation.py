import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import os
import random
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2 as cv

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

TRAIN_PATH = 'train_data_256x256_without_augmentation_special_new'

output_foldername = 'augmented_images_imgaug_5'

train_ids = next(os.walk(TRAIN_PATH))[1]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

# train images
print('Load training images and masks')

for n in range(0, len(train_ids)):
    path = TRAIN_PATH + '/' + str(n)

    img = imread(path + '/image/' + str(n) + '.png')[:,:,:IMG_CHANNELS]  
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = imread(path + '/mask/' + str(n) + '.png')
    mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1) 
    Y_train[n] = mask
    n += 1

images = X_train
segmaps = np.array(Y_train * 255, dtype=np.uint16)

ia.seed(42)

aug1 = iaa.SomeOf(1, [ # 1 zufällige auswählen
    iaa.GaussianBlur((0, 3.0)), # Blur (unscharf)
    iaa.Fliplr(0.5), # flip horizontally
    iaa.Flipud(0.5), # flip vertically
    iaa.Affine(rotate=(-45, 45)), # drehen
    iaa.Affine(translate_px={"x": (-50, 50), "y": (-50, 50)}), # verschieben
    iaa.Affine(scale=(0.5, 1.5)), # skalieren größer/kleiner
    iaa.Affine(shear=(-16, 16)), # Scherung
    iaa.ScaleX((0.5, 1.5)), # breiter/schmaler
    iaa.ScaleY((0.5, 1.5)),
    iaa.Add((-30, 30), per_channel=0.5), # Farbkanäle +/-
    iaa.AdditiveGaussianNoise(scale=0.2*255), # noise
    iaa.Multiply((0, 1.5), per_channel=0.5), # Bild heller/dunkler
    iaa.MultiplyElementwise((0, 1.5), per_channel=0.5), # Bild heller/dunkler pixelweise
    iaa.GammaContrast((0, 2), per_channel=True), # gamma contrast
    iaa.AddToHueAndSaturation((-100, 100), per_channel=True), # +/- hue/sat
    iaa.HistogramEqualization(),
    iaa.Crop(px=(0, 10))
])

# image = images[0]
# segmap = segmaps[0]

# Augment images and segmaps.
images_aug = []
segmaps_aug = []
num_augmentations = 16

for n in range(0, len(images)):
    image = images[n]
    segmap = segmaps[n]
    print('segmap.shape: ', segmap.shape)
    segmap = np.expand_dims(segmap, axis=0)
    print('segmap.shape: ', segmap.shape)
    for _ in range(num_augmentations):
        images_aug_i, segmaps_aug_i = aug1(image=image, segmentation_maps=segmap)
        images_aug.append(images_aug_i)
        segmaps_aug.append(segmaps_aug_i)

# print('images_aug.shape: ', images_aug.shape)
# print('segmaps_aug.shape: ', segmaps_aug.shape)

# images_aug, segmaps_aug = aug1(images=images, segmentation_maps=segmaps)
# images_aug2 = np.array([aug1.augment_image(images[1]) for _ in range(16)])

# samples addieren
"""
for i in range (1, 5):

    images_aug_new, segmaps_aug_new = aug1(images=images, segmentation_maps=segmaps)
    print('Hallo')
    images_aug = images_aug + images_aug_new
    segmaps_aug = segmaps_aug + segmaps_aug_new
    # aug1.show_grid([images_aug[0], images_aug[1]], cols=8, rows=8)
    print(images_aug.shape)
"""

# img_grid = aug1.show_grid([images_aug[0], images_aug[1]], cols=8, rows=8)
# aug1.show_grid([segmaps_aug[0], segmaps_aug[1]], cols=8, rows=8)

# save augmented images and masks to folders
counter = 0
save_as_counter = 100 # augmented data ab save_as_counter.PNG

for elem in images_aug:
    img_augmented = np.squeeze(images_aug[counter])
    
    mask_augmented = np.squeeze(segmaps_aug[counter])
    print(img_augmented.shape)

    # Ausgabeordner erstellen, falls nicht vorhanden
    if not os.path.isdir(output_foldername):
        os.mkdir(output_foldername)
    if not os.path.isdir(output_foldername + '/img'):
        os.mkdir(output_foldername + '/img')
    if not os.path.isdir(output_foldername + '/masks'):
        os.mkdir(output_foldername + '/masks')
    
    img_path = output_foldername + '/img/' + str(save_as_counter) + '.PNG'
    mask_path = output_foldername + '/masks/' + str(save_as_counter) + '.PNG'
    
    imsave(img_path, img_augmented)
    imsave(mask_path, mask_augmented * 255)

    counter += 1
    save_as_counter += 1

