import numpy as np
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
"""
seq = iaa.Sequential([
    iaa.GaussianBlur((0, 3.0)),
    iaa.Affine(translate_px={"x": (-40, 40)}),
    iaa.Crop(px=(0, 10))
])
"""
aug2 = iaa.SomeOf(1, [ # 1 zufällige auswählen
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

"""
images_aug, segmaps_aug = seq(images=images, segmentation_maps=segmaps)

seq.show_grid([images_aug[0], images_aug[1]], cols=8, rows=8)

seq.show_grid([segmaps_aug[0], segmaps_aug[1]], cols=8, rows=8)
"""
# num_original_samples = 43
# j = num_original_samples + 1

images_aug, segmaps_aug = aug2(images=images, segmentation_maps=segmaps)

# noch ändern:
"""
for i in range (1, 5):

    images_aug_new, segmaps_aug_new = aug2(images=images, segmentation_maps=segmaps)
    print('Hallo')
    images_aug = images_aug + images_aug_new
    segmaps_aug = segmaps_aug + segmaps_aug_new
    # aug2.show_grid([images_aug[0], images_aug[1]], cols=8, rows=8)
    print(images_aug.shape)
"""
aug2.show_grid([images_aug[0], images_aug[1]], cols=8, rows=8)
# aug2.show_grid([segmaps_aug[0], segmaps_aug[1]], cols=8, rows=8)

# save augmented images and masks to folders

counter = 0
save_as_counter = 602 # augmented data ab save_as_counter.PNG
for elem in images_aug:
    img_augmented = np.squeeze(images_aug[counter])
    mask_augmented = np.squeeze(segmaps_aug[counter])
    # print(img_augmented.shape)

    output_foldername = 'augmented_images_imgaug_3_new'

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

    # cv.imwrite(img_path, img_augmented)
    # cv.imwrite(mask_path, mask_augmented_grayscale * 255)

    counter += 1
    save_as_counter += 1

# example
"""

import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa


ia.seed(1)

# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.
images = np.array(
    [ia.quokka(size=(64, 64)) for _ in range(32)],
    dtype=np.uint8
)

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order

images_aug = seq(images=images)
"""