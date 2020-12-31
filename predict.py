import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import random
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2 as cv

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
img_size = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

# initialize

TEST_PATH = 'test_data_pseudomonas_512x512'

output_foldername = 'results_val_split_0_1_batchsize_8_imgaug_3_special_new_pseudomonas_512x512'

test_ids = next(os.walk(TEST_PATH))[1]

# test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Load test images') 
n = 0
for test_img in range(0, len(test_ids)):
    path = TEST_PATH + '/' + str(n)
    img = imread(path + '/' + str(n) + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img
    n += 1

# load pretrained model
model = load_model('my_model.h5')

model.summary()

# predict
preds_test = model.predict(X_test, verbose=1)

# make binary (aufrunden auf 1)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# save test images
counter = 0
save_as_counter = 0 # 90 # Testdata ab Bild 90.PNG
for result in preds_test:
    img_squeezed = np.squeeze(preds_test[counter])
    img_squeezed_t = np.squeeze(preds_test_t[counter])
    img_squeezed_test_original = np.squeeze(X_test[counter])
    
    # Ausgabeordner erstellen, falls nicht vorhanden
    if not os.path.isdir(output_foldername):
        os.mkdir(output_foldername)
    if not os.path.isdir(output_foldername+ '/results'):
        os.mkdir(output_foldername + '/results')
    if not os.path.isdir(output_foldername+ '/results_t'):
        os.mkdir(output_foldername + '/results_t')
    if not os.path.isdir(output_foldername+ '/results_test_original'):
        os.mkdir(output_foldername + '/results_test_original')

    result_path = output_foldername + '/results/' + str(save_as_counter) + '.PNG'
    result_path_t = output_foldername + '/results_t/' + str(save_as_counter) + '.PNG'
    result_path_test_original = output_foldername + '/results_test_original/' + str(save_as_counter) + '.PNG'

    cv.imwrite(result_path, img_squeezed * 255)
    cv.imwrite(result_path_t, img_squeezed_t * 255)
    cv.imwrite(result_path_test_original, img_squeezed_test_original)
    
    counter += 1
    save_as_counter += 1

