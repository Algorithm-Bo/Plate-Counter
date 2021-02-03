import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import random
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2 as cv

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

# configure folders
# TEST_PATH = 'test_data_staphylococcus_512x512'
TEST_PATH = 'test_data_examples_difficult'

model_name ='model_32_filters_bn_batch_size_8_test_1'
output_foldername = 'results_prediction/' + model_name

# Free up RAM
tf.keras.backend.clear_session()

# load pretrained model
model = load_model('models/' + model_name + '.h5')
model.summary()

# load test images
test_ids = next(os.walk(TEST_PATH))[1]

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

# make predictions
preds_test = model.predict(X_test, verbose=1)

# make binary
threshold = 0.5
preds_test_t = (preds_test > threshold).astype(np.uint8)

# save test images
counter = 0

# filenames starts with save_as_counter.PNG
save_as_counter = 0

for result in preds_test:
    img_squeezed = np.squeeze(preds_test[counter])
    img_squeezed_t = np.squeeze(preds_test_t[counter])
        
    # create test folder, if not existing
    if not os.path.isdir(output_foldername):
        os.mkdir(output_foldername)
    if not os.path.isdir(output_foldername + '/results'):
        os.mkdir(output_foldername + '/results')
    if not os.path.isdir(output_foldername + '/results_t'):
        os.mkdir(output_foldername + '/results_t')
    
    result_path = output_foldername + '/results/' + str(save_as_counter) + '.PNG'
    result_path_t = output_foldername + '/results_t/' + str(save_as_counter) + '.PNG'
   
    cv.imwrite(result_path, img_squeezed * 255)
    cv.imwrite(result_path_t, img_squeezed_t * 255)
   
    counter += 1
    save_as_counter += 1

