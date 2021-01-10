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
img_size = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

def get_model(img_size):
    # print(img_size)
    inputs = tf.keras.layers.Input((img_size))
    
    # change integers to floatingpoint values 0..1
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    # Encoder
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
 
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Dropout(0.2)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    p5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c5)

    c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p5)
    c6 = tf.keras.layers.BatchNormalization()(c6)
    c6 = tf.keras.layers.Dropout(0.3)(c6)
    c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = tf.keras.layers.BatchNormalization()(c6)

    # Decoder
    u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.BatchNormalization()(u7)
    u7 = tf.keras.layers.concatenate([u7, c5])
    c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.BatchNormalization()(c7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    c7 = tf.keras.layers.BatchNormalization()(c7)

    u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.BatchNormalization()(u8)
    u8 = tf.keras.layers.concatenate([u8, c4])
    c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.BatchNormalization()(c8)
    c8 = tf.keras.layers.Dropout(0.2)(c8)
    c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    c8 = tf.keras.layers.BatchNormalization()(c8)
 
    u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.BatchNormalization()(u9)
    u9 = tf.keras.layers.concatenate([u9, c3])
    c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.BatchNormalization()(c9)
    c9 = tf.keras.layers.Dropout(0.2)(c9)
    c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    c9 = tf.keras.layers.BatchNormalization()(c9)

    u10 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c9)
    u10 = tf.keras.layers.BatchNormalization()(u10)
    u10 = tf.keras.layers.concatenate([u10, c2])
    c10 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u10)
    c10 = tf.keras.layers.BatchNormalization()(c10)
    c10 = tf.keras.layers.Dropout(0.1)(c10)
    c10 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c10)
    c10 = tf.keras.layers.BatchNormalization()(c10)

    u11 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c10)
    u11 = tf.keras.layers.BatchNormalization()(u11)
    u11 = tf.keras.layers.concatenate([u11, c1], axis = 3)
    c11 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u11)
    c11 = tf.keras.layers.BatchNormalization()(c11)
    c11 = tf.keras.layers.Dropout(0.1)(c11)
    c11 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c11)
    c11 = tf.keras.layers.BatchNormalization()(c11)

    # classification layer (predictions = outputs)
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c11)

    # define model 
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model

# initialize
TRAIN_PATH = 'train_data_512x512_final_with_augmentation'
TEST_PATH = 'test_data_staphylococcus_512x512'

output_foldername = 'results_val_split_0_25_batchsize_8_imgaug_staphylococcus_512x512_final_deeper_modified'

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

# create empty arrays
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

# train images
print('Load train images and masks')

for n in range(0, len(train_ids)):
    path = TRAIN_PATH + '/' + str(n)
    img = imread(path + '/image/' + str(n) + '.png')[:,:,:IMG_CHANNELS]  
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = imread(path + '/mask/' + str(n) + '.png')
    mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1) 
    Y_train[n] = mask
    n += 1

# Free up RAM
tf.keras.backend.clear_session()

# build model
model = get_model(img_size)
model.summary()

# load pretrained model
# model = load_model('my_model.h5')

# train model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

checkpointer = tf.keras.callbacks.ModelCheckpoint('model_plate_counter.h5', verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='my_log_dir', histogram_freq=1, embeddings_freq=1,)]
 
model.fit(X_train, Y_train, validation_split=0.25, batch_size=8, epochs=50, shuffle=True, callbacks=callbacks)

model.save('my_model_512_deeper_modified.h5')

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

preds_test = model.predict(X_test, verbose=1)

# make binary
threshold = 0.5
preds_test_t = (preds_test > threshold).astype(np.uint8)

# save test images
counter = 0
save_as_counter = 0 # Testdata ab Bild 90.PNG
for result in preds_test:
    img_squeezed = np.squeeze(preds_test[counter])
    img_squeezed_t = np.squeeze(preds_test_t[counter])
    
    # create folders
    if not os.path.isdir(output_foldername):
        os.mkdir(output_foldername)
    if not os.path.isdir(output_foldername+ '/results'):
        os.mkdir(output_foldername + '/results')
    if not os.path.isdir(output_foldername+ '/results_t'):
        os.mkdir(output_foldername + '/results_t')
    
    result_path = output_foldername + '/results/' + str(save_as_counter) + '.PNG'
    result_path_t = output_foldername + '/results_t/' + str(save_as_counter) + '.PNG'
    
    cv.imwrite(result_path, img_squeezed * 255)
    cv.imwrite(result_path_t, img_squeezed_t * 255)
        
    counter += 1
    save_as_counter += 1

