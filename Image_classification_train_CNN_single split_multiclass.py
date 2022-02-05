# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:50:19 2020

@author: gouthamvasam
"""

"""
All section images resized to 256 x 256.
Images used for test are completely different that the ones used for training.
199, 201 and 83 validation images for Ctrl, PE_preterm and PE_term, respectively.
210, 204 and 79 test images for Ctrl, PE_preterm and PE_term, respectively.
944, 907 and 363 train images for Ctrl, PE_preterm and PE_term, respectively.
"""

# Convolutional Neural Network

# Importing the libraries
import matplotlib.pyplot as plt

import numpy as np
#Set the `numpy` pseudo-random generator at a fixed value
#This helps with repeatable results everytime you run the code. 
np.random.seed(42)

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization #, Dropout #(Dropout, a form of regularization is making training harder)
#import gc
#from keras.utils import np_utils
#from tf.keras.utils import to_categorical
tf.__version__

#Try Adam optimizer learning rate 0.0001 and batch_size 32 or 64 in the next try to reduce validation accuracy fluctuations
custom_optimizer = tf.keras.optimizers.Adam(lr = 0.0001)

# Part 1 - Data Preprocessing

#Used 600 pixel size. Try the original 1200 pixel size when you have enough time and decided the model code.

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   rotation_range = 30,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   brightness_range = [0.8, 1.2],
                                   channel_shift_range = 150.0,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   zca_whitening = True,
                                   fill_mode = 'reflect') #Try fill modes, e.g. nearest (default), reflect, constant, wrap
training_set = train_datagen.flow_from_directory('CNN_onset/train_O_70',
                                                 target_size = (600, 600),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('CNN_onset/val_15',
                                            target_size = (600, 600),
                                            batch_size = 32,
                                            class_mode = 'categorical')


#One-hot encoding (for multilcass classification)
#num_classes = 3
#train_labels = tf.keras.utils.to_categorical(training_set, 3)
#test_labels = tf.keras.utils.to_categorical(test_set, 3)


#Add checkpoints 
from keras.callbacks import ModelCheckpoint, CSVLogger #EarlyStopping might be another good callback - probably won't work for our dataset.
#filepath='PE_classification_CNN5.h5'
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5" #File name includes epoch and validation accuracy.
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=[600, 600, 3]))

# Step 2 - Pooling           
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(BatchNormalization(axis = -1))
#cnn.add(Dropout(0.2))

# Adding a second set of convolutional and pooling layers
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation='relu', padding='same'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(BatchNormalization(axis = -1))
#cnn.add(Dropout(0.2))



# Adding a third set of convolutional and pooling layers
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=9, activation='relu', padding='same'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(BatchNormalization(axis = -1))
#cnn.add(Dropout(0.2))



# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())


# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))
cnn.add(BatchNormalization(axis = -1))
#cnn.add(Dropout(0.2))

cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
cnn.add(BatchNormalization(axis = -1))
#cnn.add(Dropout(0.2))

cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(BatchNormalization(axis = -1))
#cnn.add(Dropout(0.2))

cnn.add(tf.keras.layers.Dense(units=64, activation='relu'))
cnn.add(BatchNormalization(axis = -1))
#cnn.add(Dropout(0.2))

cnn.add(tf.keras.layers.Dense(units=32, activation='relu'))
cnn.add(BatchNormalization(axis = -1))
#cnn.add(Dropout(0.2))



# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=3, activation='softmax'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = custom_optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
history = cnn.fit(x = training_set, validation_data = test_set, batch_size = 64, verbose = 1, epochs = 280, callbacks = callbacks_list)
#try batch_size = 8 or 16 to converge faster but it may take long time to train. Default is 32 I guess.


#CSVLogger logs epoch, acc, loss, val_acc, val_loss
log_csv = CSVLogger('my_logs.csv', separator=',', append=False)

#Plots
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

max_epoch = len(history.history['accuracy'])+1
epoch_list = list(range(1,max_epoch))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, max_epoch, 25))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, max_epoch, 25))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")

# evaluating the model
train_loss, train_acc = cnn.evaluate(training_set)
validation_loss, test_acc = cnn.evaluate(test_set)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


# plot training history
print("Values stored in history are ... \n", history.history)
#plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')
#plt.legend()
#plt.show()

#Save the model
cnn.save('PE_classification_CNN_onset.h5')