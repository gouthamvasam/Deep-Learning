# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 21:28:00 2020

@author: gvasam
"""


# Importing the libraries
#import matplotlib.pyplot as plt
import pylab as pl

import numpy as np
from keras.models import load_model
#Set the `numpy` pseudo-random generator at a fixed value
#This helps with repeatable results everytime you run the code. 
np.random.seed(42)

from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
#from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
#from keras.layers import BatchNormalization #, Dropout #(Dropout, a form of regularization is making training harder)
tf.__version__

# Load model
model = load_model('PE2_classification_CNN8_SVM_DA_data.h5')
# summarize model
model.summary()
# Load previously saved best weights
model.load_weights('weights-improvement-242-0.90.hdf5')


# Evaluating test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(
        'PE2_SVM_Divided_TT_70-30_8DA_1image/test',
        target_size=(600, 600),
        batch_size=32,
        class_mode=None,  # only data, no labels
        shuffle=False)  # keep data in same order as labels

probabilities = model.predict_generator(test_set, 1464)

# Compute the confusion matrix based on the label predictions
# For example, compare the probabilities with the case that there are 233 MVM_neg and 233 MVM_pos respectively in 466 images.

y_true = np.array([0] * 696 + [1] * 768)
y_pred = probabilities > 0.5

def showconfusionmatrix(cm):
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.show()

cm = confusion_matrix(y_true, y_pred)
print (cm)

showconfusionmatrix(cm)

classification_report(y_true, y_pred)
