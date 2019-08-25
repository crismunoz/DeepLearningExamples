#from matplotlib import pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import cv2
import argparse
import numpy as np
import tensorflow as tf
from model import Deeplabv3
from keras import optimizers
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import gridspec
from keras.utils import to_categorical
#from DeepLabUtils import vis_segmentation,LoadImage,LoadUrlImage,LoadData,vis_segmentationTGS
from ImageDataGen import ImageDataGenerator as IDG
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from glob import glob
from numpy import linalg as la
#from keras_radam import RAdam
tf.logging.set_verbosity(tf.logging.ERROR)

#deeplabDA  
#deeplabDA2 20
#deeplabDA3 100

def get_loss(weights):
    def loss(y_true, y_pred):
        #totalClass1 = np.zeros(N_CLASSES)
        #totalClass2 = np.zeros(N_CLASSES)
        #for i in range(N_CLASSES+1):
        # Get number of pixels per class
        #temp = np.where(img == i)
        #nPixels = len(temp[0])
        # Print number
        #if nPixels > 0:
        #  totalClass[i] += 1
        
        
        #pos_neg = K.cast(K.sum(K.reshape(y_true,[-1, N_CLASSES]),axis=-1)>0,'int32')
        #print (pos_neg)
        #pos_neg_oh = K.one_hot(pos_neg,N_CLASSES)
        #print (pos_neg_oh)
    
            #weights2 = K.sum(pos_neg_oh*K.reshape(weights,[-1,N_CLASSES]),axis=1)
        # scale predictions so that the class probas of each sample sum to 1
        #y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        #y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
            #loss = y_true * K.log(y_pred) * weights2
        #else:
        #nclass=4
        #y_true = K.one_hot(K.cast(y_true,tf.int32), nclass)
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss

def binary_crossentropy_with_logits(ground_truth, predictions):    
    return K.mean(K.binary_crossentropy(ground_truth, predictions,from_logits=True),axis=-1)

#---------------------------------------------------------------------------------------------
# weighted_cross_entropy uses weighted_cross_entropy_with_logits function for unbalanced data
# we assign a weight on the positive target with pos_weight, in this case is been used weights
def weighted_cross_entropy(weights):
    def loss(y_true, y_pred):
        loss = tf.nn.weighted_cross_entropy_with_logits(targets=y_true, logits=y_pred, pos_weight=weights)
        return loss
    return loss

#---------------------------------------------------------------------------------------------
def get_mean_iou(nclasses):
    def mean_iou(y_true, y_pred):
        print(y_true.shape)
        print(y_pred.shape)
        
        #y_true = K.squeeze(y_true,3)
        #y_true = K.one_hot(K.cast(y_true, tf.int32), nclasses)
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            y_pred_ = tf.to_int32(y_pred > t)
            score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, nclasses)
            #config = tf.ConfigProto()
            #config.gpu_options.per_process_gpu_memory_fraction = 0.8
            #set_session(tf.Session(config=config))
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([up_opt]):
                score = tf.identity(score)
            prec.append(score)
        return K.mean(K.stack(prec), axis=0)
    return mean_iou

#---------------------------------------------------------------------------------------------


def get_weights(directory, classes, perPixel, enable):
  # Create totalClass array
  if enable:
    totalClass = np.zeros(classes)
    totalPixels = np.zeros(classes)
    totalWeights = np.zeros(classes)
    # Set path to files
    masks_path = os.path.join(directory, 'masks', 'masks')
    image_pattern = os.path.join(masks_path, '*.png')
    image_lst = glob(image_pattern)
    n_images = len(image_lst)    
    print (n_images)
    j=0
    for image in image_lst:
      j += 1
      # Read image
      img = cv2.imread(image, 0)
      # Get number of class
      nClasses = img.max()
      for i in range(nClasses+1):
        # Get number of pixels per class
        temp = np.where(img == i)
        nPixels = len(temp[0])
        # Print number
        if nPixels > 0:
          totalClass[i] += 1
          totalPixels[i] += nPixels
    
    # Number of classes detected in total
    totalClassDetected = len(np.where(totalClass>=1)[0])
    totalDetected = np.where(totalClass>=1)[0]
    totalClass = np.where(totalClass>=1, totalClass, 0)
    totalPixels = np.where(totalPixels>=1, totalPixels, 0)
    temp = []
    for i in totalDetected:
      if perPixel:
        temp.append(totalPixels[i])
      else:
        temp.append(totalClass[i])
    temp = np.asarray(temp)
    normT = la.norm(temp)
    weights = normT/temp

    c = 0
    for i in totalDetected:
      totalWeights[i] = weights[c]
      c += 1

    print ('Total classes detected:', totalClassDetected)
    print ('Detected classes:', totalDetected)
    if perPixel:
      return totalWeights, totalPixels
    else:
      return totalWeights, totalClass
  else:
    totalWeights = np.ones(classes)
    return totalWeights, 0