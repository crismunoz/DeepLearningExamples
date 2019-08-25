import numpy as np
import pandas as pd
import os
from utils import get_loss,binary_crossentropy_with_logits,weighted_cross_entropy,get_mean_iou
from model import Deeplabv3
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from ImageDataGen import ImageDataGenerator as IDG
from keras import backend as K
import keras
import tensorflow as tf
from glob import glob
import shutil
import horovod.keras as hvd

hvd.init()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))



MODEL='model'
model_path = os.path.join('checkpoints', MODEL + '.model')
model_checkpoint = ModelCheckpoint(model_path, monitor='val_mean_iou',mode = 'max', save_best_only=True, verbose=1, period=1)
earlyStopping = EarlyStopping(monitor='val_mean_iou', mode = 'max',patience=6, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_mean_iou', mode = 'max',factor=0.5, patience=4, min_lr=0.0000001, verbose=1)

np.random.seed(0)
img_rows=256
img_cols=1600
num_channels = 3
N_CLASSES=5

cv_path ='crossvalid'
batchsize=4
print('batchsize:',batchsize)

data_gen_args = dict(height_shift_range=0.05,
                     width_shift_range=0.05,
                     rotation_range=30,
                     horizontal_flip=True,
                     vertical_flip=True)
image_datagen = IDG(rescale=1.,ismask=False,**data_gen_args)
mask_datagen = IDG(rescale=1.,ismask=True,**data_gen_args, nClasses=N_CLASSES)
image_datagen_val = IDG(rescale=1.,ismask=False)
mask_datagen_val  = IDG(rescale=1.,ismask=True, nClasses=N_CLASSES)
   

opt = keras.optimizers.Adam(lr=0.001 * hvd.size())
opt = hvd.DistributedOptimizer(opt)

seed = 1
k=10
for ind in range(k):    
    train_group_images_path = os.path.join(cv_path, 'group_'+str(ind),'train','images')
    train_group_masks_path = os.path.join(cv_path, 'group_'+str(ind),'train','masks')
    
    test_group_images_path = os.path.join(cv_path, 'group_'+str(ind),'test','images')
    test_group_masks_path = os.path.join(cv_path, 'group_'+str(ind),'test','masks')
    
    
    image_generator = image_datagen.flow_from_directory(train_group_images_path, target_size=(img_rows,img_cols),
                                                            class_mode=None, seed=seed, batch_size=batchsize)
    mask_generator = mask_datagen.flow_from_directory(train_group_masks_path, target_size=(img_rows,img_cols),
                                                           class_mode=None, seed=seed, batch_size=batchsize, 
                                                      color_mode='grayscale')

    image_generator_val = image_datagen_val.flow_from_directory(test_group_images_path, target_size=(img_rows,img_cols),
                                                            class_mode=None, seed=seed, batch_size=batchsize)
    mask_generator_val = mask_datagen_val.flow_from_directory(test_group_masks_path, target_size=(img_rows,img_cols),
                                                           class_mode=None, seed=seed, batch_size=batchsize, 
                                                              color_mode='grayscale')

    train_generator = zip(image_generator, mask_generator)
    val_generator = zip(image_generator_val, mask_generator_val)
    
    steps_per_epoch = image_generator.__len__()
    validation_steps = image_generator_val.__len__()
   
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),
        keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1),
    ]

    if hvd.rank() == 0:
        callbacks.append(keras.callbacks.ModelCheckpoint('logdir/checkpoint-{}.h5'.format(ind)))
    
    model = Deeplabv3(input_shape=(img_rows, img_cols, num_channels),classes=N_CLASSES) 
    #weights, w = get_weights(None, N_CLASSES, perPixel=0, enable=False)
    weights=np.ones(5)
    weights = K.variable(weights)
    model.compile(optimizer='adam', loss=get_loss(weights), metrics=['accuracy', get_mean_iou(N_CLASSES)])

    print('steps_per_epoch:',steps_per_epoch// hvd.size())
    results = model.fit_generator(train_generator,
                               steps_per_epoch=steps_per_epoch// hvd.size(),
                               validation_data=val_generator,
                               validation_steps=validation_steps// hvd.size(),
                               epochs=50,
                               verbose=1,
                               callbacks=callbacks,
                               shuffle=True)
    