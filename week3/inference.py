import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import argparse
import numpy as np
import pickle as pkl
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.python.platform import gfile
import sys
from inference_utils import *
import collections
from mainDeepLabV3 import get_loss,binary_crossentropy_with_logits,weighted_cross_entropy,get_mean_iou
from model import relu6, BilinearUpsampling

tf.logging.set_verbosity(tf.logging.ERROR)
from ImageDataGen import load_img

img_rows=256
img_cols=1600
N_CLASSES=5

def get_mask(image, cls, encodedPixels):
    shape = image[:,:,0].shape
    mask = np.zeros(shape).T.reshape([-1])
    blocks = encodedPixels.split()
    ng = len(blocks)#//2\n",
    for ig in range(0,ng,2):
        start = int(blocks[ig])
        end = start + int(blocks[ig+1])
        mask[start:end]=cls

    mask = mask.reshape([shape[1],shape[0]]).T
    return mask"
	
def videoInference(args):

	opt = keras.optimizers.Adam(lr=0.001 * hvd.size())
	opt = hvd.DistributedOptimizer(opt)
    weights=np.ones(5)
    weights = K.variable(weights)
	model_path = 'logdir/bkp/checkpoint-0.h5'
	model = load_model(model_path,custom_objects={'relu6':relu6,'BilinearUpsampling':BilinearUpsampling, 
	'loss': get_loss(weights), 
	'mean_iou': get_mean_iou(N_CLASSES), 
	"binary_crossentropy_with_logits":binary_crossentropy_with_logits})

	writer = open('results.txt','wb')
	
	files = glob('test/images/*.jpg')
	for file in files:
		img =  load_img(file, target_size=(img_rows , img_cols))
		img = img.reshape([1, img_rows, img_cols, 3])
		pred = model.predict(img)
		cls = np.argmax(pred,3)
		cls = cls.reshape([img_rows , img_cols,]).T
		
		for i in range(N_CLASSES):
			index = np.argwhere(cls==i)
			"" [ind  for ind in index]
			writer.write('\n')
		
		
		
	
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process video file.')
	parser.add_argument('--model', type=str, dest='modelname', help='Optimized model in .pb extension', metavar='model')
	parser.add_argument('--video', type=str, dest='filename', help='Filename to perform inference')
	parser.add_argument("--bs",	 dest='batchsize', type=int, help="Batchsize to perform the inference (1~4)")
	parser.add_argument("--out",  dest='outVideo', type=int, help="To generate video inference file (0 or 1)")
	parser.add_argument('--classes', type=int, help='Number of classes', metavar='classes')
	parser.add_argument('--fracmem', type=float, default=1.0, dest='fracmem', help='percent gpu memory use', metavar='fracmem')
	args = parser.parse_args()
		
	print("Starting inference .... ")
	videoInference(args)