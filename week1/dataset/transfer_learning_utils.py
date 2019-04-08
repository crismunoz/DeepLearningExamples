from keras.datasets import mnist
from keras import utils
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

mnist_path = ""#os.path.join(os.getcwd(),'data/mnist.npz')
def show_mnist_groups_dataset():
    if os.path.isfile(mnist_path):
        (X_train, y_train), (X_test, y_test) = mnist.load_data(mnist_path)
    else:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
	#path=os.path.join(os.getcwd(), 'data/mnist.npz'))
    X_train=X_train.reshape([-1,28,28,1])
    X_test=X_test.reshape([-1,28,28,1])

    lt5_x_train, lt5_y_train = X_train[y_train<5], y_train[y_train<5]
    lt5_x_test, lt5_y_test = X_test[y_test<5],  y_test[y_test<5]

    gt5_x_train, gt5_y_train = X_train[y_train>=5],  y_train[y_train>=5]
    gt5_x_test, gt5_y_test = X_test[y_test>=5],  y_test[y_test>=5]


    plt.figure(figsize=(15,8))
    img_index = list(range(100))[:32]
    for x in range(4):
        for y in range(4):
            plt.subplot(4,8,y+x*8+1)
            plt.imshow(lt5_x_train[img_index[y+x*8]][:,:,0], cmap='Reds')
            plt.axis('off')

        for y in range(4,8):
            plt.subplot(4,8,y+x*8+1)
            plt.imshow(gt5_x_train[img_index[y+x*8]][:,:,0], cmap='Blues')
            plt.axis('off')

def get_mnist_groups_dataset(group):
    if os.path.isfile(mnist_path):
        (X_train, y_train), (X_test, y_test) = mnist.load_data(mnist_path)
    else:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train=X_train.reshape([-1,28,28,1])
    X_test=X_test.reshape([-1,28,28,1])

    lt5_x_train, lt5_y_train = X_train[y_train<5], utils.to_categorical(y_train[y_train<5],5)
    lt5_x_test, lt5_y_test = X_test[y_test<5], utils.to_categorical(y_test[y_test<5], 5)

    gt5_x_train, gt5_y_train = X_train[y_train>=5],  utils.to_categorical(y_train[y_train>=5]-5,5)
    gt5_x_test, gt5_y_test = X_test[y_test>=5], utils.to_categorical(y_test[y_test>=5]-5,5)
    if group=='lt5':
        return (lt5_x_train, lt5_y_train) , (lt5_x_test, lt5_y_test)
    else:            
        return (gt5_x_train, gt5_y_train) , (gt5_x_test, gt5_y_test)
        
    
def PlotHistory(history):    
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def PlotFeatures(X,size_img=224):
    plt.figure(figsize=(10,10))
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        k=np.random.randint(X.shape[-1])
        plt.imshow(X[:,:,:,k].reshape(size_img,size_img), cmap=plt.get_cmap('gray'))
        plt.axis('off')
    plt.show()
	
	
def load_data(dataset_name, test_size):
  datagen = ImageDataGenerator()

  num_data = sum([len(files) for r, d, files in os.walk(dataset_name)])

  generator = datagen.flow_from_directory(dataset_name, 
                                          target_size=(224, 224), 
                                          batch_size=num_data, 
                                          shuffle=False)

  X , y = generator.next()

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  return (X_train,y_train) , (X_test,y_test)