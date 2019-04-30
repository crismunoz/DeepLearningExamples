from keras.models import model_from_json
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

def SaveModel(model,model_name='model'):
    model_json = model.to_json()
    open(model_name+'.json','w').write(model_json)
    model.save_weights(model_name+'_Pesos.hdf5')
    
def LoadModel(model_name=''):
    # Carregare arquivo json 
    json_file = open(model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # Carregar pesos de um arquivo
    model.load_weights(model_name+'_Pesos.hdf5')   
    return model

def Get_data():
    """Carregar os dados
        usar função load_data()
    """
    path = os.getcwd()
    (X_train, y_train), (X_test, y_test)=mnist.load_data(path=os.path.join(path, 'data/mnist.npz'))

    #Re-dimensionar os dados 
    X_train = X_train.reshape((X_train.shape[0],28,28,1))
    X_test  = X_test.reshape((X_test.shape[0],28,28,1))
        
    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')    
    return (X_train, y_train), (X_test, y_test)
    
# crear uma grafica de 3x3
def PlotDigits(X):
    plt.figure(figsize=(10,10))
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(X[i].reshape(28,28), cmap=plt.get_cmap('gray'))
        plt.axis('off')
    plt.show()
def PlotImage(X):
    plt.figure(figsize=(10,10))
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(array_to_img(X[i].reshape(X[i].shape[0:])))
        plt.axis('off')
    plt.show()     