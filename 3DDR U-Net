import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imshow
from natsort import natsorted


#################### set variable
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1
LAGS = 5
# province = "Ubon"
# province = "Nakhonsawan"
# province = "Phranakorn"
province = "Metropolitan"
TRAIN_PATH = 'X_train/128/Data_%s_128/'%(province)
train_ids = natsorted(os.listdir(TRAIN_PATH))
X_train = np.zeros((len(train_ids),LAGS, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
Y_train = np.zeros((len(train_ids)-LAGS,1, IMG_HEIGHT, IMG_WIDTH,1))




#################### normalize
def normalize(data, maxVal=None, minVal=None):
    # Normalize with min/max values of the data
    if maxVal is None or minVal is None: 
        data = 0.1+((data-np.min(data))*(1-0.1))/(np.max(data)-np.min(data))
    # Normalize with max and min given
    else:
        data = 0.1+((data-minVal)*(1-0.1))/(maxVal-minVal)
    data = np.nan_to_num(data, nan=0)
    return data
    
    
    
#################### เก็บ X_train, Y_train
import cv2 
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = TRAIN_PATH + id_
    img = imread(path,as_gray= True)
    # minVal = np.min(img_x)
    # maxVal = np.max(img_x)
    # img = normalize(img_x, maxVal, minVal)
    img = np.expand_dims(img, axis=-1)
    if n >= LAGS:
        Y_train[n-LAGS][0]= img
        # Y_train[n-3][0] = img
    for i in range(min(n+1, LAGS)):
        X_train[n-i][i] = img
        
        
        
        
#################### check size X_train, Y_train
import matplotlib.pyplot as plt
X_train = X_train[:len(train_ids)-LAGS]
#check shape
print(X_train.shape)
print(Y_train.shape)
#check image 
imshow(X_train[350][3]) #450
plt.show()
imshow(Y_train[200][0])
plt.show()



################### model 

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Dropout, concatenate, BatchNormalization, Activation
from tensorflow.keras.layers import Conv3D, MaxPool3D, UpSampling3D
def UNet_3DDR():
    lags = 5
    filters = 4
    dropout = 0.5 #0.5
    kernel_init=tf.keras.initializers.GlorotUniform(seed=50)
    features_output = 1
    inputs = Input(shape = (lags, 128, 128, 1)) 
    conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputs)
    conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv1)
    pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)

    conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool1)
    conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv2)
    pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)

    conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool2)
    conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv3)
    pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)

    conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool3)
    conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv4)
    drop4 = Dropout(dropout)(conv4)

    #--- Bottleneck part ---#
    pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
    conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
    compressLags = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv5)
    conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(compressLags)
    drop5 = Dropout(dropout)(conv5)

    #--- Expanding part / decoder ---#
    up6 = Conv3D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(drop5))
    compressLags = Conv3D(8*filters, (lags,1,1),activation = 'relu', padding = 'valid')(drop4)
    merge6 = concatenate([compressLags,up6], axis = -1)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv6)

    up7 = Conv3D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv6))
    compressLags = Conv3D(4*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv3)
    merge7 = concatenate([compressLags,up7], axis = -1)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv7)

    up8 = Conv3D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv7))
    compressLags = Conv3D(2*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv2)
    merge8 = concatenate([compressLags,up8], axis = -1)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv8)

    up9 = Conv3D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv8))
    compressLags = Conv3D(filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv1)
    merge9 = concatenate([compressLags,up9], axis = -1)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    conv9 = Conv3D(2*features_output, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)

    conv10 = Conv3D(features_output, 1, activation = 'relu')(conv9) #Reduce last dimension    

    return Model(inputs = inputs, outputs = conv10)
    
    
    
    
    
#################### model summary + model train
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
model = UNet_3DDR()
filepath="model_%s_5_deno.hdf5"%(province)
# model = load_model(filepath, compile=False)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
callbacks = [checkpoint]
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
history1 = model.fit(X_train, Y_train, epochs=30)
# history2 = model.fit(X_train, Y_train, epochs=30)
# history3 = model.fit(X_train, Y_train, epochs=30)
# history4 = model.fit(X_train, Y_train, epochs=30)
# loss = history.history['loss']
# mae = history.history['mse']
# rmse = history.history['rmse']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'b', label='Training Loss')
# plt.title('Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()



##################### test
TEST_PATH = 'X_test/test_Phranakorn/Flood_1/x_test/'

test_ids = natsorted(os.listdir(TEST_PATH))
# ytest_ids = os.listdir('X_test/6/y_train/')
X_test = np.zeros((1,5, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
# print(ytest_ids)
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):  
    path = TEST_PATH + id_
    print(path)
    img = imread(path,as_gray= True)
    img = np.expand_dims(img, axis=-1)
    X_test[0][n] = img
img_predict = model.predict(X_test)

#################### predict
import matplotlib.pyplot as plt
print(np.squeeze(img_predict).shape)
# plt.imshow(img_predict[0, :, :,0], cmap='gray')
# (img_predict > 0.5).astype(np.uint8)
plt.imshow(np.squeeze(img_predict),cmap="gray")
# plt.imshow(img_predict)
print(Y_train.shape)
print(X_train.shape)
print(img_predict.shape)
model.summary()



################## PSNR,MSE,SSIM
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from skimage import io
img_orig = imread("Predict/%s/%s_%s"%(province,province,date),as_gray= True)
# img_orig = imread('Predict/Ubon/Ubon_%s'%(date),as_gray= True)
# img_orig = imread('test8.jpg',as_gray= True)
img_compressed = imread('X_test/test_Phranakorn/Flood_1/key_x_test/%s'%(date),as_gray= True)
# img_compressed = imread('Seen_x_test/Ubon/Flood_5/key_x_test/%s'%(date),as_gray= True)
# img_compressed = imread('X_test/1/x_train/2019_9_5.jpg',as_gray=True)
print(img_orig.shape)
print(img_compressed.shape)
imshow(img_orig)
plt.show()
imshow(img_compressed)
plt.show()

psnr = peak_signal_noise_ratio(img_orig, img_compressed)
mse = np.mean((img_orig - img_compressed) ** 2)
ssim_score = ssim(img_orig, img_compressed, multichannel=True)
print(ssim_score)
print(mse)
print(psnr)
    
