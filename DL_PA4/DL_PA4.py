import tensorflow as tf
import pandas as pd
import numpy as np
import os
import keras
from keras.preprocessing import image


train_img_dr, test_img_dr = './cifar10_data/train/image', './cifar10_data/test/image'

n_train, n_test = len(os.listdir(train_img_dr)), len(os.listdir(test_img_dr))

x_train, x_test = np.zeros((n_train,32,32,3)), np.zeros((n_test,32,32,3))

for i in range(n_train):
    img_name = str(i)+'.jpg'
    imag_path = os.path.join(train_img_dr,img_name)
    imag = image.load_img(imag_path)
    img_tensor = image.img_to_array(imag)
    x_train[i]=img_tensor

for i in range(n_test):
    img_name = str(i)+'.jpg' ## 이미지 파일 전처리
    imag_path = os.path.join(test_img_dr,img_name)
    imag = image.load_img(imag_path)
    img_tensor = image.img_to_array(imag)
    x_test[i]=img_tensor

x_train, x_test = x_train/255, x_test/255


y_train = pd.read_csv('./cifar10_data/train/label.csv',header = None, encoding='utf-8')
y_test = pd.read_csv('./cifar10_data/test/label.csv',header = None, encoding='utf-8')


## 모델을 좀 더 간편하게 설정하기 위해 블럭 단위 구성을 위한 함수를 작성
def vgg_block(in_layer, n_conv, n_filter, filter_size=(3, 3), reduce_size=True):
    layer = in_layer
    for i in range(n_conv):
        layer = tf.keras.layers.Conv2D(n_filter, filter_size, padding='SAME')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer) ##batch normalization
        layer = tf.keras.layers.Activation(activation='relu')(layer)

    if reduce_size:
        layer = tf.keras.layers.MaxPool2D((2, 2))(layer)
    return layer


input_layer = tf.keras.layers.Input(shape=(32, 32, 3))
vgg_block01 = vgg_block(input_layer, 2, 32) # 16x16x32
vgg_block02 = vgg_block(vgg_block01, 2, 64) # 8x8x64
vgg_block03 = vgg_block(vgg_block02, 3, 128) # 4x4x128

flatten = tf.keras.layers.Flatten()(vgg_block03) # 2048
dense01 = tf.keras.layers.Dense(512, activation='relu')(flatten)
dropout = tf.keras.layers.Dropout(rate = 0.2)(dense01) ## dropout
output = tf.keras.layers.Dense(10, activation='softmax')(dropout)

model = tf.keras.models.Model(input_layer, output)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=100, epochs=30,
          validation_data=(x_test, y_test))

#### test
pred = model.predict(x_test)
print(len(pred))
for i in range(len(pred)):
    print( "predict= {}, label= {}".format(np.argmax(pred[i]),y_test[0][i]))

