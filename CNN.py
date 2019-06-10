# -*- coding: utf-8 -*-
"""
Created on Thu May 30 19:40:25 2019
@author: huangzy97
"""
# 卷积神经网络 
# part 1
from  keras.layers  import Convolution2D
from  keras.models import Sequential
from  keras.layers import MaxPool2D
from  keras.layers import Flatten
from  keras.layers import Dense
# 初始化一个分类器 
classifier = Sequential()
# 添加卷积层
classifier.add(Convolution2D(32,3,3,activation = 'relu',input_shape = (64,64,3)))
# 添加池化层
classifier.add(MaxPool2D(pool_size = (2,2)))
# =============================================================================
# ###添加卷积层
classifier.add(Convolution2D(32,3,3,activation = 'relu'))
classifier.add(MaxPool2D(pool_size = (2,2)))
# =============================================================================

# 添加扁平层
classifier.add(Flatten())
# 添加ANN隐藏层
classifier.add(Dense(output_dim = 128,activation = 'relu'))
# 添加输出层
classifier.add(Dense(output_dim = 1,activation = 'sigmoid'))
#  编译Cnn
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy' ,metrics = ['accuracy'])
 
# part2 输入图片预处理
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                            'dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=62.5)
