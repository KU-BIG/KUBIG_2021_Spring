import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers, models
import cv2
import keras
import sklearn
# https://stackoverflow.com/questions/54594388/how-to-set-shape-format-of-a-targety-array-for-multiple-output-regression-in-k
def someMethodToLoadImages(files):
    images =  []
    for file in files:
         img = keras.preprocessing.image.load_img(file, target_size=(128, 128))
         img = keras.preprocessing.image.img_to_array(img)
         images.append(img)
    images = np.array(images).reshape(len(files), 128, 128, 3) # convert the image size as 128 x 128 for faster learning
    return images/255.0

# https://stackoverflow.com/questions/47200146/keras-load-images-batch-wise-for-large-dataset
def imageLoader(files, batch_size, idx):
    L = len(files)
    # this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X = someMethodToLoadImages(files[batch_start:limit])
            Y = np.array(label.iloc[idx, :])[batch_start:limit]

            yield (X,Y) # a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size
# simple model
input = layers.Input(shape=(128, 128, 3))
x = layers.Conv2D(32, (3,3), padding='valid', activation='relu')(input)
x = layers.BatchNormalization()(x)
x = layers.MaxPool2D((2,2))(x)
x = layers.Dropout(0.3)(x)

x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPool2D((2,2))(x)
x = layers.Dropout(0.3)(x)

x = layers.Conv2D(128, (3,3), activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPool2D((2,2))(x)
x = layers.Dropout(0.3)(x)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(64, activation='relu')(x)
output = layers.Dense(26, activation='sigmoid')(x)

model1 = models.Model([input], [output])
model1.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss = 'binary_crossentropy', metrics=['accuracy'])

# resnet50
from keras.applications.resnet import  ResNet50
resnet = ResNet50(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
# unfreeze some layers
for layer in resnet.layers[:150]:
   layer.trainable = False
for layer in resnet.layers[150:]:
   layer.trainable = True
print(len(resnet.trainable_weights))

model2 = models.Sequential()
model2.add(resnet)
model2.add(layers.GlobalAveragePooling2D())
# use sigmoid for multi-label(assuming each classes are independent)
model2.add(layers.Dense(26, activation = 'sigmoid'))

import keras.backend as K
import tensorflow as tf

# define custom accuracy for multi-label classification(basic accuracy metric doesn't work for multi-label)
def score(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, 0.5), 'float32')
    equal = K.sum(K.cast(K.equal(y_true, y_pred), 'int32'))
    return equal/tf.size(y_true)

# use binary cross-entropy
model2.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss = 'binary_crossentropy', metrics=['accuracy', score])

### 규선님 여기는 label 파일 불러오는 경로입니다!
label = pd.read_csv('D:/multilabel/dirty_mnist_2nd_answer.csv')

filename = []
for i in range(len(label)):
   name = "{:0>5}".format(str(label['index'][i]))
   name = name+'.png'
   filename.append(name)


label = label.iloc[:, 1:]
### 규선님 여기는 이미지 파일 directory인데 img파일 들어있는 폴더까지만 해주시면 됩니다!
imgs_dir = np.array('D:/multilabel/dirty_mnist_2nd/' + pd.Series(filename))

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds=[]
for train_idx, valid_idx in kf.split(imgs_dir):
    folds.append((train_idx, valid_idx))

# use only one fold for training
train_idx = folds[0][0]
valid_idx = folds[0][1]

# early-stopping to avoid overfitting
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_score', patience = 5)

history = model2.fit(imageLoader(imgs_dir[train_idx],100, train_idx),
                     steps_per_epoch = len(train_idx)//100, epochs = 20,
                     callbacks=[es], validation_data=imageLoader(imgs_dir[valid_idx], 50, valid_idx),
                     validation_steps = len(valid_idx) //50)

### 규선님 여기는 모델 저장하는 부분인데 이름만 그대로 해주시고 모델 저장해주시면 감사하겠습니다!! ㅠㅠ
model2.save('D:/multilabel/resnet50_transfer.h5')