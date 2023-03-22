import pandas as pd
import cv2
import numpy as np
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

test_data = pd.read_csv("smoke\\test\\_annotations.csv")
train_data = pd.read_csv("smoke\\train\\_annotations.csv")
valid_data = pd.read_csv("smoke\\valid\\_annotations.csv")

height = 224
width = 224

def load(path):
    img = load_img(path, target_size=(height, width))
    img = img_to_array(img)
    img = img / 255.0
    return img

train_img = np.array([load('smoke\\train\\' + filename) for filename in train_data['filename']])
train_bound = train_data[['xmin', 'ymin', 'xmax', 'ymax']].values
train_labels = train_data['class'].map({'smoke':1, 'no_smoke':0}).values

test_img = np.array([load('smoke\\test\\' + filename) for filename in test_data['filename']])
test_bound = test_data[['xmin', 'ymin', 'xmax', 'ymax']].values
test_labels = test_data['class'].map({'smoke':1, 'no_smoke':0}).values

valid_img  = np.array([load('smoke\\valid\\' + filename) for filename in valid_data['filename']])
valid_bound = valid_data[['xmin', 'ymin', 'xmax', 'ymax']].values
valid_labels = valid_data['class'].map({'smoke':1, 'no_smoke':0}).values

train_cls = to_categorical(train_labels, 2)
test_cls = to_categorical(test_labels, 2)
valid_cls = to_categorical(valid_labels, 2)

# train_cls_reg = np.concatenate([train_cls, train_bound], axis=1)
# test_cls_reg = np.concatenate([test_cls, test_bound], axis=1)
# valid_cls_reg = np.concatenate([valid_cls, valid_bound], axis=1)

trainTargets = {
    "class_label": train_cls,
    "bounding_box": train_bound
}

testTargets = {
    "class_label": test_cls,
    "bounding_box": test_bound
}

validTargets = {
    "class_label": valid_cls,
    "bounding_box": valid_bound
}

losses = {
    "class_label": "categorical_crossentropy",
    "bounding_box": "mean_squared_error",
}

lossWeights = {
    "class_label": 1.0,
    "bounding_box" : 1.0
}

print(trainTargets)

opt = Adam()

vgg = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))
vgg.trainable = False
flatten = Flatten()(vgg.output)

bbox_head = Dense(128, activation='relu') (flatten)
bbox_head = Dense(64, activation='relu') (bbox_head)
bbox_head = Dense(32, activation='relu') (bbox_head)
bbox_head = Dense(4, activation='sigmoid', name = 'bounding_box') (bbox_head)

softmax_head = Dense(512, activation='relu')(flatten)
softmax_head = Dropout(0.5)(softmax_head)
softmax_head = Dense(512, activation='relu')(softmax_head)
softmax_head = Dropout(0.5)(softmax_head)
softmax_head = Dense(2, activation='softmax', name="class_label")(softmax_head)

model = Model( inputs = vgg.input, outputs = (bbox_head, softmax_head))

model.compile(loss= losses, optimizer=opt,metrics=["accuracy"], loss_weights=lossWeights)

train_labels = train_labels.astype('float32')
val_labels = valid_labels.astype('float32')

model.fit(train_img, trainTargets, epochs = 10, validation_data=(valid_img, validTargets))

test_loss, test_acc = model.evaluate(test_img, test_labels,verbose=2)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

predictions = model.predict(test_img[:10])

fig, axs = plt.subplots(2,5,figsize=(15,6))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()
for i in range(10):
    axs[i].imshow(test_img[i])
    axs[i].axis('off')

plt.show()