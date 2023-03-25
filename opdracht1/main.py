import pandas as pd
import cv2
import numpy as np
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image

test_data = pd.read_csv("smoke\\test\\_annotations.csv")
train_data = pd.read_csv("smoke\\train\\_annotations.csv")
valid_data = pd.read_csv("smoke\\valid\\_annotations.csv")

height = 224
width = 224

og_height = 480
og_width = 640

def load(path):
    img = load_img(path, target_size=(height, width))
    img = img.resize((224,224))
    img = img_to_array(img)
    img = img / 255.0
    return img

def normalize(img):
    return cv2.resize(img, (og_width, og_height))

def CheckClass(label):
    if label == 0:
        return 'Smoke'
    else:
        return 'No Smoke'

train_img = np.array([cv2.imread('smoke\\train\\' + filename) for filename in train_data['filename']])
train_img = np.array([cv2.resize(img, (height, width)) for img in train_img])
train_name = np.array([filename for filename in train_data['filename']])
train_bound = train_data[['xmin', 'ymin', 'xmax', 'ymax']].values
train_labels = train_data['class'].map({'smoke':0, 'no_smoke':1}).values

test_img = np.array([cv2.imread('smoke\\test\\' + filename) for filename in test_data['filename']])
test_img = np.array([cv2.resize(img, (height, width)) for img in test_img])
test_name = np.array([filename for filename in test_data['filename']])
test_bound = test_data[['xmin', 'ymin', 'xmax', 'ymax']].values
test_labels = test_data['class'].map({'smoke':0, 'no_smoke':1}).values

valid_img  = np.array([cv2.imread('smoke\\valid\\' + filename) for filename in valid_data['filename']])
valid_img = np.array([cv2.resize(img, (height, width)) for img in valid_img])
valid_name = np.array([filename for filename in valid_data['filename']])
valid_bound = valid_data[['xmin', 'ymin', 'xmax', 'ymax']].values
valid_labels = valid_data['class'].map({'smoke':0, 'no_smoke':1}).values

train_cls = to_categorical(train_labels, 2)
test_cls = to_categorical(test_labels, 2)
valid_cls = to_categorical(valid_labels, 2)

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
    "bounding_box": "mse",
}

lossWeights = {
    "class_label": 1.0,
    "bounding_box" : 1.0
}

opt = Adam()

vgg = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))
vgg.trainable = False
flatten = Flatten()(vgg.output)

bbox_head = Dense(128, activation='relu') (flatten)
bbox_head = Dense(64, activation='relu') (bbox_head)
bbox_head = Dense(32, activation='relu') (bbox_head)
bbox_head = Dense(4, activation='linear', name = 'bounding_box') (bbox_head)

softmax_head = Dense(512, activation='relu')(flatten)
softmax_head = Dropout(0.5)(softmax_head)
softmax_head = Dense(512, activation='relu')(softmax_head)
softmax_head = Dropout(0.5)(softmax_head)
softmax_head = Dense(2, activation='softmax', name="class_label")(softmax_head)

model = Model( inputs = vgg.input, outputs = (softmax_head, bbox_head))

model.compile(loss= losses, optimizer=opt,metrics=["accuracy"], loss_weights=lossWeights)

train_labels = train_labels.astype('float32')
val_labels = valid_labels.astype('float32')

model.fit(train_img, trainTargets, epochs = 1, validation_data=(test_img, testTargets))

res = model.evaluate(test_img, testTargets)
print('Test loss:', res[0])
print('Test accuracy:', res[1])

predLab, predBox = model.predict(valid_img)

for i in range(len(valid_img)):
    fig, ax = plt.subplots()
    img = valid_img[i]
    img = normalize(img)
    label = np.argmax(predLab, axis=-1)
    ax.imshow(img)
    ax.axis('off')
    xmin, ymin, xmax, ymax = predBox[i]
    ax.text(xmin, ymin, CheckClass(valid_labels[0]) , fontsize=10, color='black')
    plt.title(f'Figure {i}')
    box = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor='red')
    ax.add_patch(box)
    plt.show()
