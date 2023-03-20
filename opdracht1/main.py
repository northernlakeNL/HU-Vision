from keras.layers import Input, Lambda, Dense, Flatten, Dropout
from keras.models import Model
from keras.applications.vgg19 import VGG19, preprocess_inputs
from keras.preprocessing import image, ImageDataGenerator
from keras.models import Sequential
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt