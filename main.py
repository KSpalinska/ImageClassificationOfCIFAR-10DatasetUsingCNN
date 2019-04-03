"""

MGU - Projekt nr 2
Klasyfikacja obrazów przy użyciu konwolucyjnych sieci neuronowych

"""

## Ładowanie bibliotek i pakietów
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization


## Ładowanie zbiorów danych
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

n_train, img_row, img_col, img_ch = x_train.shape
n_test = x_test.shape[0]
categories = np.unique(y_train)
n_categories = len(categories)

